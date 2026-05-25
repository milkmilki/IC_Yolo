from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def find_none_class_index(classes: list[str]) -> int | None:
    for index, name in enumerate(classes):
        if str(name).strip().lower() == "none":
            return index
    return None


def build_none_aware_sampler(
    targets: list[int],
    classes: list[str],
    none_sampling_ratio: float,
    seed: int,
) -> WeightedRandomSampler:
    target_tensor = torch.tensor(targets, dtype=torch.long)
    class_counts = torch.bincount(target_tensor, minlength=len(classes)).float().clamp(min=1.0)
    target_probs = torch.zeros(len(classes), dtype=torch.float32)
    none_index = find_none_class_index(classes)
    present = [index for index, count in enumerate(class_counts.tolist()) if count > 0]
    if none_index is None or none_index not in present or len(present) <= 1:
        target_probs[present] = 1.0 / max(len(present), 1)
    else:
        none_ratio = min(max(float(none_sampling_ratio), 0.05), 0.95)
        defect_indices = [index for index in present if index != none_index]
        target_probs[none_index] = none_ratio
        target_probs[defect_indices] = (1.0 - none_ratio) / max(len(defect_indices), 1)

    sample_weights = target_probs[target_tensor] / class_counts[target_tensor]
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    summary = ", ".join(
        f"{classes[index]}:{target_probs[index].item():.3f}"
        for index in range(len(classes))
        if target_probs[index] > 0
    )
    print(f"[sampling] none-aware target mix: {summary}")
    return WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(targets),
        replacement=True,
        generator=generator,
    )


class WaferToTensor:
    def __call__(self, image) -> torch.Tensor:
        array = np.array(image, copy=True)
        if array.ndim == 2:
            array = np.expand_dims(array, axis=-1)
        tensor = torch.from_numpy(array.transpose((2, 0, 1))).float()
        return tensor.div(255.0)


class IndexedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        return image, label, index


class LowRankYoloHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.proj(feats)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.linear(x)


class CTMBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.state_proj = nn.Linear(d_model, d_model)
        self.input_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([state, x], dim=-1)))
        delta = torch.tanh(self.state_proj(state) + self.input_proj(x))
        updated = state + gate * self.dropout(delta)
        return self.norm(updated)


class SharedCrossScanMixer(nn.Module):
    """Lightweight four-direction token propagation with shared scan parameters."""

    def __init__(self, d_model: int, scale_init: float = 0.05) -> None:
        super().__init__()
        self.retention = nn.Linear(d_model, 1)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def _scan(self, sequence: torch.Tensor) -> torch.Tensor:
        retention = torch.sigmoid(self.retention(sequence))
        state = torch.zeros_like(sequence[:, :, 0, :])
        outputs: list[torch.Tensor] = []
        for index in range(sequence.shape[2]):
            keep = retention[:, :, index, :]
            state = keep * state + (1.0 - keep) * sequence[:, :, index, :]
            outputs.append(state)
        return torch.stack(outputs, dim=2)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, length, channels = tokens.shape
        if length != height * width:
            raise ValueError(f"Token length {length} does not match spatial shape {height}x{width}")
        grid = tokens.reshape(batch, height, width, channels)
        left_to_right = self._scan(grid)
        right_to_left = torch.flip(self._scan(torch.flip(grid, dims=[2])), dims=[2])
        vertical = grid.transpose(1, 2)
        top_to_bottom = self._scan(vertical).transpose(1, 2)
        bottom_to_top = torch.flip(self._scan(torch.flip(vertical, dims=[2])), dims=[2]).transpose(1, 2)
        context = (left_to_right + right_to_left + top_to_bottom + bottom_to_top) * 0.25
        delta = (context - grid).reshape(batch, length, channels)
        return tokens + self.scale * self.proj(delta)


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, class_counts: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("log_counts", class_counts.clamp(min=1.0).log())

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.log_counts.to(logits.device), labels)


class YoloCTM(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        yolo_head: nn.Module,
        num_classes: int,
        in_dim: int,
        d_model: int = 256,
        steps: int = 4,
        dropout: float = 0.1,
        adapter_rank: int = 0,
        feature_adapter: bool = True,
        feature_fusion: str = "residual",
        gate_rank: int = 16,
        token_gate_rank: int = 16,
        logprob_fusion: bool = False,
        logprob_fusion_init: float = 0.2,
        expert_fusion: str = "none",
        expert_ctm_init: float = 0.4,
        spatial_encoding: str = "none",
        spatial_encoding_scale_init: float = 0.05,
        token_mixer: str = "none",
        scan_scale_init: float = 0.05,
        ctm_readout: str = "mean",
        logit_bias: bool = False,
        logit_bias_init: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.yolo_head = yolo_head
        self.token_proj = nn.Linear(in_dim, d_model)
        self.spatial_encoding = str(spatial_encoding).lower()
        if self.spatial_encoding not in {"none", "polar"}:
            raise ValueError("spatial_encoding must be one of: none, polar")
        if self.spatial_encoding == "polar":
            self.spatial_proj = nn.Linear(3, d_model, bias=False)
            self.spatial_scale = nn.Parameter(torch.tensor(float(spatial_encoding_scale_init)))
        else:
            self.spatial_proj = None
            self.spatial_scale = None
        self.token_mixer_type = str(token_mixer).lower()
        if self.token_mixer_type not in {"none", "cross_scan"}:
            raise ValueError("token_mixer must be one of: none, cross_scan")
        self.cross_scan_mixer = (
            SharedCrossScanMixer(d_model=d_model, scale_init=scan_scale_init)
            if self.token_mixer_type == "cross_scan"
            else None
        )
        self.ctm_block = CTMBlock(d_model=d_model, dropout=dropout)
        self.steps = steps
        self.norm = nn.LayerNorm(d_model)
        self.ctm_readout = str(ctm_readout).lower()
        if self.ctm_readout not in {"mean", "attention"}:
            raise ValueError("ctm_readout must be one of: mean, attention")
        if self.ctm_readout == "attention":
            self.readout_query = nn.Parameter(torch.zeros(d_model))
        self.ctm_cls = nn.Linear(d_model, num_classes)
        if logit_bias:
            self.logit_bias = nn.Parameter(torch.zeros(num_classes, dtype=torch.float32))
            if logit_bias_init is not None:
                init = torch.as_tensor(logit_bias_init, dtype=torch.float32).view(-1)
                if init.numel() != num_classes:
                    raise ValueError(f"logit_bias_init has {init.numel()} values, expected {num_classes}")
                with torch.no_grad():
                    self.logit_bias.copy_(init)
        else:
            self.logit_bias = None
        self.logprob_fusion_enabled = bool(logprob_fusion)
        if self.logprob_fusion_enabled:
            init = min(max(float(logprob_fusion_init), 1e-4), 1.0 - 1e-4)
            init_logit = np.log(init / (1.0 - init))
            self.logprob_fusion_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        else:
            self.ctm_scale = nn.Parameter(torch.tensor(0.1))
        self.expert_fusion = str(expert_fusion).lower()
        if self.expert_fusion not in {"none", "classwise_logprob"}:
            raise ValueError("expert_fusion must be one of: none, classwise_logprob")
        if self.expert_fusion == "classwise_logprob":
            init = min(max(float(expert_ctm_init), 1e-4), 1.0 - 1e-4)
            init_logit = np.log(init / (1.0 - init))
            self.expert_ctm_logits = nn.Parameter(torch.full((num_classes,), init_logit, dtype=torch.float32))
        else:
            self.expert_ctm_logits = None
        self.feature_adapter_enabled = bool(feature_adapter)
        self.feature_fusion = str(feature_fusion).lower()
        if self.feature_fusion not in {"residual", "gated", "token"}:
            raise ValueError("feature_fusion must be one of: residual, gated, token")
        if self.feature_fusion in {"gated", "token"} and not self.feature_adapter_enabled:
            raise ValueError("feature_fusion='gated' or 'token' requires feature_adapter=True")
        self.gate_rank = max(1, int(gate_rank))
        self.token_gate_rank = max(1, int(token_gate_rank))
        self.adapter_rank = int(adapter_rank)
        if self.feature_adapter_enabled and self.adapter_rank > 0:
            self.feature_adapter = nn.Sequential(
                nn.Linear(d_model, self.adapter_rank, bias=False),
                nn.SiLU(),
                nn.Linear(self.adapter_rank, in_dim, bias=True),
            )
            nn.init.zeros_(self.feature_adapter[-1].weight)
            nn.init.zeros_(self.feature_adapter[-1].bias)
        elif self.feature_adapter_enabled:
            self.feature_adapter = nn.Linear(d_model, in_dim)
            nn.init.zeros_(self.feature_adapter.weight)
            nn.init.zeros_(self.feature_adapter.bias)
        else:
            self.feature_adapter = None
        self.feature_gate = None
        self.token_gate = None
        if self.feature_adapter_enabled:
            self.feature_adapter_scale = nn.Parameter(torch.tensor(0.05))
            if self.feature_fusion == "gated":
                self.feature_gate = nn.Sequential(
                    nn.Linear(in_dim * 2, self.gate_rank, bias=False),
                    nn.SiLU(),
                    nn.Linear(self.gate_rank, in_dim),
                )
                nn.init.zeros_(self.feature_gate[-1].weight)
                nn.init.zeros_(self.feature_gate[-1].bias)
            elif self.feature_fusion == "token":
                self.token_gate = nn.Sequential(
                    nn.Linear(d_model, self.token_gate_rank, bias=False),
                    nn.SiLU(),
                    nn.Linear(self.token_gate_rank, 1),
                )
                nn.init.zeros_(self.token_gate[-1].weight)
                nn.init.zeros_(self.token_gate[-1].bias)

    @staticmethod
    def _as_logits(output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[1]
        return output

    def forward(
        self,
        images: torch.Tensor,
        return_aux: bool = False,
        return_clean: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        feats = self.backbone(images)
        if feats.ndim == 2:
            tokens = feats.unsqueeze(1)
        else:
            tokens = feats.flatten(2).transpose(1, 2)

        inputs = self.token_proj(tokens)
        if self.spatial_encoding == "polar" and feats.ndim == 4:
            if self.spatial_proj is None or self.spatial_scale is None:
                raise RuntimeError("Polar spatial encoding parameters are not initialized")
            coords = self._polar_coordinates(feats.shape[-2], feats.shape[-1], inputs.device, inputs.dtype)
            inputs = inputs + self.spatial_scale * self.spatial_proj(coords).unsqueeze(0)
        if self.token_mixer_type == "cross_scan" and feats.ndim == 4:
            if self.cross_scan_mixer is None:
                raise RuntimeError("Cross-scan mixer parameters are not initialized")
            inputs = self.cross_scan_mixer(inputs, feats.shape[-2], feats.shape[-1])
        state = torch.zeros_like(inputs)
        for _ in range(self.steps):
            state = self.ctm_block(state, inputs)

        state = self.norm(state)
        pooled = self._pool_ctm_state(state)
        needs_clean_branch = return_clean or self.expert_fusion == "classwise_logprob"
        clean_yolo_logits = self._as_logits(self.yolo_head(feats)) if needs_clean_branch else None
        fused_feats = self._apply_feature_adapter(feats, state)
        yolo_logits = self._as_logits(self.yolo_head(fused_feats))
        ctm_logits = self.ctm_cls(pooled)
        if self.logprob_fusion_enabled:
            ctm_weight = torch.sigmoid(self.logprob_fusion_logit)
            yolo_log_probs = F.log_softmax(yolo_logits, dim=1)
            ctm_log_probs = F.log_softmax(ctm_logits, dim=1)
            logits = torch.logsumexp(
                torch.stack(
                    [
                        yolo_log_probs + torch.log1p(-ctm_weight),
                        ctm_log_probs + torch.log(ctm_weight),
                    ],
                    dim=0,
                ),
                dim=0,
            )
        else:
            logits = yolo_logits + self.ctm_scale * ctm_logits
        if self.expert_fusion == "classwise_logprob":
            if clean_yolo_logits is None or self.expert_ctm_logits is None:
                raise RuntimeError("Classwise expert fusion requires clean YOLO logits and route parameters")
            ctm_weights = torch.sigmoid(self.expert_ctm_logits).clamp(1e-5, 1.0 - 1e-5).view(1, -1)
            clean_log_probs = F.log_softmax(clean_yolo_logits, dim=1)
            ctm_log_probs = F.log_softmax(logits, dim=1)
            logits = torch.logaddexp(
                clean_log_probs + torch.log1p(-ctm_weights),
                ctm_log_probs + torch.log(ctm_weights),
            )
        if self.logit_bias is not None:
            logits = logits + self.logit_bias.view(1, -1)
        if return_aux and return_clean:
            if clean_yolo_logits is None:
                raise RuntimeError("clean_yolo_logits was not computed")
            return logits, ctm_logits, clean_yolo_logits
        if return_aux:
            return logits, ctm_logits
        if return_clean:
            if clean_yolo_logits is None:
                raise RuntimeError("clean_yolo_logits was not computed")
            return logits, clean_yolo_logits
        return logits

    def _pool_ctm_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.ctm_readout == "mean":
            return state.mean(dim=1)
        query = self.readout_query.view(1, 1, -1)
        scores = (state * query).sum(dim=-1) / float(state.shape[-1]) ** 0.5
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (state * weights).sum(dim=1)

    @staticmethod
    def _polar_coordinates(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square()) / (2.0**0.5)
        return torch.stack([xx, yy, radius], dim=-1).reshape(-1, 3)

    def _apply_feature_adapter(self, feats: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if not self.feature_adapter_enabled or self.feature_adapter is None:
            return feats
        residual = self.feature_adapter(state)
        if feats.ndim == 4:
            batch, _channels, height, width = feats.shape
            residual = residual.transpose(1, 2).reshape(batch, -1, height, width)
            if self.feature_fusion == "gated" and self.feature_gate is not None:
                gate = self._feature_gate(feats, residual)
                return feats + self.feature_adapter_scale * gate * residual
            if self.feature_fusion == "token" and self.token_gate is not None:
                gate = self._token_gate(state, height, width)
                return feats + self.feature_adapter_scale * gate * residual
            return feats + self.feature_adapter_scale * residual
        if feats.ndim == 2:
            residual = residual.squeeze(1)
            if self.feature_fusion == "gated" and self.feature_gate is not None:
                gate = self._feature_gate(feats, residual)
                return feats + self.feature_adapter_scale * gate * residual
            if self.feature_fusion == "token" and self.token_gate is not None:
                gate = torch.sigmoid(self.token_gate(state)).squeeze(-1)
                return feats + self.feature_adapter_scale * gate * residual
            return feats + self.feature_adapter_scale * residual
        return feats

    def _feature_gate(self, feats: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if self.feature_gate is None:
            raise RuntimeError("feature_gate is not initialized")
        if feats.ndim == 4:
            feat_desc = feats.mean(dim=(2, 3))
            residual_desc = residual.mean(dim=(2, 3))
            gate = torch.sigmoid(self.feature_gate(torch.cat([feat_desc, residual_desc], dim=1)))
            return gate.view(feats.shape[0], -1, 1, 1)
        gate = torch.sigmoid(self.feature_gate(torch.cat([feats, residual], dim=1)))
        return gate

    def _token_gate(self, state: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if self.token_gate is None:
            raise RuntimeError("token_gate is not initialized")
        gate = torch.sigmoid(self.token_gate(state))
        return gate.transpose(1, 2).reshape(state.shape[0], 1, height, width)


def reset_classification_head(head: nn.Module, num_classes: int) -> None:
    linear = getattr(head, "linear", None)
    if not isinstance(linear, nn.Linear):
        raise RuntimeError("Unexpected YOLO classification head: missing linear layer")
    if linear.out_features == num_classes:
        return
    head.linear = nn.Linear(linear.in_features, num_classes)


def resolve_model_path(model_name: str | None) -> str | None:
    if model_name is None or str(model_name).strip() == "":
        return None
    path = Path(model_name)
    if path.is_absolute() and path.exists():
        return str(path)
    if path.exists():
        return str(path.resolve())
    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return str(project_path)
    return str(path)


def build_yolo_backbone(
    model_config: str | None,
    weights: str | None,
    pretrained: bool,
    imgsz: int,
) -> tuple[nn.Module, int]:
    from ultralytics import YOLO

    weights = resolve_model_path(weights)

    if model_config:
        ul_model = YOLO(model_config)
        if pretrained and weights:
            ul_model.load(weights)
    elif weights:
        if not pretrained:
            raise ValueError("--model-config is required when --no-pretrained is used without a model architecture")
        ul_model = YOLO(weights)
    else:
        raise ValueError("Either --model-config or --weights must be provided")

    ul_model = ul_model.model
    layers = list(ul_model.model)
    if len(layers) < 2:
        raise RuntimeError("Unexpected YOLO classification architecture")
    backbone = nn.Sequential(*layers[:-1])

    with torch.no_grad():
        dummy = torch.zeros(1, 3, imgsz, imgsz)
        feats = backbone(dummy)
        in_dim = feats.shape[1] if feats.ndim > 2 else feats.shape[-1]
    return backbone, int(in_dim)


def build_yolo_components(
    model_config: str | None,
    weights: str | None,
    pretrained: bool,
    imgsz: int,
    num_classes: int,
) -> tuple[nn.Module, nn.Module, int]:
    from ultralytics import YOLO

    weights = resolve_model_path(weights)

    if model_config:
        ul_model = YOLO(model_config)
        if pretrained and weights:
            ul_model.load(weights)
    elif weights:
        if not pretrained:
            raise ValueError("--model-config is required when --no-pretrained is used without a model architecture")
        ul_model = YOLO(weights)
    else:
        raise ValueError("Either --model-config or --weights must be provided")

    layers = list(ul_model.model.model)
    if len(layers) < 2:
        raise RuntimeError("Unexpected YOLO classification architecture")
    backbone = nn.Sequential(*layers[:-1])
    yolo_head = layers[-1]
    reset_classification_head(yolo_head, num_classes)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, imgsz, imgsz)
        feats = backbone(dummy)
        in_dim = feats.shape[1] if feats.ndim > 2 else feats.shape[-1]
    for parameter in backbone.parameters():
        parameter.requires_grad_(True)
    for parameter in yolo_head.parameters():
        parameter.requires_grad_(True)
    return backbone, yolo_head, int(in_dim)


def resolve_device(device_arg: str) -> torch.device:
    device_arg = str(device_arg).strip().lower()
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device '{device_arg}' was requested, but CUDA is not available")
        return torch.device(f"cuda:{device_arg}")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is not available")
        return torch.device("cuda")
    if device_arg.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device '{device_arg}' was requested, but CUDA is not available")
        return torch.device(device_arg)
    if "," in device_arg:
        raise ValueError("This standalone PyTorch trainer accepts one device only, for example '0', 'cuda:0', or 'cpu'")
    return torch.device(device_arg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YoloCTM on WM-811K")
    parser.add_argument("--data", type=Path, default=Path("data/wm811k_cls"))
    parser.add_argument("--model", default=None, help="Backward-compatible alias for --weights")
    parser.add_argument("--model-config", default=None, help="YOLO classification architecture YAML")
    parser.add_argument("--weights", default="yolov8n-cls.pt", help="Pretrained YOLO classification checkpoint")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--adapter-rank", type=int, default=0)
    parser.add_argument("--feature-adapter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-fusion", choices=["residual", "gated", "token"], default="residual")
    parser.add_argument("--gate-rank", type=int, default=16)
    parser.add_argument("--token-gate-rank", type=int, default=16)
    parser.add_argument("--logprob-fusion", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--logprob-fusion-init", type=float, default=0.2)
    parser.add_argument("--expert-fusion", choices=["none", "classwise_logprob"], default="none")
    parser.add_argument("--expert-ctm-init", type=float, default=0.4)
    parser.add_argument("--freeze-yolo-anchor", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--spatial-encoding", choices=["none", "polar"], default="none")
    parser.add_argument("--spatial-encoding-scale-init", type=float, default=0.05)
    parser.add_argument("--token-mixer", choices=["none", "cross_scan"], default="none")
    parser.add_argument("--scan-scale-init", type=float, default=0.05)
    parser.add_argument("--ctm-readout", choices=["mean", "attention"], default="mean")
    parser.add_argument("--logit-bias", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--logit-bias-init", choices=["zero", "prior"], default="zero")
    parser.add_argument("--logit-bias-prior-tau", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aux-loss-weight", type=float, default=0.0)
    parser.add_argument("--anchor-loss-weight", type=float, default=0.0)
    parser.add_argument("--distill-logprobs", type=Path, default=None, help="Optional NPZ with train-split teacher log-probabilities")
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--distill-mode", choices=["full", "nontarget_zscore"], default="full")
    parser.add_argument("--classifier-cbr-weight", type=float, default=0.0)
    parser.add_argument("--classifier-cbr-power", type=float, default=1.0)
    parser.add_argument("--classifier-cbr-start-epoch", type=int, default=1)
    parser.add_argument("--loss", choices=["weighted_ce", "balanced_softmax"], default="weighted_ce")
    parser.add_argument("--train-sampling", choices=["natural", "none_aware"], default="natural")
    parser.add_argument("--none-sampling-ratio", type=float, default=0.5)
    parser.add_argument("--train-sampling-start-epoch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--project", default="runs/classify")
    parser.add_argument("--name", default="wm811k_yoloctm")
    return parser.parse_args()


def anchor_kl_loss(fused_logits: torch.Tensor, clean_logits: torch.Tensor) -> torch.Tensor:
    teacher_probs = F.softmax(clean_logits.detach(), dim=1)
    return F.kl_div(F.log_softmax(fused_logits, dim=1), teacher_probs, reduction="batchmean")


def _zscore(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = logits.mean(dim=1, keepdim=True)
    std = logits.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return (logits - mean) / std


def distillation_loss(
    logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    temperature: float,
    labels: torch.Tensor | None = None,
    mode: str = "full",
) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    mode = str(mode).lower()
    if mode == "nontarget_zscore":
        if labels is None:
            raise ValueError("labels are required for nontarget_zscore distillation")
        keep = torch.ones_like(logits, dtype=torch.bool)
        keep.scatter_(1, labels.view(-1, 1), False)
        logits = logits.masked_select(keep).view(logits.shape[0], logits.shape[1] - 1)
        teacher_log_probs = teacher_log_probs.masked_select(keep).view(teacher_log_probs.shape[0], teacher_log_probs.shape[1] - 1)
        logits = _zscore(logits)
        teacher_log_probs = _zscore(teacher_log_probs)
    elif mode != "full":
        raise ValueError(f"Unknown distillation mode: {mode}")

    student_log_probs = F.log_softmax(logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_log_probs / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * temperature * temperature


def classifier_cbr_loss(model: nn.Module, class_counts: torch.Tensor, power: float = 1.0) -> torch.Tensor:
    weights: list[torch.Tensor] = []
    yolo_linear = getattr(getattr(model, "yolo_head", None), "linear", None)
    if isinstance(yolo_linear, nn.Linear):
        weights.append(yolo_linear.weight)
    ctm_cls = getattr(model, "ctm_cls", None)
    if isinstance(ctm_cls, nn.Linear):
        weights.append(ctm_cls.weight)
    if not weights:
        return torch.zeros((), device=class_counts.device)

    freq = class_counts.to(device=weights[0].device, dtype=weights[0].dtype).clamp(min=1.0)
    freq_weights = (freq / freq.mean()).pow(float(power))
    losses = []
    for weight in weights:
        if weight.shape[0] != freq_weights.numel():
            continue
        class_norms = weight.pow(2).sum(dim=1)
        losses.append((freq_weights * class_norms).mean())
    if not losses:
        return torch.zeros((), device=weights[0].device)
    return torch.stack(losses).mean()


def unpack_batch(batch):
    if len(batch) == 3:
        images, labels, indices = batch
        return images, labels, indices
    images, labels = batch
    return images, labels, None


def load_distill_logprobs(path: Path, train_ds: datasets.ImageFolder) -> torch.Tensor:
    payload = np.load(path, allow_pickle=True)
    log_probs = np.asarray(payload["log_probs"], dtype=np.float32)
    expected_shape = (len(train_ds), len(train_ds.classes))
    if log_probs.shape != expected_shape:
        raise ValueError(f"Teacher log-probs shape {log_probs.shape} does not match {expected_shape}")

    if "classes" in payload:
        cached_classes = [str(value) for value in payload["classes"].tolist()]
        if cached_classes != list(train_ds.classes):
            raise ValueError(f"Teacher classes do not match dataset classes: {cached_classes} != {train_ds.classes}")

    if "paths" in payload:
        cached_paths = [str(value).lower().replace("/", "\\") for value in payload["paths"].tolist()]
        current_paths = [str(path).lower().replace("/", "\\") for path, _label in train_ds.samples]
        if cached_paths != current_paths:
            raise ValueError("Teacher log-prob paths do not match current train dataset order")

    return torch.from_numpy(log_probs)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    aux_loss_weight: float = 0.0,
    anchor_loss_weight: float = 0.0,
    distill_logprobs: torch.Tensor | None = None,
    distill_weight: float = 0.0,
    distill_temperature: float = 2.0,
    distill_mode: str = "full",
    class_counts: torch.Tensor | None = None,
    classifier_cbr_weight: float = 0.0,
    classifier_cbr_power: float = 1.0,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        images, labels, indices = unpack_batch(batch)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if aux_loss_weight > 0 and anchor_loss_weight > 0:
            logits, ctm_logits, clean_logits = model(images, return_aux=True, return_clean=True)
            loss = (
                criterion(logits, labels)
                + aux_loss_weight * criterion(ctm_logits, labels)
                + anchor_loss_weight * anchor_kl_loss(logits, clean_logits)
            )
        elif aux_loss_weight > 0:
            logits, ctm_logits = model(images, return_aux=True)
            loss = criterion(logits, labels) + aux_loss_weight * criterion(ctm_logits, labels)
        elif anchor_loss_weight > 0:
            logits, clean_logits = model(images, return_clean=True)
            loss = criterion(logits, labels) + anchor_loss_weight * anchor_kl_loss(logits, clean_logits)
        else:
            logits = model(images)
            loss = criterion(logits, labels)
        if distill_logprobs is not None and indices is not None and distill_weight > 0:
            teacher_log_probs = distill_logprobs[indices].to(device=device, non_blocking=True)
            loss = loss + float(distill_weight) * distillation_loss(
                logits,
                teacher_log_probs,
                distill_temperature,
                labels=labels,
                mode=distill_mode,
            )
        if class_counts is not None and classifier_cbr_weight > 0:
            loss = loss + float(classifier_cbr_weight) * classifier_cbr_loss(
                model,
                class_counts,
                power=float(classifier_cbr_power),
            )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in loader:
            images, labels, _indices = unpack_batch(batch)
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return total_loss / max(total, 1), correct / max(total, 1), macro_f1


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)

    train_transform = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz), interpolation=InterpolationMode.NEAREST),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180, interpolation=InterpolationMode.NEAREST),
        WaferToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz), interpolation=InterpolationMode.NEAREST),
        WaferToTensor(),
    ])

    train_ds = IndexedImageFolder(args.data / "train", transform=train_transform)
    val_ds = IndexedImageFolder(args.data / "val", transform=val_transform)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(args.seed))
    natural_train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    rebalanced_train_loader = natural_train_loader
    if args.train_sampling == "none_aware":
        train_sampler = build_none_aware_sampler(
            targets=list(train_ds.targets),
            classes=list(train_ds.classes),
            none_sampling_ratio=float(args.none_sampling_ratio),
            seed=int(args.seed),
        )
        rebalanced_train_loader = DataLoader(
            train_ds,
            batch_size=args.batch,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=seed_worker,
        )
        if int(args.train_sampling_start_epoch) > 1:
            print(f"[sampling] deferred none-aware sampler starts at epoch {int(args.train_sampling_start_epoch)}")
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    class_counts = torch.bincount(torch.tensor(train_ds.targets), minlength=len(train_ds.classes)).float()
    logit_bias_init = None
    if bool(args.logit_bias) and str(args.logit_bias_init).lower() == "prior":
        logit_bias_init = float(args.logit_bias_prior_tau) * class_counts.clamp(min=1.0).log()

    weights = args.model if args.model is not None else args.weights
    backbone, yolo_head, in_dim = build_yolo_components(
        args.model_config,
        weights,
        args.pretrained,
        args.imgsz,
        len(train_ds.classes),
    )
    model = YoloCTM(
        backbone=backbone,
        yolo_head=yolo_head,
        num_classes=len(train_ds.classes),
        in_dim=in_dim,
        d_model=args.d_model,
        steps=args.steps,
        dropout=args.dropout,
        adapter_rank=args.adapter_rank,
        feature_adapter=bool(args.feature_adapter),
        feature_fusion=str(args.feature_fusion),
        gate_rank=int(args.gate_rank),
        token_gate_rank=int(args.token_gate_rank),
        logprob_fusion=bool(args.logprob_fusion),
        logprob_fusion_init=float(args.logprob_fusion_init),
        expert_fusion=str(args.expert_fusion),
        expert_ctm_init=float(args.expert_ctm_init),
        spatial_encoding=str(args.spatial_encoding),
        spatial_encoding_scale_init=float(args.spatial_encoding_scale_init),
        token_mixer=str(args.token_mixer),
        scan_scale_init=float(args.scan_scale_init),
        ctm_readout=str(args.ctm_readout),
        logit_bias=bool(args.logit_bias),
        logit_bias_init=logit_bias_init,
    ).to(device)
    if bool(args.freeze_yolo_anchor):
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in model.yolo_head.parameters():
            parameter.requires_grad_(False)
        print("[anchor] frozen YOLO backbone and classification head; training CTM residual correction only")

    if args.loss == "balanced_softmax":
        criterion = BalancedSoftmaxLoss(class_counts).to(device)
    else:
        class_weights = (class_counts.sum() / class_counts.clamp(min=1.0)).pow(args.class_weight_power)
        class_weights = (class_weights / class_weights.mean()).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    distill_logprobs = None
    if args.distill_logprobs is not None and float(args.distill_weight) > 0:
        distill_logprobs = load_distill_logprobs(args.distill_logprobs, train_ds)
        print(
            f"[distill] loaded teacher log-probs: {args.distill_logprobs} "
            f"weight={float(args.distill_weight):.4f} temperature={float(args.distill_temperature):.4f} "
            f"mode={args.distill_mode}"
        )
    if float(args.classifier_cbr_weight) > 0:
        print(
            f"[cbr] classifier regularization weight={float(args.classifier_cbr_weight):.4f} "
            f"power={float(args.classifier_cbr_power):.4f} starts at epoch {int(args.classifier_cbr_start_epoch)}"
        )

    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0
    best_val_acc = 0.0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        classifier_cbr_weight = (
            float(args.classifier_cbr_weight)
            if epoch >= int(args.classifier_cbr_start_epoch)
            else 0.0
        )
        train_loader = (
            rebalanced_train_loader
            if args.train_sampling == "none_aware" and epoch >= int(args.train_sampling_start_epoch)
            else natural_train_loader
        )
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            aux_loss_weight=float(args.aux_loss_weight),
            anchor_loss_weight=float(args.anchor_loss_weight),
            distill_logprobs=distill_logprobs,
            distill_weight=float(args.distill_weight),
            distill_temperature=float(args.distill_temperature),
            distill_mode=str(args.distill_mode),
            class_counts=class_counts,
            classifier_cbr_weight=classifier_cbr_weight,
            classifier_cbr_power=float(args.classifier_cbr_power),
        )
        val_loss, val_acc, val_macro_f1 = evaluate(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
        })
        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f}"
        )
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": train_ds.classes,
                    "args": vars(args),
                    "in_dim": in_dim,
                    "architecture": (
                        f"yolo_head_{args.feature_fusion}_shared_ctm_{args.ctm_readout}_readout_"
                        "feature_adapter_macro_f1_selected"
                        if args.feature_adapter
                        else f"yolo_head_residual_shared_ctm_{args.ctm_readout}_readout_macro_f1_selected"
                    ),
                    "best_val_acc": best_val_acc,
                    "best_val_macro_f1": best_val_f1,
                    "classifier_cbr_weight": float(args.classifier_cbr_weight),
                    "classifier_cbr_power": float(args.classifier_cbr_power),
                    "classifier_cbr_start_epoch": int(args.classifier_cbr_start_epoch),
                },
                out_dir / "best_yoloctm.pt",
            )

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"best_val_acc": best_val_acc, "best_val_macro_f1": best_val_f1, "history": history, "classes": train_ds.classes},
            f,
            indent=2,
        )

    print(f"Saved best model to {out_dir / 'best_yoloctm.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
