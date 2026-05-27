from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class Tee:
    def __init__(self, console_stream, log_stream):
        self.console_stream = console_stream
        self.log_stream = log_stream
        self._line_buffer = ""

    def write(self, data: str) -> int:
        self.console_stream.write(data)
        self.console_stream.flush()
        self._write_clean_log(data)
        return len(data)

    def flush(self) -> None:
        self.console_stream.flush()
        self.log_stream.flush()

    def _write_clean_log(self, data: str) -> None:
        clean = ANSI_ESCAPE_RE.sub("", data)
        clean = clean.replace("\x08", "")
        for char in clean:
            if char == "\r":
                self._line_buffer = ""
                continue
            if char == "\n":
                self.log_stream.write(self._line_buffer.rstrip() + "\n")
                self._line_buffer = ""
                continue
            self._line_buffer += char
        self.log_stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the WM-811K classification pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "wm811k_cls.yaml",
        help="Pipeline YAML config",
    )
    parser.add_argument("--skip-prepare", action="store_true", help="Skip dataset preparation")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--skip-val", action="store_true", help="Skip validation on the val split")
    parser.add_argument("--skip-test", action="store_true", help="Skip evaluation on the test split")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip report/metric generation and defer AutoResearch logging")
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        default=None,
        help="Resume evaluation/logging in an existing run directory, or resume training with --train-resume-checkpoint",
    )
    parser.add_argument("--train-resume-checkpoint", type=Path, default=None, help="Resume YoloCTM training state in an existing run")
    parser.add_argument("--eval-device", default=None, help="Override the device used for validation/test/metrics")
    parser.add_argument(
        "--prior-logit-tau",
        type=float,
        default=None,
        help="Use a known prior-logit calibration tau instead of selecting it again",
    )
    parser.add_argument("--check-config", action="store_true", help="Load config and print the resolved plan")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return config


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise TypeError(f"Config section '{name}' must be a mapping")
    return section


def build_run_name(train_config: dict[str, Any]) -> str:
    name = str(train_config.get("name", "wm811k"))
    if train_config.get("add_timestamp", False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{name}_{timestamp}"
    return name


def build_run_dir(train_config: dict[str, Any], run_name: str) -> Path:
    project = resolve_path(train_config.get("project", "runs/classify"))
    return project / run_name


def setup_logging(run_dir: Path, filename: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / filename
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    stdout = Tee(sys.__stdout__, log_file)
    stderr = Tee(sys.__stderr__, log_file)
    return log_path, log_file, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr)


def prepare_dataset(dataset_config: dict[str, Any]) -> None:
    from prepare_wm811k_classification import main as prepare_main

    source = resolve_path(dataset_config.get("source", "data/MIR-WM811K"))
    output = resolve_path(dataset_config.get("output", "data/wm811k_cls"))
    ratios = dataset_config.get("ratios", [60, 15, 25])

    argv = [
        "prepare_wm811k_classification.py",
        "--source",
        str(source),
        "--output",
        str(output),
        "--image-size",
        str(dataset_config.get("image_size", 224)),
        "--seed",
        str(dataset_config.get("seed", 42)),
        "--ratios",
        *[str(value) for value in ratios],
    ]
    if dataset_config.get("include_none", False):
        argv.append("--include-none")
    if dataset_config.get("overwrite", False):
        argv.append("--overwrite")

    original_argv = sys.argv[:]
    try:
        sys.argv = argv
        prepare_main()
    finally:
        sys.argv = original_argv


def require_dataset(data_root: Path) -> None:
    missing = [split for split in ("train", "val", "test") if not (data_root / split).exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset split folders under {data_root}: {', '.join(missing)}")


def log_torch_runtime(device: str) -> None:
    import torch

    print(f"[runtime] torch={torch.__version__}")
    print(f"[runtime] torch_cuda={torch.version.cuda}")
    print(f"[runtime] cuda_available={torch.cuda.is_available()}")
    print(f"[runtime] requested_device={device}")
    if torch.cuda.is_available():
        print(f"[runtime] cuda_device_count={torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"[runtime] cuda:{index}={torch.cuda.get_device_name(index)}")
    elif device.lower() != "cpu":
        raise RuntimeError(
            f"Config requested device '{device}', but PyTorch cannot access CUDA. "
            "Install a CUDA-enabled PyTorch build or set train.device to 'cpu'."
        )


def resolve_model_path(model_name: str) -> str:
    model_path = Path(model_name)
    if model_path.exists() or model_path.is_absolute():
        return str(model_path)

    root_model_path = PROJECT_ROOT / model_path
    if root_model_path.exists():
        return str(root_model_path)

    return model_name


def resolve_optional_model_path(value: Any) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return resolve_model_path(str(value))


def get_model_algorithm(config: dict[str, Any]) -> str:
    algorithm = str(get_section(config, "model").get("algorithm", "yolo")).lower()
    if algorithm not in {"yolo", "yoloctm"}:
        raise ValueError("model.algorithm must be one of: yolo, yoloctm")
    return algorithm


def get_model_sources(model_config: dict[str, Any]) -> tuple[str | None, str | None]:
    model_config_path = resolve_optional_model_path(model_config.get("config"))
    legacy_name = model_config.get("name")
    weights = resolve_optional_model_path(model_config.get("weights", legacy_name))
    return model_config_path, weights


def train_yolo_model(config: dict[str, Any], run_name: str):
    from ultralytics import YOLO

    dataset_config = get_section(config, "dataset")
    model_config = get_section(config, "model")
    train_config = get_section(config, "train")

    data_root = resolve_path(dataset_config.get("output", "data/wm811k_cls"))
    require_dataset(data_root)

    device = str(train_config.get("device", "0"))
    log_torch_runtime(device)
    model_config_path, weights = get_model_sources(model_config)
    pretrained = bool(model_config.get("pretrained", False))
    model_source = model_config_path or weights or "yolo11m-cls.pt"
    print(f"[runtime] algorithm=yolo")
    print(f"[runtime] model_config={model_config_path}")
    print(f"[runtime] weights={weights}")

    model = YOLO(model_source)
    if model_config_path and pretrained and weights:
        model.load(weights)
    train_pretrained = False if model_config_path and pretrained and weights else pretrained
    model.train(
        data=str(data_root),
        epochs=int(train_config.get("epochs", 40)),
        imgsz=int(train_config.get("imgsz", dataset_config.get("image_size", 224))),
        batch=int(train_config.get("batch", 64)),
        device=device,
        workers=int(train_config.get("workers", 8)),
        patience=int(train_config.get("patience", 20)),
        project=str(resolve_path(train_config.get("project", "runs/classify"))),
        name=run_name,
        exist_ok=bool(train_config.get("exist_ok", True)),
        pretrained=train_pretrained,
    )
    return model


def train_yoloctm_model(config: dict[str, Any], run_name: str, resume_checkpoint: Path | None = None) -> Path:
    from train_wm811k_yoloctm import main as train_yoloctm_main

    dataset_config = get_section(config, "dataset")
    model_config = get_section(config, "model")
    train_config = get_section(config, "train")
    ctm_config = get_section(model_config, "ctm")

    data_root = resolve_path(dataset_config.get("output", "data/wm811k_cls"))
    require_dataset(data_root)

    device = str(train_config.get("device", "0"))
    log_torch_runtime(device)
    model_config_path, weights = get_model_sources(model_config)

    print(f"[runtime] algorithm=yoloctm")
    print(f"[runtime] model_config={model_config_path}")
    print(f"[runtime] weights={weights}")

    argv = [
        "train_wm811k_yoloctm.py",
        "--data",
        str(data_root),
        "--epochs",
        str(train_config.get("epochs", 40)),
        "--batch",
        str(train_config.get("batch", 64)),
        "--micro-batch",
        str(train_config.get("micro_batch", 0)),
        "--imgsz",
        str(train_config.get("imgsz", dataset_config.get("image_size", 224))),
        "--lr",
        str(train_config.get("lr", 1e-3)),
        "--weight-decay",
        str(train_config.get("weight_decay", 1e-4)),
        "--d-model",
        str(ctm_config.get("d_model", 256)),
        "--steps",
        str(ctm_config.get("steps", 4)),
        "--dropout",
        str(ctm_config.get("dropout", 0.1)),
        "--class-weight-power",
        str(ctm_config.get("class_weight_power", 0.5)),
        "--adapter-rank",
        str(ctm_config.get("adapter_rank", 0)),
        "--feature-adapter" if bool(ctm_config.get("feature_adapter", True)) else "--no-feature-adapter",
        "--feature-fusion",
        str(ctm_config.get("feature_fusion", "residual")),
        "--gate-rank",
        str(ctm_config.get("gate_rank", 16)),
        "--token-gate-rank",
        str(ctm_config.get("token_gate_rank", 16)),
        "--logprob-fusion" if bool(ctm_config.get("logprob_fusion", False)) else "--no-logprob-fusion",
        "--logprob-fusion-init",
        str(ctm_config.get("logprob_fusion_init", 0.2)),
        "--expert-fusion",
        str(ctm_config.get("expert_fusion", "none")),
        "--expert-ctm-init",
        str(ctm_config.get("expert_ctm_init", 0.4)),
        "--freeze-yolo-anchor" if bool(ctm_config.get("freeze_yolo_anchor", False)) else "--no-freeze-yolo-anchor",
        "--spatial-encoding",
        str(ctm_config.get("spatial_encoding", "none")),
        "--spatial-encoding-scale-init",
        str(ctm_config.get("spatial_encoding_scale_init", 0.05)),
        "--token-mixer",
        str(ctm_config.get("token_mixer", "none")),
        "--scan-scale-init",
        str(ctm_config.get("scan_scale_init", 0.05)),
        "--ctm-readout",
        str(ctm_config.get("readout", "mean")),
        "--logit-bias" if bool(ctm_config.get("logit_bias", False)) else "--no-logit-bias",
        "--logit-bias-init",
        str(ctm_config.get("logit_bias_init", "zero")),
        "--logit-bias-prior-tau",
        str(ctm_config.get("logit_bias_prior_tau", 0.4)),
        "--seed",
        str(train_config.get("seed", dataset_config.get("seed", 42))),
        "--aux-loss-weight",
        str(ctm_config.get("aux_loss_weight", 0.0)),
        "--anchor-loss-weight",
        str(ctm_config.get("anchor_loss_weight", 0.0)),
        "--distill-weight",
        str(ctm_config.get("distill_weight", 0.0)),
        "--distill-temperature",
        str(ctm_config.get("distill_temperature", 2.0)),
        "--distill-mode",
        str(ctm_config.get("distill_mode", "full")),
        "--prototype-bcl-weight",
        str(ctm_config.get("prototype_bcl_weight", 0.0)),
        "--prototype-bcl-temperature",
        str(ctm_config.get("prototype_bcl_temperature", 0.1)),
        "--classifier-cbr-weight",
        str(ctm_config.get("classifier_cbr_weight", 0.0)),
        "--classifier-cbr-power",
        str(ctm_config.get("classifier_cbr_power", 1.0)),
        "--classifier-cbr-start-epoch",
        str(ctm_config.get("classifier_cbr_start_epoch", 1)),
        "--loss",
        str(ctm_config.get("loss", "weighted_ce")),
        "--train-sampling",
        str(train_config.get("sampling", "natural")),
        "--none-sampling-ratio",
        str(train_config.get("none_sampling_ratio", 0.5)),
        "--train-sampling-start-epoch",
        str(train_config.get("sampling_start_epoch", 1)),
        "--workers",
        str(train_config.get("workers", 8)),
        "--device",
        device,
        "--project",
        str(resolve_path(train_config.get("project", "runs/classify"))),
        "--name",
        run_name,
        "--pretrained" if bool(model_config.get("pretrained", True)) else "--no-pretrained",
    ]
    if model_config_path:
        argv.extend(["--model-config", model_config_path])
    if weights:
        argv.extend(["--weights", weights])
    distill_logprobs = ctm_config.get("distill_logprobs")
    if distill_logprobs:
        argv.extend(["--distill-logprobs", str(resolve_path(distill_logprobs))])
    if resume_checkpoint is not None:
        argv.extend(["--resume-checkpoint", str(resume_checkpoint.resolve())])

    original_argv = sys.argv[:]
    try:
        sys.argv = argv
        train_yoloctm_main()
    finally:
        sys.argv = original_argv

    checkpoint = build_run_dir(train_config, run_name) / "best_yoloctm.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"YoloCTM checkpoint was not written: {checkpoint}")
    return checkpoint


def train_model(config: dict[str, Any], run_name: str, resume_checkpoint: Path | None = None) -> dict[str, Any]:
    algorithm = get_model_algorithm(config)
    if algorithm == "yoloctm":
        checkpoint = train_yoloctm_model(config, run_name, resume_checkpoint=resume_checkpoint)
        return {"algorithm": "yoloctm", "checkpoint": checkpoint}
    return {"algorithm": "yolo", "model": train_yolo_model(config, run_name)}


def find_best_model(run_dir: Path, fallback_model: str):
    from ultralytics import YOLO

    best = run_dir / "weights" / "best.pt"
    if best.exists():
        return YOLO(str(best))
    return YOLO(resolve_model_path(fallback_model))


def evaluate(model, data_root: Path, split: str, phase: str) -> None:
    print(f"\n[{phase}] Evaluating split='{split}'")
    model.val(data=str(data_root), split=split)


def load_yoloctm_checkpoint(checkpoint: Path, device: str):
    import torch
    import torch.nn as nn
    from train_wm811k_yoloctm import YoloCTM, build_yolo_components, resolve_device

    torch_device = resolve_device(device)
    try:
        saved = torch.load(checkpoint, map_location=torch_device, weights_only=False)
    except TypeError:
        saved = torch.load(checkpoint, map_location=torch_device)
    args = saved.get("args", {})
    weights = args.get("model") or args.get("weights")
    backbone, yolo_head, in_dim = build_yolo_components(
        args.get("model_config"),
        weights,
        bool(args.get("pretrained", True)),
        int(args.get("imgsz", 224)),
        len(saved["classes"]),
    )
    model_state = saved.get("model_state", {})
    has_feature_adapter = any(
        key == "feature_adapter_scale" or key.startswith("feature_adapter.") for key in model_state
    )
    feature_adapter = bool(args.get("feature_adapter", has_feature_adapter))
    feature_fusion = str(args.get("feature_fusion", "residual"))
    has_token_gate = any(key.startswith("token_gate.") for key in model_state)
    if feature_fusion == "token" and "token_gate_rank" not in args:
        token_gate_0 = model_state.get("token_gate.0.weight")
        if hasattr(token_gate_0, "shape") and len(token_gate_0.shape) >= 1:
            args["token_gate_rank"] = int(token_gate_0.shape[0])
    if has_token_gate:
        feature_fusion = "token"
    has_logit_bias = "logit_bias" in model_state
    model = YoloCTM(
        backbone=backbone,
        yolo_head=yolo_head,
        num_classes=len(saved["classes"]),
        in_dim=int(saved.get("in_dim", in_dim)),
        d_model=int(args.get("d_model", 256)),
        steps=int(args.get("steps", 4)),
        dropout=float(args.get("dropout", 0.1)),
        adapter_rank=int(args.get("adapter_rank", 0)),
        feature_adapter=feature_adapter,
        feature_fusion=feature_fusion,
        gate_rank=int(args.get("gate_rank", 16)),
        token_gate_rank=int(args.get("token_gate_rank", 16)),
        logprob_fusion=bool(args.get("logprob_fusion", False)),
        logprob_fusion_init=float(args.get("logprob_fusion_init", 0.2)),
        expert_fusion=str(args.get("expert_fusion", "none")),
        expert_ctm_init=float(args.get("expert_ctm_init", 0.4)),
        spatial_encoding=str(args.get("spatial_encoding", "none")),
        spatial_encoding_scale_init=float(args.get("spatial_encoding_scale_init", 0.05)),
        token_mixer=str(args.get("token_mixer", "none")),
        scan_scale_init=float(args.get("scan_scale_init", 0.05)),
        ctm_readout=str(args.get("ctm_readout", "mean")),
        logit_bias=bool(args.get("logit_bias", has_logit_bias)),
    ).to(torch_device)
    try:
        model.load_state_dict(saved["model_state"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint does not exactly match YoloCTM model; refusing partial load. "
            f"checkpoint={checkpoint}"
        ) from exc
    print(f"[checkpoint] strict load ok: {checkpoint}")
    model.eval()
    return model, saved["classes"], torch_device, nn.CrossEntropyLoss()


def evaluate_yoloctm(
    checkpoint: Path,
    data_root: Path,
    split: str,
    phase: str,
    device: str,
    imgsz: int,
    batch: int,
    write_metrics: bool = False,
    prior_logit_tau: float = 0.0,
) -> None:
    import json
    import numpy as np
    import torch
    from evaluate_wm811k_cls import write_report_csv
    from sklearn.metrics import classification_report, confusion_matrix
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.transforms import InterpolationMode
    from train_wm811k_yoloctm import WaferToTensor

    print(f"\n[{phase}] Evaluating YoloCTM split='{split}'")
    transform = transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz), interpolation=InterpolationMode.NEAREST),
            WaferToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(data_root / split, transform=transform)
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0, pin_memory=False)
    model, checkpoint_classes, torch_device, criterion = load_yoloctm_checkpoint(checkpoint, device)
    if dataset.classes != checkpoint_classes:
        raise ValueError(f"Dataset classes do not match checkpoint classes: {dataset.classes} != {checkpoint_classes}")
    log_prior = None
    if prior_logit_tau != 0.0:
        train_dataset = datasets.ImageFolder(data_root / "train", transform=transform)
        if train_dataset.classes != checkpoint_classes:
            raise ValueError(f"Train classes do not match checkpoint classes: {train_dataset.classes} != {checkpoint_classes}")
        class_counts = torch.bincount(torch.tensor(train_dataset.targets), minlength=len(checkpoint_classes)).float()
        log_prior = class_counts.clamp(min=1.0).log().to(torch_device)
        print(f"[{phase}] prior_logit_tau={prior_logit_tau:.4f}")

    total_loss = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(torch_device), labels.to(torch_device)
            logits = model(images)
            if log_prior is not None:
                logits = logits + float(prior_logit_tau) * log_prior.unsqueeze(0)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    print(f"[{phase}] loss={loss:.4f} acc={acc:.4f} images={total}")

    if not write_metrics:
        return

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(dataset.classes))),
        target_names=dataset.classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(dataset.classes))))

    metrics_dir = checkpoint.parent / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / f"{split}_classification_report.json"
    csv_path = metrics_dir / f"{split}_classification_report.csv"
    matrix_path = metrics_dir / f"{split}_confusion_matrix.csv"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report_csv(report, dataset.classes, csv_path)
    np.savetxt(matrix_path, matrix, delimiter=",", fmt="%d")
    print(f"Report JSON: {json_path}")
    print(f"Report CSV: {csv_path}")
    print(f"Confusion matrix CSV: {matrix_path}")


def select_yoloctm_prior_tau(
    checkpoint: Path,
    data_root: Path,
    split: str,
    device: str,
    imgsz: int,
    batch: int,
    candidates: list[float],
) -> float:
    import json
    import torch
    from sklearn.metrics import f1_score, precision_score, recall_score
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.transforms import InterpolationMode
    from train_wm811k_yoloctm import WaferToTensor

    print(f"\n[calibration] Selecting prior_logit_tau on split='{split}'")
    transform = transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz), interpolation=InterpolationMode.NEAREST),
            WaferToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(data_root / split, transform=transform)
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0, pin_memory=False)
    model, checkpoint_classes, torch_device, _criterion = load_yoloctm_checkpoint(checkpoint, device)
    if dataset.classes != checkpoint_classes:
        raise ValueError(f"Dataset classes do not match checkpoint classes: {dataset.classes} != {checkpoint_classes}")

    train_dataset = datasets.ImageFolder(data_root / "train", transform=transform)
    if train_dataset.classes != checkpoint_classes:
        raise ValueError(f"Train classes do not match checkpoint classes: {train_dataset.classes} != {checkpoint_classes}")
    class_counts = torch.bincount(torch.tensor(train_dataset.targets), minlength=len(checkpoint_classes)).float()
    log_prior = class_counts.clamp(min=1.0).log().to(torch_device)

    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in loader:
            logits_list.append(model(images.to(torch_device)).cpu())
            labels_list.append(labels.cpu())
    logits = torch.cat(logits_list, dim=0).to(torch_device)
    labels = torch.cat(labels_list, dim=0).numpy()

    rows: list[dict[str, float]] = []
    for tau in candidates:
        adjusted = logits + float(tau) * log_prior.unsqueeze(0)
        preds = adjusted.argmax(dim=1).cpu().numpy()
        row = {
            "tau": float(tau),
            "macro_p": float(precision_score(labels, preds, average="macro", zero_division=0)),
            "macro_r": float(recall_score(labels, preds, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        }
        rows.append(row)
        print(
            f"[calibration] tau={row['tau']:.3f} "
            f"macro_p={row['macro_p']:.4f} macro_r={row['macro_r']:.4f} macro_f1={row['macro_f1']:.4f}"
        )

    best = max(rows, key=lambda row: row["macro_f1"])
    metrics_dir = checkpoint.parent / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = {"split": split, "selected_tau": best["tau"], "candidates": rows}
    (metrics_dir / "prior_tau_selection.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[calibration] selected prior_logit_tau={best['tau']:.4f}")
    return float(best["tau"])


def compute_classification_metrics(
    run_dir: Path,
    data_root: Path,
    split: str,
    device: str,
    imgsz: int,
    batch: int,
) -> None:
    from evaluate_wm811k_cls import main as evaluate_metrics_main

    print(f"\n[metrics] Computing precision/recall/F1 for split='{split}'")
    argv = [
        "evaluate_wm811k_cls.py",
        "--run-dir",
        str(run_dir),
        "--data",
        str(data_root),
        "--split",
        split,
        "--device",
        device,
        "--imgsz",
        str(imgsz),
        "--batch",
        str(batch),
    ]
    original_argv = sys.argv[:]
    try:
        sys.argv = argv
        evaluate_metrics_main()
    finally:
        sys.argv = original_argv


def copy_config(config_path: Path, run_dir: Path, config: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml")
    resolved = json.dumps(config, indent=2, ensure_ascii=False)
    (run_dir / "resolved_config.json").write_text(resolved, encoding="utf-8")


def should_log_autoresearch(config_path: Path, config: dict[str, Any]) -> bool:
    logging_config = get_section(config, "logging")
    if "autoresearch" in logging_config:
        return bool(logging_config["autoresearch"])
    try:
        config_path.relative_to(PROJECT_ROOT / "AutoResearch")
    except ValueError:
        return False
    return True


def log_autoresearch_run(run_dir: Path, status: str, description: str) -> None:
    script_path = PROJECT_ROOT / "AutoResearch" / "scripts" / "log_experiment.py"
    if not script_path.exists():
        print(f"[autoresearch] Skipped logging; script not found: {script_path}")
        return

    spec = importlib.util.spec_from_file_location("autoresearch_log_experiment", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load AutoResearch logger: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row, payload = module.build_row(run_dir.resolve(), status, description)
    module.append_row(row)
    payload = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    log_path = module.write_json_log(run_dir.resolve(), payload)
    print(f"[autoresearch] Logged row to {module.RESULTS_PATH}")
    print(f"[autoresearch] JSON summary: {log_path}")


def resolve_autoresearch_status(logging_config: dict[str, Any], run_dir: Path) -> str:
    status = str(logging_config.get("status", "keep")).lower()
    if status != "auto":
        return status
    threshold = logging_config.get("keep_test_macro_f1_min")
    if threshold is None:
        raise ValueError("logging.status='auto' requires logging.keep_test_macro_f1_min")
    report_path = run_dir / "metrics" / "test_classification_report.json"
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    test_macro_f1 = float(report.get("macro avg", {}).get("f1-score", float("-inf")))
    status = "keep" if test_macro_f1 >= float(threshold) else "discard"
    print(
        f"[autoresearch] auto status={status}: test_macro_f1={test_macro_f1:.6f} "
        f"threshold={float(threshold):.6f}"
    )
    return status


def print_plan(config_path: Path, config: dict[str, Any], run_name: str, run_dir: Path) -> None:
    dataset = get_section(config, "dataset")
    prepare = get_section(config, "prepare")
    model = get_section(config, "model")
    train = get_section(config, "train")
    model_config_path, weights = get_model_sources(model)
    plan = {
        "config": str(config_path.resolve()),
        "source": str(resolve_path(dataset.get("source", "data/MIR-WM811K"))),
        "dataset": str(resolve_path(dataset.get("output", "data/wm811k_cls"))),
        "ratios": dataset.get("ratios", [60, 15, 25]),
        "prepare_enabled": prepare.get("enabled", True),
        "algorithm": get_model_algorithm(config),
        "model_config": model_config_path,
        "weights": weights,
        "pretrained": model.get("pretrained", True),
        "epochs": train.get("epochs", 40),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "autoresearch_logging": should_log_autoresearch(config_path, config),
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    train_config = get_section(config, "train")
    logging_config = get_section(config, "logging")
    dataset_config = get_section(config, "dataset")
    prepare_config = get_section(config, "prepare")
    validate_config = get_section(config, "validate")
    test_config = get_section(config, "test")
    metrics_config = get_section(config, "metrics")

    if args.resume_run_dir is not None:
        if not args.skip_train and args.train_resume_checkpoint is None:
            raise ValueError("--resume-run-dir while training requires --train-resume-checkpoint")
        run_dir = resolve_path(args.resume_run_dir).resolve()
        run_name = run_dir.name
    else:
        run_name = build_run_name(train_config)
        run_dir = build_run_dir(train_config, run_name)

    if args.check_config:
        print_plan(config_path, config, run_name, run_dir)
        return 0

    log_path, log_file, out_redirect, err_redirect = setup_logging(
        run_dir, str(logging_config.get("filename", "pipeline.log"))
    )
    with log_file, out_redirect, err_redirect:
        print(f"[pipeline] Config: {config_path}")
        print(f"[pipeline] Run directory: {run_dir}")
        print(f"[pipeline] Log file: {log_path}")
        if args.resume_run_dir is None:
            copy_config(config_path, run_dir, config)
        else:
            print("[pipeline] Resuming existing run; preserving saved config and checkpoint")

        prepare_enabled = bool(prepare_config.get("enabled", True))
        if args.skip_prepare or not prepare_enabled:
            print("[prepare] Skipped")
        else:
            print("\n[prepare] Preparing WM-811K classification dataset")
            prepare_dataset(dataset_config)

        algorithm = get_model_algorithm(config)
        model_result: dict[str, Any]
        if args.skip_train:
            print("[train] Skipped")
            if algorithm == "yoloctm":
                checkpoint = run_dir / "best_yoloctm.pt"
                if not checkpoint.exists():
                    raise FileNotFoundError(f"YoloCTM checkpoint does not exist: {checkpoint}")
                model_result = {"algorithm": "yoloctm", "checkpoint": checkpoint}
            else:
                model_config_path, weights = get_model_sources(get_section(config, "model"))
                fallback_model = weights or model_config_path or "yolo11m-cls.pt"
                model_result = {"algorithm": "yolo", "model": find_best_model(run_dir, fallback_model)}
        else:
            print(f"\n[train] Training {algorithm} classification model")
            resume_checkpoint = resolve_path(args.train_resume_checkpoint).resolve() if args.train_resume_checkpoint else None
            model_result = train_model(config, run_name, resume_checkpoint=resume_checkpoint)
            if algorithm == "yolo":
                model_config_path, weights = get_model_sources(get_section(config, "model"))
                fallback_model = weights or model_config_path or "yolo11m-cls.pt"
                model_result["model"] = find_best_model(run_dir, fallback_model)

        data_root = resolve_path(dataset_config.get("output", "data/wm811k_cls"))
        train_device = str(args.eval_device or train_config.get("device", "0"))
        train_imgsz = int(train_config.get("imgsz", dataset_config.get("image_size", 224)))
        evaluation_requested = (
            (validate_config.get("enabled", True) and not args.skip_val)
            or (test_config.get("enabled", True) and not args.skip_test)
            or (metrics_config.get("enabled", False) and not args.skip_metrics)
        )
        prior_logit_tau_config = metrics_config.get("prior_logit_tau", 0.0)
        if not evaluation_requested:
            prior_logit_tau = 0.0
            print("[calibration] Deferred until evaluation is run")
        elif args.prior_logit_tau is not None:
            prior_logit_tau = float(args.prior_logit_tau)
            print(f"[calibration] Using supplied prior_logit_tau={prior_logit_tau:.4f}")
        elif model_result["algorithm"] == "yoloctm" and str(prior_logit_tau_config).strip().lower() == "auto":
            candidates = [float(value) for value in metrics_config.get("prior_logit_tau_candidates", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7])]
            prior_logit_tau = select_yoloctm_prior_tau(
                model_result["checkpoint"],
                data_root,
                str(validate_config.get("split", "val")),
                train_device,
                train_imgsz,
                int(metrics_config.get("batch", train_config.get("batch", 64))),
                candidates,
            )
        else:
            prior_logit_tau = float(prior_logit_tau_config)
        if validate_config.get("enabled", True) and not args.skip_val:
            if model_result["algorithm"] == "yoloctm":
                evaluate_yoloctm(
                    model_result["checkpoint"],
                    data_root,
                    str(validate_config.get("split", "val")),
                    "validate",
                    train_device,
                    train_imgsz,
                    int(train_config.get("batch", 64)),
                    prior_logit_tau=prior_logit_tau,
                )
            else:
                evaluate(model_result["model"], data_root, str(validate_config.get("split", "val")), "validate")
        else:
            print("[validate] Skipped")

        if test_config.get("enabled", True) and not args.skip_test:
            if model_result["algorithm"] == "yoloctm":
                evaluate_yoloctm(
                    model_result["checkpoint"],
                    data_root,
                    str(test_config.get("split", "test")),
                    "test",
                    train_device,
                    train_imgsz,
                    int(train_config.get("batch", 64)),
                    prior_logit_tau=prior_logit_tau,
                )
            else:
                evaluate(model_result["model"], data_root, str(test_config.get("split", "test")), "test")
        else:
            print("[test] Skipped")

        if metrics_config.get("enabled", False) and not args.skip_metrics:
            metrics_batch = int(metrics_config.get("batch", train_config.get("batch", 64)))
            for split in metrics_config.get("splits", []):
                if model_result["algorithm"] == "yoloctm":
                    evaluate_yoloctm(
                        model_result["checkpoint"],
                        data_root,
                        str(split),
                        "metrics",
                        train_device,
                        train_imgsz,
                        metrics_batch,
                        write_metrics=True,
                        prior_logit_tau=prior_logit_tau,
                    )
                else:
                    compute_classification_metrics(run_dir, data_root, str(split), train_device, train_imgsz, metrics_batch)
        else:
            print("[metrics] Skipped")

        if should_log_autoresearch(config_path, config) and not args.skip_metrics:
            description = str(logging_config.get("description", "Fixed-budget YoloCTM baseline"))
            status = resolve_autoresearch_status(logging_config, run_dir)
            log_autoresearch_run(run_dir, status, description)
        elif should_log_autoresearch(config_path, config):
            print("[autoresearch] Deferred until metric reports are generated")

        print("\n[pipeline] Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
