from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


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


class YoloCTM(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        in_dim: int,
        d_model: int = 256,
        steps: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.token_proj = nn.Linear(in_dim, d_model)
        self.ctm_blocks = nn.ModuleList(CTMBlock(d_model=d_model, dropout=dropout) for _ in range(steps))
        self.norm = nn.LayerNorm(d_model)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images)
        if feats.ndim == 2:
            tokens = feats.unsqueeze(1)
        else:
            tokens = feats.flatten(2).transpose(1, 2)

        inputs = self.token_proj(tokens)
        state = torch.zeros_like(inputs)
        for block in self.ctm_blocks:
            state = block(state, inputs)

        pooled = self.norm(state).mean(dim=1)
        return self.cls(pooled)


def build_yolo_backbone(
    model_config: str | None,
    weights: str | None,
    pretrained: bool,
    imgsz: int,
) -> tuple[nn.Module, int]:
    from ultralytics import YOLO

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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--project", default="runs/classify")
    parser.add_argument("--name", default="wm811k_yoloctm")
    return parser.parse_args()


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)

    train_transform = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz), interpolation=InterpolationMode.NEAREST),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180, interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(args.data / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(args.data / "val", transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    weights = args.model if args.model is not None else args.weights
    backbone, in_dim = build_yolo_backbone(args.model_config, weights, args.pretrained, args.imgsz)
    model = YoloCTM(
        backbone=backbone,
        num_classes=len(train_ds.classes),
        in_dim=in_dim,
        d_model=args.d_model,
        steps=args.steps,
        dropout=args.dropout,
    ).to(device)

    class_counts = torch.bincount(torch.tensor(train_ds.targets), minlength=len(train_ds.classes)).float()
    class_weights = (class_counts.sum() / class_counts.clamp(min=1.0)).pow(args.class_weight_power)
    class_weights = (class_weights / class_weights.mean()).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": train_ds.classes,
                    "args": vars(args),
                    "in_dim": in_dim,
                    "best_val_acc": best_val_acc,
                },
                out_dir / "best_yoloctm.pt",
            )

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"best_val_acc": best_val_acc, "history": history, "classes": train_ds.classes}, f, indent=2)

    print(f"Saved best model to {out_dir / 'best_yoloctm.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
