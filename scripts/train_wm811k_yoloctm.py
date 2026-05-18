from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_dim, d_model)
        self.ctm_blocks = nn.ModuleList(CTMBlock(d_model=d_model, dropout=dropout) for _ in range(steps))
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images)
        if feats.ndim == 2:
            token = feats
        else:
            token = self.pool(feats).flatten(1)
        state = self.proj(token)
        for block in self.ctm_blocks:
            state = block(state, state)
        return self.cls(state)


def build_yolo_backbone(model_name: str) -> tuple[nn.Module, int]:
    from ultralytics import YOLO

    ul_model = YOLO(model_name).model
    layers = list(ul_model.model)
    if len(layers) < 2:
        raise RuntimeError("Unexpected YOLO classification architecture")
    backbone = nn.Sequential(*layers[:-1])

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        feats = backbone(dummy)
        in_dim = feats.shape[1] if feats.ndim > 2 else feats.shape[-1]
    return backbone, int(in_dim)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YoloCTM on WM-811K")
    parser.add_argument("--data", type=Path, default=Path("data/wm811k_cls"))
    parser.add_argument("--model", default="yolov8n-cls.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    device = torch.device(args.device)

    transform = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(args.data / "train", transform=transform)
    val_ds = datasets.ImageFolder(args.data / "val", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    backbone, in_dim = build_yolo_backbone(args.model)
    model = YoloCTM(backbone=backbone, num_classes=len(train_ds.classes), in_dim=in_dim, d_model=args.d_model, steps=args.steps).to(device)

    class_counts = torch.bincount(torch.tensor(train_ds.targets), minlength=len(train_ds.classes)).float()
    class_weights = (class_counts.sum() / class_counts.clamp(min=1.0)).to(device)
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
            torch.save(model.state_dict(), out_dir / "best_yoloctm.pt")

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"best_val_acc": best_val_acc, "history": history, "classes": train_ds.classes}, f, indent=2)

    print(f"Saved best model to {out_dir / 'best_yoloctm.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
