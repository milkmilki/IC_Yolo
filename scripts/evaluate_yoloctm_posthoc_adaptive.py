from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from evaluate_wm811k_cls import write_report_csv
from run_wm811k_pipeline import load_yoloctm_checkpoint
from train_wm811k_yoloctm import WaferToTensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc adaptive-step evaluation for a trained YoloCTM checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/wm811k_cls"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--prior-logit-tau", type=float, default=0.4)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--min-steps", type=int, default=4)
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    transform = transforms.Compose(
        [
            transforms.Resize((int(args.imgsz), int(args.imgsz)), interpolation=InterpolationMode.NEAREST),
            WaferToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(args.data_root / str(args.split), transform=transform)
    loader = DataLoader(dataset, batch_size=int(args.batch), shuffle=False, num_workers=0, pin_memory=False)

    model, checkpoint_classes, device, criterion = load_yoloctm_checkpoint(args.checkpoint, str(args.device))
    if dataset.classes != checkpoint_classes:
        raise ValueError(f"Dataset classes do not match checkpoint classes: {dataset.classes} != {checkpoint_classes}")
    if int(args.min_steps) > int(args.max_steps):
        raise ValueError("--min-steps must be <= --max-steps")
    if not 0.0 < float(args.confidence_threshold) <= 1.0:
        raise ValueError("--confidence-threshold must be in (0, 1]")

    model.steps = int(args.max_steps)
    model.adaptive_steps_enabled = True
    model.adaptive_min_steps = int(args.min_steps)
    model.adaptive_confidence_threshold = float(args.confidence_threshold)
    model.eval()

    log_prior = None
    if float(args.prior_logit_tau) != 0.0:
        train_dataset = datasets.ImageFolder(args.data_root / "train", transform=transform)
        if train_dataset.classes != checkpoint_classes:
            raise ValueError(f"Train classes do not match checkpoint classes: {train_dataset.classes} != {checkpoint_classes}")
        class_counts = torch.bincount(torch.tensor(train_dataset.targets), minlength=len(checkpoint_classes)).float()
        log_prior = class_counts.clamp(min=1.0).log().to(device)

    total_loss = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    adaptive_step_sum = 0.0
    adaptive_step_total = 0
    adaptive_max_step_count = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            if log_prior is not None:
                logits = logits + float(args.prior_logit_tau) * log_prior.unsqueeze(0)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            steps_used = getattr(model, "last_adaptive_steps_used", None)
            if steps_used is not None:
                steps_cpu = steps_used.detach().cpu().float()
                adaptive_step_sum += steps_cpu.sum().item()
                adaptive_step_total += int(steps_cpu.numel())
                adaptive_max_step_count += int((steps_cpu >= int(args.max_steps)).sum().item())

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(dataset.classes))),
        target_names=dataset.classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(dataset.classes))))
    loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    adaptive_stats = {
        "adaptive_avg_steps": adaptive_step_sum / max(adaptive_step_total, 1),
        "adaptive_max_step_fraction": adaptive_max_step_count / max(adaptive_step_total, 1),
    }
    summary = {
        "checkpoint": str(args.checkpoint),
        "split": str(args.split),
        "loss": loss,
        "acc": acc,
        "prior_logit_tau": float(args.prior_logit_tau),
        "max_steps": int(args.max_steps),
        "min_steps": int(args.min_steps),
        "confidence_threshold": float(args.confidence_threshold),
        **adaptive_stats,
        "macro_avg": report["macro avg"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.split}_classification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report_csv(report, dataset.classes, args.output_dir / f"{args.split}_classification_report.csv")
    np.savetxt(args.output_dir / f"{args.split}_confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
