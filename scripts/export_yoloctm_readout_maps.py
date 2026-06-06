from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from run_wm811k_pipeline import load_yoloctm_checkpoint
from train_wm811k_yoloctm import WaferToTensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export validation-only CTM readout evidence maps from a trained YoloCTM checkpoint. "
            "The script reads model-cached readout weights and does not change the checkpoint or training objective."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/wm811k_cls"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--prior-logit-tau", type=float, default=0.4)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument(
        "--per-class-samples",
        type=int,
        default=0,
        help="If positive, export an interleaved validation subset with up to N samples per class.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def infer_grid(token_count: int) -> tuple[int, int]:
    height = int(round(float(token_count) ** 0.5))
    if height > 0 and height * height == token_count:
        return height, height
    for height in range(int(float(token_count) ** 0.5), 0, -1):
        if token_count % height == 0:
            return height, token_count // height
    return 1, token_count


def polar_coordinates(height: int, width: int) -> np.ndarray:
    y = np.linspace(-1.0, 1.0, num=height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, num=width, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    radius = np.sqrt(np.square(xx) + np.square(yy)) / np.sqrt(np.float32(2.0))
    return np.stack([xx, yy, radius], axis=-1).reshape(-1, 3).astype(np.float32)


def select_sample_indices(dataset: datasets.ImageFolder, max_samples: int, per_class_samples: int) -> list[int]:
    max_samples = int(max_samples)
    per_class_samples = int(per_class_samples)
    if per_class_samples <= 0:
        return list(range(min(max_samples, len(dataset))))

    by_class: list[list[int]] = [[] for _ in dataset.classes]
    for index, target in enumerate(dataset.targets):
        bucket = by_class[int(target)]
        if len(bucket) < per_class_samples:
            bucket.append(index)

    selected: list[int] = []
    for offset in range(per_class_samples):
        for bucket in by_class:
            if offset < len(bucket):
                selected.append(bucket[offset])
                if len(selected) >= max_samples:
                    return selected
    return selected


def main() -> int:
    args = parse_args()
    if str(args.split).lower() == "test":
        raise ValueError("This diagnostic is validation-only by policy; do not export test readout maps.")
    if int(args.max_samples) <= 0:
        raise ValueError("--max-samples must be positive")
    if int(args.per_class_samples) < 0:
        raise ValueError("--per-class-samples must be non-negative")

    transform = transforms.Compose(
        [
            transforms.Resize((int(args.imgsz), int(args.imgsz)), interpolation=InterpolationMode.NEAREST),
            WaferToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(args.data_root / str(args.split), transform=transform)
    selected_indices = select_sample_indices(dataset, int(args.max_samples), int(args.per_class_samples))
    export_dataset = Subset(dataset, selected_indices)
    loader = DataLoader(export_dataset, batch_size=int(args.batch), shuffle=False, num_workers=0, pin_memory=False)

    model, checkpoint_classes, device, _criterion = load_yoloctm_checkpoint(args.checkpoint, str(args.device))
    if dataset.classes != checkpoint_classes:
        raise ValueError(f"Dataset classes do not match checkpoint classes: {dataset.classes} != {checkpoint_classes}")
    model.eval()
    model_readout = str(getattr(model, "ctm_readout", "unknown"))
    polar_params = getattr(model, "class_readout_polar", None)
    polar_params_np = (
        polar_params.detach().cpu().numpy().astype(np.float32)
        if polar_params is not None
        else None
    )

    log_prior = None
    if float(args.prior_logit_tau) != 0.0:
        train_dataset = datasets.ImageFolder(args.data_root / "train", transform=transform)
        if train_dataset.classes != checkpoint_classes:
            raise ValueError(f"Train classes do not match checkpoint classes: {train_dataset.classes} != {checkpoint_classes}")
        class_counts = torch.bincount(torch.tensor(train_dataset.targets), minlength=len(checkpoint_classes)).float()
        log_prior = class_counts.clamp(min=1.0).log().to(device)

    exported = 0
    sample_records: list[dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            if exported >= int(args.max_samples):
                break
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            if log_prior is not None:
                logits = logits + float(args.prior_logit_tau) * log_prior.unsqueeze(0)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            class_weights = getattr(model, "last_class_readout_weights", None)
            shared_weights = getattr(model, "last_readout_weights", None)
            if class_weights is not None:
                weights_np = class_weights.detach().cpu().numpy().astype(np.float32)
                readout_kind = (
                    "class_attention_polar"
                    if model_readout == "class_attention_polar"
                    else "class_attention_blend"
                    if model_readout == "class_attention_blend"
                    else "class_attention"
                )
                token_count = int(weights_np.shape[1])
                grid_h, grid_w = infer_grid(token_count)
            elif shared_weights is not None:
                weights_np = shared_weights.detach().cpu().numpy().astype(np.float32)
                readout_kind = "attention"
                token_count = int(weights_np.shape[1])
                grid_h, grid_w = infer_grid(token_count)
            else:
                weights_np = np.empty((labels.shape[0], 0), dtype=np.float32)
                readout_kind = "mean"
                token_count = 0
                grid_h, grid_w = 0, 0

            batch_size = int(labels.shape[0])
            for offset in range(batch_size):
                if exported >= int(args.max_samples):
                    break
                dataset_index = selected_indices[batch_index * int(args.batch) + offset]
                path, _target = dataset.samples[dataset_index]
                sample_id = f"{exported:05d}"
                weight_file = args.output_dir / f"{sample_id}_readout_weights.npz"
                if readout_kind in {"class_attention", "class_attention_blend"}:
                    np.savez_compressed(
                        weight_file,
                        class_readout_weights=weights_np[offset],
                        grid_shape=np.array([grid_h, grid_w], dtype=np.int16),
                    )
                elif readout_kind == "class_attention_polar":
                    np.savez_compressed(
                        weight_file,
                        class_readout_weights=weights_np[offset],
                        grid_shape=np.array([grid_h, grid_w], dtype=np.int16),
                        polar_coordinates=polar_coordinates(grid_h, grid_w),
                        class_readout_polar=polar_params_np,
                    )
                elif readout_kind == "attention":
                    np.savez_compressed(
                        weight_file,
                        readout_weights=weights_np[offset],
                        grid_shape=np.array([grid_h, grid_w], dtype=np.int16),
                    )
                else:
                    np.savez_compressed(
                        weight_file,
                        readout_weights=weights_np[offset],
                        grid_shape=np.array([grid_h, grid_w], dtype=np.int16),
                    )
                sample_records.append(
                    {
                        "sample_id": sample_id,
                        "image_path": str(path),
                        "true_index": int(labels[offset].item()),
                        "true_class": dataset.classes[int(labels[offset].item())],
                        "pred_index": int(preds[offset].item()),
                        "pred_class": dataset.classes[int(preds[offset].item())],
                        "pred_confidence": float(probs[offset, int(preds[offset].item())].item()),
                        "readout_kind": readout_kind,
                        "token_count": token_count,
                        "grid_shape": [grid_h, grid_w],
                        "weights_file": weight_file.name,
                    }
                )
                exported += 1

    summary = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "split": str(args.split),
        "classes": dataset.classes,
        "model_readout": model_readout,
        "prior_logit_tau": float(args.prior_logit_tau),
        "per_class_samples": int(args.per_class_samples),
        "exported_samples": exported,
        "note": "Validation-only CTM readout evidence export; test split is intentionally rejected.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "samples.json").write_text(
        json.dumps(sample_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
