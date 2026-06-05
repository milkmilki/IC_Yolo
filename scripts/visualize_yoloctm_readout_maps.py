from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render PNG overlays from validation-only YoloCTM readout evidence exports. "
            "Use this after export_yoloctm_readout_maps.py to make paper/debug figures."
        )
    )
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-mode", choices=["pred", "true", "all"], default="pred")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--max-samples", type=int, default=64)
    return parser.parse_args()


def normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    min_value = float(np.nanmin(values)) if values.size else 0.0
    max_value = float(np.nanmax(values)) if values.size else 0.0
    if max_value <= min_value:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_value) / (max_value - min_value)


def colorize(values: np.ndarray) -> Image.Image:
    values = normalize(values)
    red = np.clip(2.0 * values, 0.0, 1.0)
    green = np.clip(2.0 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - values), 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")


def resize_heatmap(heatmap: Image.Image, size: tuple[int, int]) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return heatmap.resize(size, resample=resampling)


def load_sample_weight(export_dir: Path, sample: dict[str, object]) -> tuple[str, np.ndarray, tuple[int, int]]:
    weights_path = export_dir / str(sample["weights_file"])
    with np.load(weights_path) as data:
        grid_shape_raw = data["grid_shape"].astype(int).tolist()
        grid_shape = (int(grid_shape_raw[0]), int(grid_shape_raw[1]))
        if "class_readout_weights" in data:
            return "class_attention", data["class_readout_weights"].astype(np.float32), grid_shape
        if "readout_weights" in data:
            return "attention", data["readout_weights"].astype(np.float32), grid_shape
    raise ValueError(f"No supported readout weights found in {weights_path}")


def token_map(values: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    height, width = grid_shape
    if height <= 0 or width <= 0:
        raise ValueError("Cannot render mean readout without token weights")
    if values.size != height * width:
        raise ValueError(f"Weight size {values.size} does not match grid shape {grid_shape}")
    return values.reshape(height, width)


def render_overlay(image_path: Path, weights: np.ndarray, grid_shape: tuple[int, int], alpha: float) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    heatmap = resize_heatmap(colorize(token_map(weights, grid_shape)), image.size)
    return Image.blend(image, heatmap, alpha=float(alpha))


def selected_classes(sample: dict[str, object], classes: list[str], mode: str) -> list[tuple[int, str]]:
    if mode == "pred":
        index = int(sample["pred_index"])
        return [(index, f"pred_{classes[index]}")]
    if mode == "true":
        index = int(sample["true_index"])
        return [(index, f"true_{classes[index]}")]
    return [(index, f"class_{name}") for index, name in enumerate(classes)]


def safe_name(text: str) -> str:
    keep = []
    for char in text:
        keep.append(char if char.isalnum() or char in {"-", "_"} else "_")
    return "".join(keep)


def main() -> int:
    args = parse_args()
    if not 0.0 <= float(args.alpha) <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    if int(args.max_samples) <= 0:
        raise ValueError("--max-samples must be positive")

    summary = json.loads((args.export_dir / "summary.json").read_text(encoding="utf-8"))
    if str(summary.get("split", "")).lower() == "test":
        raise ValueError("Refusing to visualize test readout exports; use validation-only diagnostics.")
    classes = list(summary.get("classes", []))
    samples = json.loads((args.export_dir / "samples.json").read_text(encoding="utf-8"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for sample in samples[: int(args.max_samples)]:
        readout_kind, weights, grid_shape = load_sample_weight(args.export_dir, sample)
        if readout_kind == "class_attention":
            class_items = selected_classes(sample, classes, str(args.class_mode))
        elif readout_kind == "attention":
            class_items = [(-1, "shared_attention")]
        else:
            continue

        for class_index, label in class_items:
            if readout_kind == "class_attention":
                class_weights = weights[:, class_index]
            else:
                class_weights = weights
            image_path = Path(str(sample["image_path"]))
            overlay = render_overlay(image_path, class_weights, grid_shape, alpha=float(args.alpha))
            filename = f"{sample['sample_id']}_{safe_name(label)}.png"
            overlay.save(args.output_dir / filename)
            records.append(
                {
                    "sample_id": sample["sample_id"],
                    "image_path": str(image_path),
                    "true_class": sample["true_class"],
                    "pred_class": sample["pred_class"],
                    "pred_confidence": sample["pred_confidence"],
                    "readout_kind": readout_kind,
                    "rendered_class_index": class_index,
                    "rendered_class_label": label,
                    "grid_shape": list(grid_shape),
                    "png": filename,
                }
            )

    index = {
        "source_export_dir": str(args.export_dir),
        "output_dir": str(args.output_dir),
        "class_mode": str(args.class_mode),
        "alpha": float(args.alpha),
        "rendered_images": len(records),
        "records": records,
    }
    (args.output_dir / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: index[k] for k in ["source_export_dir", "output_dir", "class_mode", "rendered_images"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
