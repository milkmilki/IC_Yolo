from __future__ import annotations

import argparse
import json
import math
import importlib
import random
import shutil
import sys
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


CLASS_NAMES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Random",
    "Scratch",
    "Near-full",
    "none",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare WM-811K for YOLO classification")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Source directory containing LSWMD_compat.pkl or an extracted WM-811K archive",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "wm811k_cls",
        help="Output directory for the YOLO classification dataset",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(60.0, 15.0, 25.0),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios like [x,y,z]; values are normalized automatically.",
    )
    parser.add_argument(
        "--include-none",
        action="store_true",
        help="Keep the none class (label 8). By default it is excluded to match the common 8-class setup.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory")
    parser.add_argument("--image-size", type=int, default=224, help="Optional square resize before saving")
    return parser.parse_args()


def normalize_ratios(ratios: tuple[float, float, float]) -> tuple[float, float, float]:
    total = sum(ratios)
    if total <= 0:
        raise ValueError("Ratios must sum to a positive number.")
    return tuple(r / total for r in ratios)


def compute_split_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    normalized = normalize_ratios(ratios)
    raw_counts = [total * ratio for ratio in normalized]
    counts = [math.floor(value) for value in raw_counts]
    remainder = total - sum(counts)
    order = sorted(range(3), key=lambda idx: (raw_counts[idx] - counts[idx], -idx), reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1
    if sum(counts) != total:
        counts[0] += total - sum(counts)
    return counts[0], counts[1], counts[2]


def find_lswmd_pickle(source_root: Path) -> Path:
    candidates = [
        source_root / "LSWMD.pkl",
        source_root / "dataset" / "LSWMD.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for candidate in source_root.rglob("LSWMD.pkl"):
        return candidate

    raise FileNotFoundError(f"Could not locate LSWMD.pkl under {source_root}")


def install_pickle_compatibility_shims() -> None:
    """Install import aliases for old pandas pickle module paths."""

    aliases = {
        "pandas.indexes": "pandas.core.indexes",
        "pandas.indexes.base": "pandas.core.indexes.base",
        "pandas.indexes.datetimes": "pandas.core.indexes.datetimes",
        "pandas.indexes.interval": "pandas.core.indexes.interval",
        "pandas.indexes.multi": "pandas.core.indexes.multi",
        "pandas.indexes.period": "pandas.core.indexes.period",
        "pandas.indexes.range": "pandas.core.indexes.range",
        "pandas.indexes.timedeltas": "pandas.core.indexes.timedeltas",
        "pandas.indexes.numeric": "pandas.core.indexes.numeric",
    }

    for old_name, new_name in aliases.items():
        try:
            sys.modules.setdefault(old_name, importlib.import_module(new_name))
        except Exception:
            continue


def load_legacy_pickle(pickle_path: Path):
    with pickle_path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def unwrap_scalar(value):
    current = value
    while isinstance(current, np.ndarray) and current.size == 1:
        current = current.item()
    return current


def normalize_label(value) -> str | None:
    current = unwrap_scalar(value)
    if current is None:
        return None
    if isinstance(current, bytes):
        current = current.decode("utf-8", errors="ignore")
    if isinstance(current, str):
        text = current.strip()
        if text.startswith("[[") and text.endswith("]]"):
            text = text[2:-2].strip()
        if text in CLASS_NAMES:
            return text
        return text if text else None
    if isinstance(current, (int, np.integer)):
        index = int(current)
        if 0 <= index < len(CLASS_NAMES):
            return CLASS_NAMES[index]
    return None


def extract_label_and_map(row: pd.Series) -> tuple[str | None, np.ndarray | None]:
    failure_type = normalize_label(row.get("failureType"))
    wafer_map_raw = row.get("waferMap")
    if wafer_map_raw is None:
        return failure_type, None
    wafer_map = np.asarray(wafer_map_raw)
    if wafer_map.ndim != 2:
        return failure_type, None
    return failure_type, wafer_map


def save_wafer_map(wafer_map: np.ndarray, dst: Path, image_size: int) -> None:
    wafer = np.asarray(wafer_map)
    if wafer.size == 0:
        return

    wafer = wafer.astype(np.float32)
    wafer = np.nan_to_num(wafer, nan=0.0)
    max_value = float(np.max(wafer))
    if max_value > 0:
        wafer = wafer / max_value * 255.0
    wafer = np.clip(wafer, 0.0, 255.0).astype(np.uint8)

    image = Image.fromarray(wafer, mode="L")
    if image_size > 0:
        image = image.resize((image_size, image_size), Image.Resampling.NEAREST)
    image.save(dst)


def clear_output(output_root: Path, overwrite: bool, include_none: bool) -> None:
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    for split_name in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            if class_name == "none" and not include_none:
                continue
            (output_root / split_name / class_name).mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    source_root = args.source.resolve()
    output_root = args.output.resolve()
    pickle_path = find_lswmd_pickle(source_root)

    clear_output(output_root, args.overwrite, args.include_none)

    install_pickle_compatibility_shims()

    df = load_legacy_pickle(pickle_path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame in {pickle_path}, got {type(df).__name__}")

    samples_by_class: dict[str, list[np.ndarray]] = defaultdict(list)
    skipped = 0
    for _, row in df.iterrows():
        failure_type, wafer_map = extract_label_and_map(row)
        if failure_type is None or wafer_map is None:
            skipped += 1
            continue
        if failure_type not in CLASS_NAMES:
            skipped += 1
            continue
        if failure_type == "none" and not args.include_none:
            continue
        samples_by_class[failure_type].append(wafer_map)

    if not samples_by_class:
        raise FileNotFoundError("No usable WM-811K samples were found in the loaded dataframe.")

    rng = random.Random(args.seed)
    summary: dict[str, dict[str, int]] = {}
    total_written = 0

    for class_name in CLASS_NAMES:
        if class_name not in samples_by_class:
            continue

        samples = samples_by_class[class_name][:]
        rng.shuffle(samples)
        train_count, val_count, test_count = compute_split_counts(len(samples), tuple(args.ratios))
        split_samples = {
            "train": samples[:train_count],
            "val": samples[train_count : train_count + val_count],
            "test": samples[train_count + val_count : train_count + val_count + test_count],
        }

        summary[class_name] = {name: len(items) for name, items in split_samples.items()}
        for split_name, items in split_samples.items():
            for index, wafer_map in enumerate(items):
                filename = f"{class_name.replace('-', '_').lower()}_{index:06d}.png"
                dst = output_root / split_name / class_name / filename
                save_wafer_map(wafer_map, dst, args.image_size)
                total_written += 1

    meta = {
        "source": str(pickle_path),
        "output": str(output_root),
        "include_none": args.include_none,
        "ratios": normalize_ratios(tuple(args.ratios)),
        "classes": [name for name in CLASS_NAMES if name in samples_by_class and (args.include_none or name != "none")],
        "per_class_counts": summary,
        "total_written": total_written,
        "skipped_rows": skipped,
    }
    (output_root / "dataset_summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Loaded: {pickle_path}")
    print(f"Output: {output_root}")
    print(f"Normalized ratios: {meta['ratios']}")
    print(f"Total written: {total_written}")
    print(f"Summary saved: {output_root / 'dataset_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





