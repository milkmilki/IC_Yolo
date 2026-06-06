from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_WM811K_CLASSES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit an external wafer-map dataset layout before any cross-dataset evaluation. "
            "This is a metadata-only check: it counts files/classes and compares label names; "
            "it does not load a model or compute performance metrics."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-classes", nargs="*", default=DEFAULT_WM811K_CLASSES)
    parser.add_argument(
        "--include-test-metadata",
        action="store_true",
        help="Include folders named 'test' in metadata counts. Default skips them to avoid accidental test feedback.",
    )
    return parser.parse_args()


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def looks_like_split(path: Path) -> bool:
    child_dirs = [child for child in path.iterdir() if child.is_dir()]
    return any(child.name.lower() in {"train", "val", "valid", "validation", "test"} for child in child_dirs)


def collect_imagefolder_counts(root: Path, include_test_metadata: bool) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    if looks_like_split(root):
        split_dirs = [child for child in root.iterdir() if child.is_dir()]
    else:
        split_dirs = [root]

    for split_dir in sorted(split_dirs, key=lambda item: item.name.lower()):
        split_name = split_dir.name if split_dir != root else "all"
        if split_name.lower() == "test" and not include_test_metadata:
            continue
        class_dirs = [child for child in split_dir.iterdir() if child.is_dir()]
        for class_dir in sorted(class_dirs, key=lambda item: item.name.lower()):
            image_count = sum(1 for item in class_dir.rglob("*") if is_image(item))
            if image_count:
                counts[split_name][class_dir.name] += image_count
    return counts


def summarize(counts: dict[str, Counter[str]], reference_classes: list[str]) -> dict[str, object]:
    total_by_class: Counter[str] = Counter()
    for split_counts in counts.values():
        total_by_class.update(split_counts)

    reference_casefold = {name.casefold(): name for name in reference_classes}
    exact_overlap = sorted(set(total_by_class) & set(reference_classes))
    casefold_overlap = {
        label: reference_casefold[label.casefold()]
        for label in sorted(total_by_class)
        if label.casefold() in reference_casefold and label not in exact_overlap
    }
    unmapped_external = [
        label
        for label in sorted(total_by_class)
        if label not in exact_overlap and label not in casefold_overlap
    ]
    missing_reference = [
        label
        for label in reference_classes
        if label not in exact_overlap and label.casefold() not in {name.casefold() for name in total_by_class}
    ]

    return {
        "total_images": int(sum(total_by_class.values())),
        "num_external_classes": len(total_by_class),
        "reference_classes": reference_classes,
        "splits": {split: dict(counter) for split, counter in counts.items()},
        "total_by_class": dict(total_by_class),
        "exact_label_overlap": exact_overlap,
        "casefold_label_overlap": casefold_overlap,
        "unmapped_external_labels": unmapped_external,
        "missing_reference_labels": missing_reference,
        "policy_note": (
            "Metadata-only audit. Do not report cross-dataset performance until label mapping, "
            "single-label vs multi-label semantics, and evaluation split policy are declared."
        ),
    }


def write_markdown(summary: dict[str, object], output_path: Path) -> None:
    lines = [
        "# External Wafer Dataset Audit",
        "",
        f"- total images: `{summary['total_images']}`",
        f"- external classes: `{summary['num_external_classes']}`",
        f"- exact WM811K label overlap: `{len(summary['exact_label_overlap'])}`",
        f"- unmapped external labels: `{len(summary['unmapped_external_labels'])}`",
        "",
        "## Split Counts",
        "",
    ]
    splits = summary["splits"]
    assert isinstance(splits, dict)
    for split, counter in splits.items():
        assert isinstance(counter, dict)
        lines.append(f"### {split}")
        lines.append("")
        lines.append("| class | images |")
        lines.append("|---|---:|")
        for label, count in sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0]).lower())):
            lines.append(f"| `{label}` | {count} |")
        lines.append("")

    lines.extend(
        [
            "## Label Mapping Status",
            "",
            f"- exact overlap: `{summary['exact_label_overlap']}`",
            f"- casefold overlap: `{summary['casefold_label_overlap']}`",
            f"- unmapped external labels: `{summary['unmapped_external_labels']}`",
            f"- missing WM811K reference labels: `{summary['missing_reference_labels']}`",
            "",
            "## Policy",
            "",
            str(summary["policy_note"]),
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.dataset_root
    if not root.exists():
        raise FileNotFoundError(f"External dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"External dataset root is not a directory: {root}")

    counts = collect_imagefolder_counts(root, bool(args.include_test_metadata))
    summary = summarize(counts, list(args.reference_classes))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "external_dataset_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(summary, args.output_dir / "external_dataset_audit.md")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
