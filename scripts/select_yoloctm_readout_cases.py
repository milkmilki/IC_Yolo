from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select representative validation-only YoloCTM readout overlay cases for paper/debug figures. "
            "Input is the index.json produced by visualize_yoloctm_readout_maps.py."
        )
    )
    parser.add_argument("--visualization-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-class-correct", type=int, default=2)
    parser.add_argument("--per-class-error", type=int, default=2)
    parser.add_argument("--low-confidence", type=int, default=12)
    return parser.parse_args()


def load_index(visualization_dir: Path) -> dict[str, Any]:
    index_path = visualization_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    source_export_dir = Path(str(index["source_export_dir"]))
    if not source_export_dir.is_absolute():
        source_export_dir = (visualization_dir.parent / source_export_dir).resolve()
        if not source_export_dir.exists():
            source_export_dir = (Path.cwd() / str(index["source_export_dir"])).resolve()
    summary_path = source_export_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(summary.get("split", "")).lower() == "test":
        raise ValueError("Refusing to select cases from test readout exports; use validation-only diagnostics.")
    index["source_summary"] = summary
    return index


def sorted_correct(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (record for record in records if record["true_class"] == record["pred_class"]),
        key=lambda record: float(record["pred_confidence"]),
        reverse=True,
    )


def sorted_errors(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (record for record in records if record["true_class"] != record["pred_class"]),
        key=lambda record: float(record["pred_confidence"]),
        reverse=True,
    )


def brief_record(record: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "sample_id",
        "image_path",
        "true_class",
        "pred_class",
        "pred_confidence",
        "readout_kind",
        "rendered_class_label",
        "grid_shape",
        "png",
    ]
    return {key: record[key] for key in keys if key in record}


def markdown_table(title: str, records: list[dict[str, Any]]) -> list[str]:
    lines = [f"## {title}", "", "| sample | true | pred | conf | overlay |", "|---|---|---|---:|---|"]
    for record in records:
        lines.append(
            "| {sample_id} | {true_class} | {pred_class} | {conf:.4f} | {png} |".format(
                sample_id=record["sample_id"],
                true_class=record["true_class"],
                pred_class=record["pred_class"],
                conf=float(record["pred_confidence"]),
                png=record["png"],
            )
        )
    if not records:
        lines.append("| n/a | n/a | n/a | n/a | n/a |")
    lines.append("")
    return lines


def main() -> int:
    args = parse_args()
    if int(args.per_class_correct) < 0 or int(args.per_class_error) < 0 or int(args.low_confidence) < 0:
        raise ValueError("Selection counts must be non-negative")

    index = load_index(args.visualization_dir)
    records = [dict(record) for record in index.get("records", [])]
    by_true_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_true_class[str(record["true_class"])].append(record)

    selected: dict[str, Any] = {
        "source_visualization_dir": str(args.visualization_dir),
        "source_export_dir": str(index["source_export_dir"]),
        "split": index["source_summary"].get("split"),
        "per_class": {},
    }

    for class_name in sorted(by_true_class):
        class_records = by_true_class[class_name]
        selected["per_class"][class_name] = {
            "correct_high_confidence": [
                brief_record(record) for record in sorted_correct(class_records)[: int(args.per_class_correct)]
            ],
            "errors_high_confidence": [
                brief_record(record) for record in sorted_errors(class_records)[: int(args.per_class_error)]
            ],
        }

    low_confidence_records = sorted(records, key=lambda record: float(record["pred_confidence"]))[
        : int(args.low_confidence)
    ]
    selected["low_confidence"] = [brief_record(record) for record in low_confidence_records]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selected_cases.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md_lines = [
        "# YoloCTM Readout Case Selection",
        "",
        f"Source visualization dir: `{args.visualization_dir}`",
        f"Source split: `{selected['split']}`",
        "",
    ]
    for class_name, groups in selected["per_class"].items():
        md_lines.extend(markdown_table(f"{class_name}: correct high-confidence", groups["correct_high_confidence"]))
        md_lines.extend(markdown_table(f"{class_name}: high-confidence errors", groups["errors_high_confidence"]))
    md_lines.extend(markdown_table("Lowest-confidence predictions", selected["low_confidence"]))
    (args.output_dir / "selected_cases.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "source_visualization_dir": str(args.visualization_dir),
                "split": selected["split"],
                "classes": len(selected["per_class"]),
                "low_confidence": len(selected["low_confidence"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
