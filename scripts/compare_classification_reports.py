from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_KEYS = {"accuracy", "macro avg", "weighted avg", "micro avg", "samples avg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two sklearn-style classification_report JSON files and emit class-wise "
            "precision/recall/F1 deltas for validation-only ablation analysis."
        )
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if "macro avg" not in report:
        raise ValueError(f"{path} does not look like a sklearn classification_report JSON")
    return report


def class_names(report: dict[str, Any]) -> list[str]:
    return [key for key, value in report.items() if key not in SUMMARY_KEYS and isinstance(value, dict)]


def metric(report: dict[str, Any], name: str, field: str) -> float:
    return float(report[name].get(field, 0.0))


def support(report: dict[str, Any], name: str) -> int:
    return int(round(float(report[name].get("support", 0.0))))


def build_rows(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    names = class_names(candidate)
    missing = [name for name in names if name not in baseline]
    if missing:
        raise ValueError(f"Classes missing from baseline report: {missing}")
    rows: list[dict[str, Any]] = []
    for name in names:
        base_f1 = metric(baseline, name, "f1-score")
        cand_f1 = metric(candidate, name, "f1-score")
        rows.append(
            {
                "class": name,
                "support": support(candidate, name),
                "baseline_precision": metric(baseline, name, "precision"),
                "candidate_precision": metric(candidate, name, "precision"),
                "delta_precision": metric(candidate, name, "precision") - metric(baseline, name, "precision"),
                "baseline_recall": metric(baseline, name, "recall"),
                "candidate_recall": metric(candidate, name, "recall"),
                "delta_recall": metric(candidate, name, "recall") - metric(baseline, name, "recall"),
                "baseline_f1": base_f1,
                "candidate_f1": cand_f1,
                "delta_f1": cand_f1 - base_f1,
            }
        )
    rows.sort(key=lambda row: row["delta_f1"], reverse=True)
    return rows


def summary(baseline: dict[str, Any], candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "baseline_name": str(args.baseline_name),
        "candidate_name": str(args.candidate_name),
        "baseline_macro_f1": metric(baseline, "macro avg", "f1-score"),
        "candidate_macro_f1": metric(candidate, "macro avg", "f1-score"),
        "delta_macro_f1": metric(candidate, "macro avg", "f1-score") - metric(baseline, "macro avg", "f1-score"),
        "baseline_accuracy": float(baseline.get("accuracy", 0.0)),
        "candidate_accuracy": float(candidate.get("accuracy", 0.0)),
        "delta_accuracy": float(candidate.get("accuracy", 0.0)) - float(baseline.get("accuracy", 0.0)),
    }


def format_float(value: float) -> str:
    return f"{float(value):.6f}"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], metadata: dict[str, Any], path: Path) -> None:
    lines = [
        "# Classification Report Delta",
        "",
        f"Baseline: `{metadata['baseline_name']}`",
        f"Candidate: `{metadata['candidate_name']}`",
        "",
        "| metric | baseline | candidate | delta |",
        "|---|---:|---:|---:|",
        "| accuracy | {b} | {c} | {d} |".format(
            b=format_float(metadata["baseline_accuracy"]),
            c=format_float(metadata["candidate_accuracy"]),
            d=format_float(metadata["delta_accuracy"]),
        ),
        "| macro F1 | {b} | {c} | {d} |".format(
            b=format_float(metadata["baseline_macro_f1"]),
            c=format_float(metadata["candidate_macro_f1"]),
            d=format_float(metadata["delta_macro_f1"]),
        ),
        "",
        "## Class Deltas",
        "",
        "| class | support | base F1 | cand F1 | delta F1 | delta P | delta R |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {cls} | {support} | {base_f1} | {cand_f1} | {delta_f1} | {delta_p} | {delta_r} |".format(
                cls=row["class"],
                support=row["support"],
                base_f1=format_float(row["baseline_f1"]),
                cand_f1=format_float(row["candidate_f1"]),
                delta_f1=format_float(row["delta_f1"]),
                delta_p=format_float(row["delta_precision"]),
                delta_r=format_float(row["delta_recall"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    baseline = load_report(args.baseline)
    candidate = load_report(args.candidate)
    rows = build_rows(baseline, candidate)
    metadata = summary(baseline, candidate, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "class_delta.csv")
    write_markdown(rows, metadata, args.output_dir / "class_delta.md")
    (args.output_dir / "class_delta.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
