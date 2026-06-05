from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


VAL_FIELDS = ("val_acc", "val_macro_p", "val_macro_r", "val_macro_f1")
TEST_FIELDS = ("test_acc", "test_macro_p", "test_macro_r", "test_macro_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize AutoResearch validation-only YoloCTM ablations into manuscript-ready "
            "Markdown/CSV/JSON tables without reading test metrics."
        )
    )
    parser.add_argument("--results", type=Path, default=Path("AutoResearch/results.tsv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--require-val-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keyword", default="nodistill")
    parser.add_argument("--top-k", type=int, default=24)
    return parser.parse_args()


def to_float(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def to_int(value: str | None) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    return int(number)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader if any((value or "").strip() for value in row.values())]


def is_val_only(row: dict[str, str]) -> bool:
    return all((row.get(field, "") or "").strip() == "" for field in TEST_FIELDS)


def eligible(row: dict[str, str], args: argparse.Namespace) -> bool:
    if row.get("algorithm") != "yoloctm":
        return False
    epochs = to_int(row.get("epochs"))
    if epochs is None or epochs > int(args.max_epochs):
        return False
    if to_float(row.get("val_macro_f1")) is None:
        return False
    if bool(args.require_val_only) and not is_val_only(row):
        return False
    keyword = str(args.keyword).strip().lower()
    if keyword:
        blob = " ".join([row.get("run_dir", ""), row.get("model_summary", ""), row.get("description", "")]).lower()
        if keyword not in blob:
            return False
    return True


def short_run_name(run_dir: str) -> str:
    return Path(run_dir).name or run_dir


def compact_description(description: str, limit: int = 150) -> str:
    description = " ".join(str(description).split())
    if len(description) <= limit:
        return description
    return description[: limit - 3] + "..."


def summarize_row(row: dict[str, str], best_f1: float) -> dict[str, Any]:
    val_f1 = float(to_float(row.get("val_macro_f1")) or 0.0)
    return {
        "timestamp": row.get("timestamp", ""),
        "status": row.get("status", ""),
        "run": short_run_name(row.get("run_dir", "")),
        "params_m": to_float(row.get("params_m")),
        "epochs": to_int(row.get("epochs")),
        "val_acc": to_float(row.get("val_acc")),
        "val_macro_p": to_float(row.get("val_macro_p")),
        "val_macro_r": to_float(row.get("val_macro_r")),
        "val_macro_f1": val_f1,
        "delta_to_best": val_f1 - best_f1,
        "model_summary": row.get("model_summary", ""),
        "description": compact_description(row.get("description", "")),
    }


def format_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "timestamp",
        "status",
        "run",
        "params_m",
        "epochs",
        "val_acc",
        "val_macro_p",
        "val_macro_r",
        "val_macro_f1",
        "delta_to_best",
        "model_summary",
        "description",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows: list[dict[str, Any]], path: Path, metadata: dict[str, Any]) -> None:
    lines = [
        "# AutoResearch Validation-Only Ablation Table",
        "",
        "This table is generated from `AutoResearch/results.tsv` and excludes rows with test metrics by default.",
        "",
        "## Filters",
        "",
        f"- keyword: `{metadata['keyword']}`",
        f"- max epochs: `{metadata['max_epochs']}`",
        f"- require val-only: `{metadata['require_val_only']}`",
        f"- selected rows: `{metadata['selected_rows']}`",
        f"- best val macro F1: `{metadata['best_val_macro_f1']:.12f}`",
        "",
        "## Table",
        "",
        "| rank | status | run | params M | val macro F1 | delta | val P | val R | note |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {status} | `{run}` | {params} | {f1} | {delta} | {prec} | {rec} | {note} |".format(
                rank=rank,
                status=row["status"],
                run=row["run"],
                params=format_float(row["params_m"], 3),
                f1=format_float(row["val_macro_f1"]),
                delta=format_float(row["delta_to_best"]),
                prec=format_float(row["val_macro_p"]),
                rec=format_float(row["val_macro_r"]),
                note=str(row["description"]).replace("|", "/"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = load_rows(args.results)
    selected_source = [row for row in rows if eligible(row, args)]
    if not selected_source:
        raise ValueError("No eligible rows found for the requested filters")
    selected_source.sort(key=lambda row: float(to_float(row.get("val_macro_f1")) or -1.0), reverse=True)
    best_f1 = float(to_float(selected_source[0].get("val_macro_f1")) or 0.0)
    summary_rows = [summarize_row(row, best_f1) for row in selected_source[: int(args.top_k)]]

    metadata = {
        "results": str(args.results),
        "keyword": str(args.keyword),
        "max_epochs": int(args.max_epochs),
        "require_val_only": bool(args.require_val_only),
        "selected_rows": len(selected_source),
        "emitted_rows": len(summary_rows),
        "best_val_macro_f1": best_f1,
        "best_run": summary_rows[0]["run"],
        "note": "Validation-only ablation summary; test metric columns are excluded by default.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary_rows, args.output_dir / "ablation_table.csv")
    write_markdown(summary_rows, args.output_dir / "ablation_table.md", metadata)
    (args.output_dir / "ablation_table.json").write_text(
        json.dumps({"metadata": metadata, "rows": summary_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
