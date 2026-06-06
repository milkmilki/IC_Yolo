from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the validation-only YoloCTM readout figure pipeline: "
            "export weights, render overlays, select cases, and assemble a multi-panel figure."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/wm811k_cls"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--prior-logit-tau", type=float, default=0.4)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--per-class-samples", type=int, default=0)
    parser.add_argument("--class-mode", choices=["pred", "true", "all"], default="pred")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--per-class-correct", type=int, default=2)
    parser.add_argument("--per-class-error", type=int, default=2)
    parser.add_argument("--low-confidence", type=int, default=12)
    parser.add_argument("--max-panels", type=int, default=12)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=224)
    return parser.parse_args()


def run_step(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    if str(args.split).lower() == "test":
        raise ValueError("This figure pipeline is validation-only by policy; do not use --split test.")

    script_dir = Path(__file__).resolve().parent
    output_root = args.output_root
    export_dir = output_root / "readout_export"
    overlay_dir = output_root / "readout_overlays"
    selection_dir = output_root / "case_selection"
    figure_path = output_root / "readout_figure.png"
    output_root.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    run_step(
        [
            python,
            str(script_dir / "export_yoloctm_readout_maps.py"),
            "--checkpoint",
            str(args.checkpoint),
            "--data-root",
            str(args.data_root),
            "--split",
            str(args.split),
            "--device",
            str(args.device),
            "--imgsz",
            str(args.imgsz),
            "--batch",
            str(args.batch),
            "--prior-logit-tau",
            str(args.prior_logit_tau),
            "--max-samples",
            str(args.max_samples),
            "--per-class-samples",
            str(args.per_class_samples),
            "--output-dir",
            str(export_dir),
        ]
    )
    run_step(
        [
            python,
            str(script_dir / "visualize_yoloctm_readout_maps.py"),
            "--export-dir",
            str(export_dir),
            "--output-dir",
            str(overlay_dir),
            "--class-mode",
            str(args.class_mode),
            "--alpha",
            str(args.alpha),
            "--max-samples",
            str(args.max_samples),
        ]
    )
    run_step(
        [
            python,
            str(script_dir / "select_yoloctm_readout_cases.py"),
            "--visualization-dir",
            str(overlay_dir),
            "--output-dir",
            str(selection_dir),
            "--per-class-correct",
            str(args.per_class_correct),
            "--per-class-error",
            str(args.per_class_error),
            "--low-confidence",
            str(args.low_confidence),
        ]
    )
    run_step(
        [
            python,
            str(script_dir / "assemble_yoloctm_readout_figure.py"),
            "--selection-dir",
            str(selection_dir),
            "--visualization-dir",
            str(overlay_dir),
            "--output",
            str(figure_path),
            "--max-panels",
            str(args.max_panels),
            "--columns",
            str(args.columns),
            "--tile-size",
            str(args.tile_size),
        ]
    )

    manifest = {
        "checkpoint": str(args.checkpoint),
        "split": str(args.split),
        "output_root": str(output_root),
        "export_dir": str(export_dir),
        "overlay_dir": str(overlay_dir),
        "selection_dir": str(selection_dir),
        "figure": str(figure_path),
        "figure_manifest": str(figure_path.with_suffix(".json")),
        "note": "Validation-only YoloCTM readout figure pipeline; test split is intentionally rejected.",
    }
    manifest_path = output_root / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
