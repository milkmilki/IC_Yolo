from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AUTO_DIR = ROOT / "AutoResearch"
RESULTS_PATH = AUTO_DIR / "results.tsv"
LOG_DIR = AUTO_DIR / "logs"

TSV_COLUMNS = [
    "timestamp",
    "status",
    "algorithm",
    "run_dir",
    "params_m",
    "epochs",
    "imgsz",
    "batch",
    "device",
    "model_summary",
    "val_acc",
    "val_macro_p",
    "val_macro_r",
    "val_macro_f1",
    "test_acc",
    "test_macro_p",
    "test_macro_r",
    "test_macro_f1",
    "description",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a WM-811K AutoResearch log row")
    parser.add_argument("--run-dir", type=Path, required=True, help="Ultralytics or YoloCTM run directory")
    parser.add_argument("--status", default="keep", help="keep, discard, crash, or another short tag")
    parser.add_argument("--description", default="", help="Short experiment note")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def read_report(path: Path) -> dict[str, Any]:
    report = load_json(path)
    if not report:
        return {}
    macro = report.get("macro avg", {}) if isinstance(report.get("macro avg"), dict) else {}
    return {
        "acc": report.get("accuracy", ""),
        "macro_p": macro.get("precision", ""),
        "macro_r": macro.get("recall", ""),
        "macro_f1": macro.get("f1-score", ""),
    }


def summarize_model(config: dict[str, Any], checkpoint_meta: dict[str, Any], algorithm: str, checkpoint_name: str) -> str:
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    ctm_cfg = model_cfg.get("ctm", {}) if isinstance(model_cfg.get("ctm"), dict) else {}
    weights = model_cfg.get("weights") or checkpoint_meta.get("weights") or checkpoint_name
    feature_fusion = ctm_cfg.get("feature_fusion") or checkpoint_meta.get("feature_fusion")
    gate_rank = ctm_cfg.get("gate_rank") or checkpoint_meta.get("gate_rank")
    token_gate_rank = ctm_cfg.get("token_gate_rank") or checkpoint_meta.get("token_gate_rank")
    logit_bias = ctm_cfg.get("logit_bias") if "logit_bias" in ctm_cfg else checkpoint_meta.get("logit_bias")
    logit_bias_init = ctm_cfg.get("logit_bias_init") or checkpoint_meta.get("logit_bias_init")
    logit_bias_tau = ctm_cfg.get("logit_bias_prior_tau") or checkpoint_meta.get("logit_bias_prior_tau")
    anchor_loss_weight = ctm_cfg.get("anchor_loss_weight") or checkpoint_meta.get("anchor_loss_weight")
    distill_weight = ctm_cfg.get("distill_weight") or checkpoint_meta.get("distill_weight")
    distill_temperature = ctm_cfg.get("distill_temperature") or checkpoint_meta.get("distill_temperature")

    if algorithm == "yoloctm":
        summary = (
            f"{weights} | ctm("
            f"d={ctm_cfg.get('d_model', checkpoint_meta.get('d_model', '?'))},"
            f"s={ctm_cfg.get('steps', checkpoint_meta.get('steps', '?'))},"
            f"drop={ctm_cfg.get('dropout', checkpoint_meta.get('dropout', '?'))},"
            f"pow={ctm_cfg.get('class_weight_power', checkpoint_meta.get('class_weight_power', '?'))}"
            f")"
        )
        if feature_fusion:
            summary += f" | fusion={feature_fusion}"
        if feature_fusion == "gated":
            summary += f"(rank={gate_rank})"
        elif feature_fusion == "token":
            summary += f"(rank={token_gate_rank})"
        if logit_bias:
            summary += f" | logit_bias={logit_bias_init or 'learned'}"
            if logit_bias_tau is not None:
                summary += f"(tau={logit_bias_tau})"
        if anchor_loss_weight:
            summary += f" | anchor_kl={anchor_loss_weight}"
        if distill_weight:
            summary += f" | distill(w={distill_weight},T={distill_temperature})"
        return summary
    return str(weights)


def load_checkpoint_meta(run_dir: Path) -> tuple[dict[str, Any], Path | None, str]:
    checkpoint = first_existing(
        [
            run_dir / "best_yoloctm.pt",
            run_dir / "weights" / "best.pt",
            run_dir / "weights" / "last.pt",
        ]
    )
    if checkpoint is None:
        return {}, None, "unknown"

    if checkpoint.name == "best_yoloctm.pt":
        try:
            import torch

            try:
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            except TypeError:
                saved = torch.load(checkpoint, map_location="cpu")
        except Exception:
            return {}, checkpoint, "yoloctm"

        args = saved.get("args", {}) if isinstance(saved, dict) else {}
        meta = {
            "args": args,
            "weights": args.get("weights") or args.get("model") or "",
            "d_model": args.get("d_model"),
            "steps": args.get("steps"),
            "dropout": args.get("dropout"),
            "class_weight_power": args.get("class_weight_power"),
            "feature_fusion": args.get("feature_fusion"),
            "gate_rank": args.get("gate_rank"),
            "token_gate_rank": args.get("token_gate_rank"),
            "logit_bias": args.get("logit_bias"),
            "logit_bias_init": args.get("logit_bias_init"),
            "logit_bias_prior_tau": args.get("logit_bias_prior_tau"),
            "anchor_loss_weight": args.get("anchor_loss_weight"),
            "distill_weight": args.get("distill_weight"),
            "distill_temperature": args.get("distill_temperature"),
            "params": sum(t.numel() for t in saved.get("model_state", {}).values() if hasattr(t, "numel")),
        }
        return meta, checkpoint, "yoloctm"

    try:
        from ultralytics import YOLO

        model = YOLO(str(checkpoint))
        params = sum(parameter.numel() for parameter in model.model.parameters())
        return {"params": params, "weights": str(checkpoint)}, checkpoint, "yolo"
    except Exception:
        return {"weights": str(checkpoint)}, checkpoint, "yolo"


def load_config(run_dir: Path) -> dict[str, Any]:
    config = load_json(run_dir / "resolved_config.json")
    if config:
        return config
    return {}


def metric_value(report: dict[str, Any], key: str) -> str:
    value = report.get(key, "")
    return "" if value is None else str(value)


def format_params_m(params: Any) -> str:
    try:
        return f"{float(params) / 1_000_000:.3f}"
    except Exception:
        return ""


def build_row(run_dir: Path, status: str, description: str) -> tuple[dict[str, str], dict[str, Any]]:
    config = load_config(run_dir)
    checkpoint_meta, checkpoint_path, algorithm = load_checkpoint_meta(run_dir)

    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    train_cfg = config.get("train", {}) if isinstance(config.get("train"), dict) else {}
    algorithm = str(model_cfg.get("algorithm") or algorithm).lower()
    if algorithm not in {"yolo", "yoloctm"}:
        algorithm = "unknown"

    checkpoint_name = checkpoint_path.name if checkpoint_path else ""
    model_summary = summarize_model(config, checkpoint_meta, algorithm, checkpoint_name)
    params_m = format_params_m(checkpoint_meta.get("params"))

    val_report = read_report(run_dir / "metrics" / "val_classification_report.json")
    test_report = read_report(run_dir / "metrics" / "test_classification_report.json")

    train_args = checkpoint_meta.get("args", {}) if isinstance(checkpoint_meta.get("args"), dict) else {}
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "algorithm": algorithm,
        "run_dir": str(run_dir),
        "params_m": params_m,
        "epochs": str(train_cfg.get("epochs") or train_args.get("epochs") or ""),
        "imgsz": str(train_cfg.get("imgsz") or train_args.get("imgsz") or ""),
        "batch": str(train_cfg.get("batch") or train_args.get("batch") or ""),
        "device": str(train_cfg.get("device") or train_args.get("device") or ""),
        "model_summary": model_summary,
        "val_acc": metric_value(val_report, "acc"),
        "val_macro_p": metric_value(val_report, "macro_p"),
        "val_macro_r": metric_value(val_report, "macro_r"),
        "val_macro_f1": metric_value(val_report, "macro_f1"),
        "test_acc": metric_value(test_report, "acc"),
        "test_macro_p": metric_value(test_report, "macro_p"),
        "test_macro_r": metric_value(test_report, "macro_r"),
        "test_macro_f1": metric_value(test_report, "macro_f1"),
        "description": description,
    }
    payload = {
        "row": row,
        "config": config,
        "checkpoint_meta": checkpoint_meta,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        "reports": {"val": val_report, "test": test_report},
    }
    return row, payload


def append_row(row: dict[str, str]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_PATH.exists()
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TSV_COLUMNS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_json_log(run_dir: Path, payload: dict[str, Any]) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{timestamp}_{run_dir.name}.json"
    log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return log_path


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    row, payload = build_row(run_dir, args.status, args.description)
    append_row(row)
    log_path = write_json_log(run_dir, payload)
    print(f"Logged: {RESULTS_PATH}")
    print(f"JSON: {log_path}")
    print(f"Run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
