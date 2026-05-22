from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_wm811k_cls import write_report_csv  # noqa: E402
from run_wm811k_pipeline import load_yoloctm_checkpoint  # noqa: E402
from train_wm811k_yoloctm import WaferToTensor  # noqa: E402


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fixed WM811K log-probability ensemble")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "wm811k_cls")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory used for ensemble outputs and logs")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--yolo-run", type=Path, required=True, help="YOLO classification run directory")
    parser.add_argument("--ctm-checkpoint", type=Path, required=True)
    parser.add_argument("--lowrank-checkpoint", type=Path, required=True)
    parser.add_argument("--weights", default="0.6,0.2,0.2", help="Comma-separated weights: yolo,ctm,lowrank")
    parser.add_argument("--prior-tau", type=float, default=0.025)
    parser.add_argument("--status", default="keep")
    parser.add_argument("--description", default="Fixed log-probability ensemble of YOLO26m, CTM adapter, and low-rank CTM adapter")
    parser.add_argument("--log-autoresearch", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_weights(value: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("--weights must contain exactly three comma-separated values")
    total = sum(parts)
    if total <= 0:
        raise ValueError("Ensemble weights must sum to a positive value")
    return tuple(part / total for part in parts)  # type: ignore[return-value]


def build_transform(imgsz: int):
    return transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz), interpolation=InterpolationMode.NEAREST),
            WaferToTensor(),
        ]
    )


def class_log_prior(data_root: Path, classes: list[str], imgsz: int) -> np.ndarray:
    train_ds = datasets.ImageFolder(data_root / "train", transform=build_transform(imgsz))
    if list(train_ds.classes) != list(classes):
        raise ValueError(f"Train classes do not match: {train_ds.classes} != {classes}")
    counts = np.bincount(np.asarray(train_ds.targets, dtype=np.int64), minlength=len(classes)).astype(np.float64)
    return np.log(np.maximum(counts, 1.0))


def yolo_log_probs(
    model_path: Path,
    image_paths: list[str],
    classes: list[str],
    batch: int,
    imgsz: int,
    device: str,
) -> np.ndarray:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = model.names
    model_classes = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    reorder = [model_classes.index(class_name) for class_name in classes]

    chunks: list[np.ndarray] = []
    for start in range(0, len(image_paths), batch):
        batch_paths = [str(path) for path in image_paths[start : start + batch]]
        results = model.predict(batch_paths, imgsz=imgsz, device=device, verbose=False)
        probs = np.stack([result.probs.data.detach().cpu().numpy()[reorder] for result in results], axis=0)
        chunks.append(np.log(np.clip(probs, EPS, 1.0)))
    return np.concatenate(chunks, axis=0)


def ctm_log_probs(checkpoint: Path, data_root: Path, split: str, classes: list[str], batch: int, imgsz: int, device_arg: str) -> np.ndarray:
    dataset = datasets.ImageFolder(data_root / split, transform=build_transform(imgsz))
    if list(dataset.classes) != list(classes):
        raise ValueError(f"Dataset classes do not match: {dataset.classes} != {classes}")
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0, pin_memory=False)
    model, checkpoint_classes, device, _criterion = load_yoloctm_checkpoint(checkpoint, device_arg)
    if list(checkpoint_classes) != list(classes):
        raise ValueError(f"Checkpoint classes do not match: {checkpoint_classes} != {classes}")

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for images, _labels in loader:
            logits = model(images.to(device))
            chunks.append(torch.log_softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def evaluate_split(
    split: str,
    args: argparse.Namespace,
    classes: list[str],
    weights: tuple[float, float, float],
    log_prior: np.ndarray,
) -> dict[str, Any]:
    dataset = datasets.ImageFolder(args.data / split, transform=build_transform(args.imgsz))
    if list(dataset.classes) != list(classes):
        raise ValueError(f"{split} classes do not match: {dataset.classes} != {classes}")
    y_true = np.asarray(dataset.targets, dtype=np.int64)

    yolo_path = args.yolo_run / "weights" / "best.pt"
    image_paths = [str(path) for path, _label in dataset.samples]
    yolo_scores = yolo_log_probs(yolo_path, image_paths, classes, args.batch, args.imgsz, args.device)
    ctm_scores = ctm_log_probs(args.ctm_checkpoint, args.data, split, classes, args.batch, args.imgsz, args.device)
    lowrank_scores = ctm_log_probs(args.lowrank_checkpoint, args.data, split, classes, args.batch, args.imgsz, args.device)

    scores = weights[0] * yolo_scores + weights[1] * ctm_scores + weights[2] * lowrank_scores
    scores = scores + float(args.prior_tau) * log_prior.reshape(1, -1)
    y_pred = scores.argmax(axis=1)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    return {"report": report, "matrix": matrix, "num_images": int(len(y_true))}


def save_report(run_dir: Path, split: str, classes: list[str], report: dict[str, Any], matrix: np.ndarray) -> None:
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / f"{split}_classification_report.json"
    csv_path = metrics_dir / f"{split}_classification_report.csv"
    matrix_path = metrics_dir / f"{split}_confusion_matrix.csv"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report_csv(report, classes, csv_path)
    np.savetxt(matrix_path, matrix, delimiter=",", fmt="%d")
    print(f"[{split}] Report JSON: {json_path}")
    print(f"[{split}] Report CSV: {csv_path}")
    print(f"[{split}] Confusion matrix CSV: {matrix_path}")


def write_config(run_dir: Path, args: argparse.Namespace, weights: tuple[float, float, float]) -> None:
    yolo_path = (args.yolo_run / "weights" / "best.pt").resolve()
    config = {
        "dataset": {
            "output": str(args.data),
            "image_size": args.imgsz,
            "ratios": [70, 15, 15],
            "include_none": True,
            "seed": 42,
        },
        "prepare": {"enabled": False},
        "model": {
            "algorithm": "yoloctm",
            "weights": "ensemble:yolo26m+ctm_adapter+ctm_lowrank",
            "ctm": {"d_model": 96, "steps": 4, "dropout": 0.1, "class_weight_power": 0.5},
            "ensemble": {
                "type": "fixed_log_probability_average",
                "weights": {"yolo26m": weights[0], "ctm_adapter": weights[1], "ctm_lowrank": weights[2]},
                "prior_logit_tau": args.prior_tau,
                "sources": {
                    "yolo26m": str(yolo_path),
                    "ctm_adapter": str(args.ctm_checkpoint.resolve()),
                    "ctm_lowrank": str(args.lowrank_checkpoint.resolve()),
                },
            },
        },
        "train": {"epochs": 10, "imgsz": args.imgsz, "batch": 64, "device": args.device, "workers": 0},
        "metrics": {"enabled": True, "splits": ["val", "test"], "batch": args.batch, "prior_logit_tau": args.prior_tau},
        "logging": {"autoresearch": True, "description": args.description},
    }
    (run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "config.yaml").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def write_ensemble_metadata(run_dir: Path, args: argparse.Namespace, weights: tuple[float, float, float], classes: list[str]) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "classes": classes,
        "weights": {"yolo26m": weights[0], "ctm_adapter": weights[1], "ctm_lowrank": weights[2]},
        "prior_logit_tau": args.prior_tau,
        "sources": {
            "yolo26m": str((args.yolo_run / "weights" / "best.pt").resolve()),
            "ctm_adapter": str(args.ctm_checkpoint.resolve()),
            "ctm_lowrank": str(args.lowrank_checkpoint.resolve()),
        },
        "hypothesis": "A fixed log-probability ensemble combines YOLO precision with complementary CTM adapter recall while keeping the original fixed data protocol.",
    }
    (run_dir / "ensemble_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def log_autoresearch(run_dir: Path, status: str, description: str, params_m: str) -> None:
    script_path = PROJECT_ROOT / "AutoResearch" / "scripts" / "log_experiment.py"
    spec = importlib.util.spec_from_file_location("autoresearch_log_experiment", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load AutoResearch logger: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row, payload = module.build_row(run_dir.resolve(), status, description)
    row["algorithm"] = "ensemble"
    row["params_m"] = params_m
    row["model_summary"] = "fixed logprob ensemble: 0.6*yolo26m + 0.2*ctm_adapter + 0.2*ctm_lowrank + tau=0.025"
    module.append_row(row)
    payload = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    payload["row"] = row
    payload["ensemble_metadata"] = json.loads((run_dir / "ensemble_metadata.json").read_text(encoding="utf-8"))
    log_path = module.write_json_log(run_dir.resolve(), payload)
    print(f"[autoresearch] Logged row to {module.RESULTS_PATH}")
    print(f"[autoresearch] JSON summary: {log_path}")


def main() -> int:
    args = parse_args()
    args.data = args.data.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    weights = parse_weights(args.weights)

    classes = datasets.ImageFolder(args.data / "val", transform=build_transform(args.imgsz)).classes
    log_prior = class_log_prior(args.data, classes, args.imgsz)
    write_config(run_dir, args, weights)
    write_ensemble_metadata(run_dir, args, weights, classes)

    for split in ("val", "test"):
        result = evaluate_split(split, args, classes, weights, log_prior)
        save_report(run_dir, split, classes, result["report"], result["matrix"])
        macro = result["report"]["macro avg"]
        print(
            f"[{split}] images={result['num_images']} acc={result['report']['accuracy']:.6f} "
            f"macro_p={macro['precision']:.6f} macro_r={macro['recall']:.6f} macro_f1={macro['f1-score']:.6f}"
        )

    # The ensemble uses all three source models at inference time.
    params_m = f"{11.634 + 10.525 + 10.496:.3f}"
    if args.log_autoresearch:
        log_autoresearch(run_dir, args.status, args.description, params_m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
