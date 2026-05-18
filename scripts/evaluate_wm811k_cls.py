from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute precision/recall/F1 for a WM-811K classification run")
    parser.add_argument("--run-dir", type=Path, required=True, help="Ultralytics run directory containing weights/best.pt")
    parser.add_argument("--data", type=Path, default=Path("data/wm811k_cls"), help="Prepared classification dataset root")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="Dataset split to evaluate")
    parser.add_argument("--model", type=Path, default=None, help="Optional model path; defaults to run-dir/weights/best.pt")
    parser.add_argument("--device", default="0", help="CUDA device id or cpu")
    parser.add_argument("--imgsz", type=int, default=224, help="Inference image size")
    parser.add_argument("--batch", type=int, default=128, help="Inference batch size")
    return parser.parse_args()


def list_samples(split_dir: Path) -> tuple[list[Path], list[int], list[str]]:
    class_names = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    class_to_index = {name: index for index, name in enumerate(class_names)}
    image_paths: list[Path] = []
    labels: list[int] = []

    for class_name in class_names:
        for image_path in sorted((split_dir / class_name).rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                image_paths.append(image_path)
                labels.append(class_to_index[class_name])

    if not image_paths:
        raise FileNotFoundError(f"No images found under {split_dir}")
    return image_paths, labels, class_names


def predict(model: YOLO, image_paths: list[Path], batch: int, imgsz: int, device: str) -> list[int]:
    predictions: list[int] = []
    for start in range(0, len(image_paths), batch):
        batch_paths = [str(path) for path in image_paths[start : start + batch]]
        results = model.predict(batch_paths, imgsz=imgsz, device=device, verbose=False)
        predictions.extend(int(result.probs.top1) for result in results)
    return predictions


def write_report_csv(report: dict, class_names: list[str], output_path: Path) -> None:
    rows = []
    for name in class_names + ["accuracy", "macro avg", "weighted avg"]:
        value = report[name]
        if isinstance(value, dict):
            rows.append(
                {
                    "class": name,
                    "precision": value.get("precision", ""),
                    "recall": value.get("recall", ""),
                    "f1-score": value.get("f1-score", ""),
                    "support": value.get("support", ""),
                }
            )
        else:
            rows.append({"class": name, "precision": "", "recall": "", "f1-score": value, "support": ""})

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "precision", "recall", "f1-score", "support"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    data_root = args.data.resolve()
    split_dir = data_root / args.split
    model_path = args.model.resolve() if args.model else run_dir / "weights" / "best.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model does not exist: {model_path}")

    image_paths, y_true, class_names = list_samples(split_dir)
    model = YOLO(str(model_path))
    y_pred = predict(model, image_paths, args.batch, args.imgsz, args.device)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / f"{args.split}_classification_report.json"
    csv_path = metrics_dir / f"{args.split}_classification_report.csv"
    matrix_path = metrics_dir / f"{args.split}_confusion_matrix.csv"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report_csv(report, class_names, csv_path)
    np.savetxt(matrix_path, matrix, delimiter=",", fmt="%d")

    print(f"Model: {model_path}")
    print(f"Split: {split_dir}")
    print(f"Images: {len(image_paths)}")
    print(f"Report JSON: {json_path}")
    print(f"Report CSV: {csv_path}")
    print(f"Confusion matrix CSV: {matrix_path}")
    print(json.dumps({key: report[key] for key in ("accuracy", "macro avg", "weighted avg")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
