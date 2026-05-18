from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO classification on WM-811K")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "wm811k_cls",
        help="Classification dataset root with train/val/test folders",
    )
    parser.add_argument("--model", default="yolov8n-cls.pt", help="Pretrained YOLO classification checkpoint")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--device", default="0", help="CUDA device id, cpu, or comma-separated devices")
    parser.add_argument("--project", default="runs/classify", help="Output project directory")
    parser.add_argument("--name", default="wm811k", help="Run name")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
