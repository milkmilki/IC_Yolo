from __future__ import annotations

import argparse
import warnings
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a quick YOLO classification test on WM-811K")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "wm811k_cls",
        help="Prepared classification dataset root with train/val/test folders",
    )
    parser.add_argument(
        "--model",
        default="yolo26m-cls.pt",
        help="YOLO classification checkpoint or model name",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Test training epochs")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="cpu", help="CUDA device id, cpu, or comma-separated devices")
    parser.add_argument("--project", default="runs/classify", help="Output project directory")
    parser.add_argument("--name", default="wm811k_yolo26m_test", help="Run name")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights if available")
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.05,
        help="Train/val fraction to use for a quick smoke test",
    )
    return parser.parse_args()


def validate_dataset(data_root: Path) -> None:
    required_splits = ("train", "val", "test")
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    missing = [split for split in required_splits if not (data_root / split).exists()]
    if missing:
        raise FileNotFoundError(
            f"Dataset is not prepared yet. Missing folders under {data_root}: {', '.join(missing)}"
        )


def resolve_model_name(model_arg: str) -> str:
    model_path = Path(model_arg)
    if model_path.exists():
        return str(model_path)

    candidates = [
        model_arg,
        model_arg.lower(),
        "yolo26m-cls.pt",
        "yolo11m-cls.pt",
        "yolov8m-cls.pt",
        "yolov8n-cls.pt",
    ]

    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return str(candidate_path)

    warnings.warn(
        f"Could not find requested model '{model_arg}'. Falling back to 'yolo26m-cls.pt'.",
        RuntimeWarning,
    )
    return "yolo26m-cls.pt"


def main() -> int:
    args = parse_args()
    data_root = args.data.resolve()
    validate_dataset(data_root)

    from ultralytics import YOLO

    model_name = resolve_model_name(args.model)
    print(f"Using classification model: {model_name}")
    model = YOLO(model_name)
    model.train(
        data=str(data_root),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=args.pretrained,
        fraction=args.fraction,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
