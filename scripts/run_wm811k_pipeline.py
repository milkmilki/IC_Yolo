from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class Tee:
    def __init__(self, console_stream, log_stream):
        self.console_stream = console_stream
        self.log_stream = log_stream
        self._line_buffer = ""

    def write(self, data: str) -> int:
        self.console_stream.write(data)
        self.console_stream.flush()
        self._write_clean_log(data)
        return len(data)

    def flush(self) -> None:
        self.console_stream.flush()
        self.log_stream.flush()

    def _write_clean_log(self, data: str) -> None:
        clean = ANSI_ESCAPE_RE.sub("", data)
        clean = clean.replace("\x08", "")
        for char in clean:
            if char == "\r":
                self._line_buffer = ""
                continue
            if char == "\n":
                self.log_stream.write(self._line_buffer.rstrip() + "\n")
                self._line_buffer = ""
                continue
            self._line_buffer += char
        self.log_stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the WM-811K classification pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "wm811k_cls.yaml",
        help="Pipeline YAML config",
    )
    parser.add_argument("--skip-prepare", action="store_true", help="Skip dataset preparation")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--skip-val", action="store_true", help="Skip validation on the val split")
    parser.add_argument("--skip-test", action="store_true", help="Skip evaluation on the test split")
    parser.add_argument("--check-config", action="store_true", help="Load config and print the resolved plan")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return config


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise TypeError(f"Config section '{name}' must be a mapping")
    return section


def build_run_name(train_config: dict[str, Any]) -> str:
    name = str(train_config.get("name", "wm811k"))
    if train_config.get("add_timestamp", False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{name}_{timestamp}"
    return name


def build_run_dir(train_config: dict[str, Any], run_name: str) -> Path:
    project = resolve_path(train_config.get("project", "runs/classify"))
    return project / run_name


def setup_logging(run_dir: Path, filename: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / filename
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    stdout = Tee(sys.__stdout__, log_file)
    stderr = Tee(sys.__stderr__, log_file)
    return log_path, log_file, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr)


def prepare_dataset(dataset_config: dict[str, Any]) -> None:
    from prepare_wm811k_classification import main as prepare_main

    source = resolve_path(dataset_config.get("source", "data/MIR-WM811K"))
    output = resolve_path(dataset_config.get("output", "data/wm811k_cls"))
    ratios = dataset_config.get("ratios", [60, 15, 25])

    argv = [
        "prepare_wm811k_classification.py",
        "--source",
        str(source),
        "--output",
        str(output),
        "--image-size",
        str(dataset_config.get("image_size", 224)),
        "--seed",
        str(dataset_config.get("seed", 42)),
        "--ratios",
        *[str(value) for value in ratios],
    ]
    if dataset_config.get("include_none", False):
        argv.append("--include-none")
    if dataset_config.get("overwrite", False):
        argv.append("--overwrite")

    original_argv = sys.argv[:]
    try:
        sys.argv = argv
        prepare_main()
    finally:
        sys.argv = original_argv


def require_dataset(data_root: Path) -> None:
    missing = [split for split in ("train", "val", "test") if not (data_root / split).exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset split folders under {data_root}: {', '.join(missing)}")


def log_torch_runtime(device: str) -> None:
    import torch

    print(f"[runtime] torch={torch.__version__}")
    print(f"[runtime] torch_cuda={torch.version.cuda}")
    print(f"[runtime] cuda_available={torch.cuda.is_available()}")
    print(f"[runtime] requested_device={device}")
    if torch.cuda.is_available():
        print(f"[runtime] cuda_device_count={torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"[runtime] cuda:{index}={torch.cuda.get_device_name(index)}")
    elif device.lower() != "cpu":
        raise RuntimeError(
            f"Config requested device '{device}', but PyTorch cannot access CUDA. "
            "Install a CUDA-enabled PyTorch build or set train.device to 'cpu'."
        )


def resolve_model_path(model_name: str) -> str:
    model_path = Path(model_name)
    if model_path.exists() or model_path.is_absolute():
        return str(model_path)

    root_model_path = PROJECT_ROOT / model_path
    if root_model_path.exists():
        return str(root_model_path)

    return model_name


def train_model(config: dict[str, Any], run_name: str):
    from ultralytics import YOLO

    dataset_config = get_section(config, "dataset")
    model_config = get_section(config, "model")
    train_config = get_section(config, "train")

    data_root = resolve_path(dataset_config.get("output", "data/wm811k_cls"))
    require_dataset(data_root)

    device = str(train_config.get("device", "0"))
    log_torch_runtime(device)
    model_name = resolve_model_path(str(model_config.get("name", "yolo11m-cls.pt")))
    print(f"[runtime] model={model_name}")

    model = YOLO(model_name)
    model.train(
        data=str(data_root),
        epochs=int(train_config.get("epochs", 40)),
        imgsz=int(train_config.get("imgsz", dataset_config.get("image_size", 224))),
        batch=int(train_config.get("batch", 64)),
        device=device,
        workers=int(train_config.get("workers", 8)),
        patience=int(train_config.get("patience", 20)),
        project=str(resolve_path(train_config.get("project", "runs/classify"))),
        name=run_name,
        exist_ok=bool(train_config.get("exist_ok", True)),
        pretrained=bool(model_config.get("pretrained", False)),
    )
    return model


def find_best_model(run_dir: Path, fallback_model: str):
    from ultralytics import YOLO

    best = run_dir / "weights" / "best.pt"
    if best.exists():
        return YOLO(str(best))
    return YOLO(resolve_model_path(fallback_model))


def evaluate(model, data_root: Path, split: str, phase: str) -> None:
    print(f"\n[{phase}] Evaluating split='{split}'")
    model.val(data=str(data_root), split=split)


def copy_config(config_path: Path, run_dir: Path, config: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml")
    resolved = json.dumps(config, indent=2, ensure_ascii=False)
    (run_dir / "resolved_config.json").write_text(resolved, encoding="utf-8")


def print_plan(config_path: Path, config: dict[str, Any], run_name: str, run_dir: Path) -> None:
    dataset = get_section(config, "dataset")
    model = get_section(config, "model")
    train = get_section(config, "train")
    plan = {
        "config": str(config_path.resolve()),
        "source": str(resolve_path(dataset.get("source", "data/MIR-WM811K"))),
        "dataset": str(resolve_path(dataset.get("output", "data/wm811k_cls"))),
        "ratios": dataset.get("ratios", [60, 15, 25]),
        "model": model.get("name", "yolo11m-cls.pt"),
        "epochs": train.get("epochs", 40),
        "run_name": run_name,
        "run_dir": str(run_dir),
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    train_config = get_section(config, "train")
    logging_config = get_section(config, "logging")
    dataset_config = get_section(config, "dataset")
    validate_config = get_section(config, "validate")
    test_config = get_section(config, "test")

    run_name = build_run_name(train_config)
    run_dir = build_run_dir(train_config, run_name)

    if args.check_config:
        print_plan(config_path, config, run_name, run_dir)
        return 0

    log_path, log_file, out_redirect, err_redirect = setup_logging(
        run_dir, str(logging_config.get("filename", "pipeline.log"))
    )
    with log_file, out_redirect, err_redirect:
        print(f"[pipeline] Config: {config_path}")
        print(f"[pipeline] Run directory: {run_dir}")
        print(f"[pipeline] Log file: {log_path}")
        copy_config(config_path, run_dir, config)

        if args.skip_prepare:
            print("[prepare] Skipped")
        else:
            print("\n[prepare] Preparing WM-811K classification dataset")
            prepare_dataset(dataset_config)

        model = None
        if args.skip_train:
            print("[train] Skipped")
            model = find_best_model(run_dir, str(get_section(config, "model").get("name", "yolo11m-cls.pt")))
        else:
            print("\n[train] Training YOLO classification model")
            model = train_model(config, run_name)
            model = find_best_model(run_dir, str(get_section(config, "model").get("name", "yolo11m-cls.pt")))

        data_root = resolve_path(dataset_config.get("output", "data/wm811k_cls"))
        if validate_config.get("enabled", True) and not args.skip_val:
            evaluate(model, data_root, str(validate_config.get("split", "val")), "validate")
        else:
            print("[validate] Skipped")

        if test_config.get("enabled", True) and not args.skip_test:
            evaluate(model, data_root, str(test_config.get("split", "test")), "test")
        else:
            print("[test] Skipped")

        print("\n[pipeline] Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
