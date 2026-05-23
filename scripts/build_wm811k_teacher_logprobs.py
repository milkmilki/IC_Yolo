from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from torchvision import datasets

from evaluate_wm811k_ensemble import (
    build_transform,
    class_log_prior,
    ctm_log_probs,
    parse_weights,
    yolo_log_probs,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached WM811K teacher log-probabilities for distillation")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "wm811k_cls")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--yolo-run", type=Path, required=True)
    parser.add_argument("--ctm-checkpoint", type=Path, required=True)
    parser.add_argument("--lowrank-checkpoint", type=Path, required=True)
    parser.add_argument("--weights", default="0.6,0.2,0.2", help="Comma-separated weights: yolo,ctm,lowrank")
    parser.add_argument("--prior-tau", type=float, default=0.025)
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N batches per teacher")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.data = args.data.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    weights = parse_weights(args.weights)

    dataset = datasets.ImageFolder(args.data / args.split, transform=build_transform(args.imgsz))
    classes = list(dataset.classes)
    image_paths = [str(path) for path, _label in dataset.samples]
    log_prior = class_log_prior(args.data, classes, args.imgsz)

    print(
        f"[teacher] split={args.split} images={len(dataset)} classes={len(classes)} "
        f"batch={args.batch} imgsz={args.imgsz} output={output}",
        flush=True,
    )
    yolo_path = args.yolo_run / "weights" / "best.pt"
    print(f"[teacher] computing YOLO teacher: {yolo_path}", flush=True)
    yolo_scores = yolo_log_probs(
        yolo_path,
        image_paths,
        classes,
        args.batch,
        args.imgsz,
        args.device,
        progress_label="yolo26m",
        progress_every=args.progress_every,
    )
    print(f"[teacher] computing CTM teacher: {args.ctm_checkpoint}", flush=True)
    ctm_scores = ctm_log_probs(
        args.ctm_checkpoint,
        args.data,
        args.split,
        classes,
        args.batch,
        args.imgsz,
        args.device,
        progress_label="ctm_adapter",
        progress_every=args.progress_every,
    )
    print(f"[teacher] computing low-rank CTM teacher: {args.lowrank_checkpoint}", flush=True)
    lowrank_scores = ctm_log_probs(
        args.lowrank_checkpoint,
        args.data,
        args.split,
        classes,
        args.batch,
        args.imgsz,
        args.device,
        progress_label="ctm_lowrank",
        progress_every=args.progress_every,
    )

    print("[teacher] blending log-probs and writing compressed cache", flush=True)
    teacher_scores = weights[0] * yolo_scores + weights[1] * ctm_scores + weights[2] * lowrank_scores
    teacher_scores = teacher_scores + float(args.prior_tau) * log_prior.reshape(1, -1)
    teacher_log_probs = teacher_scores - np.logaddexp.reduce(teacher_scores, axis=1, keepdims=True)

    np.savez_compressed(
        output,
        log_probs=teacher_log_probs.astype(np.float32),
        classes=np.asarray(classes, dtype=object),
        paths=np.asarray(image_paths, dtype=object),
        labels=np.asarray(dataset.targets, dtype=np.int64),
        weights=np.asarray(weights, dtype=np.float32),
        prior_tau=np.asarray([float(args.prior_tau)], dtype=np.float32),
    )
    print(f"Saved teacher log-probs: {output}")
    print(f"Split: {args.split} images={len(dataset)} classes={len(classes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
