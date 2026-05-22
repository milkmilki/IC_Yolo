# Agent Notes for PCB_Yolo

For the latest WM811K AutoResearch runbook, including local development-host launch commands, health checks, and current best YoloCTM result notes, also read `agent.md`.

## Project Summary

- This repository is a WM-811K wafer-map defect classification experiment, not a PCB detection project.
- The main experiment compares/uses Ultralytics YOLO classification and a custom YoloCTM variant.
- YoloCTM means a YOLO classification backbone with the original classification head removed, followed by a spatial CTM-style recurrent token head implemented in `scripts/train_wm811k_yoloctm.py`.
- The prepared dataset follows the Ultralytics image-folder classification layout: `data/wm811k_cls/{train,val,test}/{class_name}/*.png`.

## Important Files

- `README.md`: user-facing setup, dataset preparation, training, pipeline, and smoke-test commands.
- `configs/wm811k_cls.yaml`: pipeline configuration for dataset preparation, model selection, training, validation, test, metrics, and logging.
- `scripts/prepare_wm811k_classification.py`: converts `LSWMD.pkl` from WM-811K into class folders of PNG wafer maps.
- `scripts/train_wm811k_cls.py`: minimal Ultralytics YOLO classification baseline trainer.
- `scripts/train_wm811k_yoloctm.py`: standalone PyTorch trainer for YOLO backbone + CTM head.
- `scripts/run_wm811k_pipeline.py`: YAML-driven prepare/train/validate/test/metrics pipeline.
- `scripts/evaluate_wm811k_cls.py`: precision, recall, F1, and confusion-matrix export for Ultralytics YOLO classification runs.
- `scripts/run_wm811k_cls_test.py`: small CPU-friendly smoke test using a dataset fraction.

## Development Environment

- Long training and evaluation should be run from a terminal opened directly on the development machine, not by wrapping commands through SSH from the mounted network drive.
- On the development machine, treat the project as a local workspace, usually `E:\Cjn\PCB_Yolo`.
- Prefer the project Python explicitly: `D:\anaconda3\envs\pcb_yolo\python.exe`.
- Use local Windows commands such as `cd /d E:\Cjn\PCB_Yolo`, `nvidia-smi`, `wmic`, and direct Python invocations.
- The mounted `R:\Cjn\PCB_Yolo` view may be useful for editing/inspection, but do not launch long training jobs through that mount.

## Dataset State

- Expected raw WM-811K pickle: `data/MIR-WM811K/LSWMD.pkl`.
- Prepared data is already present under `data/wm811k_cls`.
- Current `data/wm811k_cls/dataset_summary.json` indicates `include_none: true`, normalized split ratios `0.7 / 0.15 / 0.15`, 9 classes, and `172950` written images.
- Class imbalance is extreme because `none` has `147431` samples total, while `Near-full` has only `149`; be careful interpreting accuracy.
- Do not recursively print or enumerate all files under `data/wm811k_cls`; it is large and noisy.

## Common Commands

- Install dependencies: `pip install -r requirements.txt`.
- Prepare dataset: `python scripts/prepare_wm811k_classification.py --source data/MIR-WM811K --output data/wm811k_cls --image-size 224 --include-none --ratios 70 15 15 --overwrite`.
- Train YOLO baseline: `python scripts/train_wm811k_cls.py --data data/wm811k_cls --model yolov8n-cls.pt --epochs 40 --imgsz 224 --batch 64 --device 0`.
- Train YoloCTM directly: `python scripts/train_wm811k_yoloctm.py --data data/wm811k_cls --weights yolo26m-cls.pt --epochs 40 --imgsz 224 --batch 64 --device 0 --name wm811k_yoloctm`.
- Run configured pipeline on the development machine: `D:\anaconda3\envs\pcb_yolo\python.exe scripts\run_wm811k_pipeline.py --config configs\wm811k_cls.yaml`.
- Check resolved pipeline plan without training: `D:\anaconda3\envs\pcb_yolo\python.exe scripts\run_wm811k_pipeline.py --config configs\wm811k_cls.yaml --check-config`.
- CPU smoke test: `python scripts/run_wm811k_cls_test.py --data data/wm811k_cls --model yolo26m-cls.pt --epochs 1 --device cpu --fraction 0.05`.

## Implementation Notes

- `run_wm811k_pipeline.py` supports `model.algorithm: yolo` and `model.algorithm: yoloctm`.
- For `algorithm: yolo`, the pipeline uses Ultralytics `YOLO(...).train(...)` and later loads `runs/classify/<run>/weights/best.pt`.
- For `algorithm: yoloctm`, the pipeline calls `train_wm811k_yoloctm.py` and expects `runs/classify/<run>/best_yoloctm.pt`.
- YoloCTM builds the backbone from Ultralytics YOLO classification models by taking `layers[:-1]`; the CTM head receives spatial feature tokens when the backbone output has more than 2 dimensions.
- YoloCTM uses nearest-neighbor resizing and simple flips/rotation augmentations to preserve wafer-map categorical geometry.
- YoloCTM applies class weights using inverse-frequency weights raised to `class_weight_power`; the default is `0.5`.
- Metrics for YoloCTM are written by the pipeline itself; metrics for YOLO baseline use `scripts/evaluate_wm811k_cls.py`.

## Known Pitfalls

- The current base Python environment may not have `pyyaml`; prefer the `pcb_yolo` conda environment shown in the README/pipeline comments.
- `configs/wm811k_cls.yaml` currently has mojibake comments, and some intended keys appear on the same commented line. Before trusting it for a run, use `--check-config` in the correct environment and verify that keys such as `ratios`, `include_none`, `pretrained`, `ctm.steps`, `train.batch`, `train.device`, `metrics.splits`, and `train.name` are actually present in the resolved plan/config.
- Git may report dubious ownership when using the mounted network path; prefer running git from the development-machine local path.
- Large artifacts are intentionally ignored: datasets, runs, caches, Kaggle credentials, and YOLO checkpoint weights.
- Avoid changing data split defaults casually; reproducibility depends on `seed`, `ratios`, and `include_none`.

## Style Guidance

- Keep changes surgical and experiment-focused.
- Prefer adding small validation commands over launching long training jobs unless the user asks.
- When modifying scripts, preserve Windows/PowerShell-friendly commands because the project is used from Windows paths.
- When touching training logic, update `README.md` and this file if workflow assumptions change.
