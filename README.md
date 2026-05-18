# IC_Yolo

WM-811K wafer map defect classification with Ultralytics YOLO classification models.

This repository focuses on preparing the WM-811K dataset into an image-folder
classification layout, then training or smoke-testing a YOLO classification model.
It no longer contains a PCB detection workflow.

## Project Files

- `scripts/prepare_wm811k_classification.py`: converts `LSWMD.pkl` wafer maps into PNG images arranged by class and split.
- `scripts/train_wm811k_cls.py`: trains a YOLO classification model on the prepared WM-811K dataset.
- `scripts/run_wm811k_cls_test.py`: runs a small CPU-friendly smoke test using a fraction of the dataset.
- `requirements.txt`: Python dependencies.

## Dataset

The expected source file is the WM-811K pickle file:

```text
data/MIR-WM811K/LSWMD.pkl
```

The preparation script also searches recursively under the source directory for
`LSWMD.pkl`.

By default, the script uses the common 8-class setup and excludes the `none`
class. Add `--include-none` if you want to keep it.

Classes:

- `Center`
- `Donut`
- `Edge-Loc`
- `Edge-Ring`
- `Loc`
- `Random`
- `Scratch`
- `Near-full`
- `none` optional

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Prepare The Dataset

Convert WM-811K wafer maps into YOLO classification folders:

```powershell
python scripts/prepare_wm811k_classification.py --source data/MIR-WM811K --output data/wm811k_cls --image-size 224 --overwrite
```

Default split ratios are `60 / 15 / 25` for `train / val / test`. You can change
them with:

```powershell
python scripts/prepare_wm811k_classification.py --source data/MIR-WM811K --output data/wm811k_cls --ratios 7 2 1 --image-size 224 --overwrite
```

Prepared layout:

```text
data/wm811k_cls/
  train/
    Center/
    Donut/
    ...
  val/
    Center/
    Donut/
    ...
  test/
    Center/
    Donut/
    ...
  dataset_summary.json
```

## Train

Train a YOLO classification model:

```powershell
python scripts/train_wm811k_cls.py --data data/wm811k_cls --model yolov8n-cls.pt --epochs 40 --imgsz 224 --batch 64 --device 0
```

Use `--device cpu` if CUDA is not available.

## Smoke Test

Run a quick test on a small fraction of the dataset:

```powershell
python scripts/run_wm811k_cls_test.py --data data/wm811k_cls --model yolo11m-cls.pt --epochs 1 --device cpu --fraction 0.05
```

The script validates that `train`, `val`, and `test` folders exist before
starting training.

## Outputs

Training outputs are written under:

```text
runs/classify/
```

Dataset files, model checkpoints, training runs, Kaggle credentials, and cache
files are intentionally ignored by Git.
