# IC_Yolo

WM-811K wafer map defect classification with Ultralytics YOLO classification models.

This repository focuses on preparing the WM-811K dataset into an image-folder
classification layout, then training or smoke-testing a YOLO classification model.
It no longer contains a PCB detection workflow.

## Project Files

- `scripts/prepare_wm811k_classification.py`: converts `LSWMD.pkl` wafer maps into PNG images arranged by class and split.
- `scripts/train_wm811k_cls.py`: trains a YOLO classification model on the prepared WM-811K dataset.
- `scripts/train_wm811k_yoloctm.py`: trains the proposed YoloCTM (YOLO backbone + spatial CTM head) classifier on WM-811K.
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


Train the YoloCTM variant:

```powershell
python scripts/train_wm811k_yoloctm.py --data data/wm811k_cls --model yolov8n-cls.pt --epochs 40 --imgsz 224 --batch 64 --device 0 --name wm811k_yoloctm
```


## Run From YAML

The full prepare/train/validate/test workflow can be driven by:

```powershell
conda run --no-capture-output -n pcb_yolo python scripts/run_wm811k_pipeline.py --config configs/wm811k_cls.yaml
```

The YAML config controls the algorithm, model config file, pretrained weights,
dataset split ratios, training arguments, validation split, test split, and log filename. Foreground and
background PowerShell commands are included as comments at the top of
`configs/wm811k_cls.yaml`.

Select the training algorithm with:

```yaml
model:
  algorithm: yolo      # or yoloctm
  config: null         # optional architecture YAML
  weights: yolo26m-cls.pt
  pretrained: true
```

For the spatial CTM head, set `algorithm: yoloctm`; CTM-specific parameters
live under `model.ctm`.

For calibrated YoloCTM experiments, `train.selection_prior_logit_tau` optionally
applies a fixed class-prior logit adjustment only while selecting the best
validation checkpoint. Final metric calibration remains configured under
`metrics.prior_logit_tau`.

During iterative AutoResearch screening, set `test.enabled: false`, evaluate
only `metrics.splits: [val]`, and use a predeclared calibration value. Reserve
the test split for a promoted milestone checkpoint instead of selecting
experiments from repeated test feedback.

`train.ema_decay` optionally exports an exponential-moving-average YoloCTM
checkpoint for validation and inference; set it to `0` to keep the ordinary
online-weight behavior.

For non-distilled long-tail trials, `model.ctm.loss: ldam_drw` enables a
deferred LDAM-inspired target margin controlled by `ldam_max_margin` and
`ldam_start_epoch`; it preserves the existing weighted cross-entropy outside
the configured margin phase.

Independent external benchmarks such as MixedWM38 require a separately declared
label-mapping or mixed-defect evaluation protocol; they should not be silently
substituted for the WM811K held-out split.

Before any external wafer benchmark evaluation, run the metadata-only audit
described in `research-wiki/external_wafer_robustness_protocol.md`. The audit
does not score a model and skips folders named `test` by default.

Set `prepare.enabled: false` in the YAML to reuse the existing prepared dataset
without regenerating the train/val/test split on every run.

Check the resolved plan without running training:

```powershell
python scripts/run_wm811k_pipeline.py --config configs/wm811k_cls.yaml --check-config
```

Pipeline logs are streamed to the console and written in real time under the
current run directory, for example:

```text
runs/classify/wm811k_yolo26m_YYYYMMDD_HHMMSS/pipeline.log
```

When `metrics.enabled: true`, the pipeline also writes precision, recall,
F1-score, support, and confusion matrix CSV files under:

```text
runs/classify/<run_name>/metrics/
```

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
