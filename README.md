# IC_Yolo / WM811K YoloCTM

WM-811K wafer-map defect classification experiments built around Ultralytics
YOLO classification backbones and a custom YoloCTM recurrent token head.

Despite the historical repository name, this is no longer a PCB detection
project. The current code prepares WM-811K wafer maps into an image-folder
classification dataset, trains YOLO baselines, and runs YoloCTM ablations for
long-tailed wafer defect recognition.

## Current Snapshot

- Dataset: WM-811K, prepared as `data/wm811k_cls/{train,val,test}/{class}/*.png`
- Classes: 9 classes when `none` is included
- Fixed split used by recent experiments: `70 / 15 / 15`
- Image size: `224`
- Main baseline: Ultralytics YOLO classification
- Main research model: YOLO classification backbone + spatial CTM-style
  recurrent token head
- Experiment ledger: `AutoResearch/results.tsv`
- Detailed run summaries: `AutoResearch/logs/*.json`

The best frozen single-model milestone recorded in this archive is:

- Run: `autoresearch_yoloctm_slim_dkd_ema_calselect_priorcal_20260527_175645`
- Params: about `10.525M`
- Test accuracy: `0.98242`
- Test macro precision / recall / F1: `0.91093 / 0.92941 / 0.91878`

Later AutoResearch entries after that milestone are validation-only screening
runs unless explicitly marked otherwise. The latest no-distillation validation
best in the checked-in ledger is:

- Run: `autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr010_cwp04_tau04_e10_20260607_132608`
- Validation accuracy: `0.98227`
- Validation macro precision / recall / F1: `0.92595 / 0.90571 / 0.91546`
- Test split: not evaluated for this screening run

## Repository Map

- `configs/wm811k_cls.yaml` - main YAML-driven pipeline config.
- `scripts/prepare_wm811k_classification.py` - converts `LSWMD.pkl` into
  Ultralytics image-folder classification format.
- `scripts/train_wm811k_cls.py` - minimal Ultralytics YOLO classifier trainer.
- `scripts/train_wm811k_yoloctm.py` - standalone PyTorch YoloCTM trainer.
- `scripts/run_wm811k_pipeline.py` - prepare/train/validate/test pipeline.
- `scripts/evaluate_wm811k_cls.py` - metrics export for YOLO classification
  checkpoints.
- `AutoResearch/` - experiment queue, configs, run ledger, and JSON summaries.
- `research-wiki/` - notes for robustness, papers, gaps, and report material.
- `refine-logs/` and `review-stage/` - historical refinement and review notes.
- `agent.md` - operational runbook for this workstation and experiment history.

Large local artifacts are intentionally ignored by Git: raw data, prepared
images, training runs, checkpoints, caches, and Kaggle credentials.

## Environment

Recommended Windows environment for the development machine:

```powershell
D:\anaconda3\envs\pcb_yolo\python.exe -m pip install -r requirements.txt
```

If using another Python environment:

```powershell
pip install -r requirements.txt
```

Core dependencies are Ultralytics, PyTorch through Ultralytics, PyYAML, NumPy,
Pandas, Pillow, scikit-learn, tqdm, OpenCV, and Requests.

## Dataset Preparation

Expected raw source:

```text
data/MIR-WM811K/LSWMD.pkl
```

Prepare the current 9-class dataset, including the dominant `none` class:

```powershell
python scripts/prepare_wm811k_classification.py `
  --source data/MIR-WM811K `
  --output data/wm811k_cls `
  --image-size 224 `
  --include-none `
  --ratios 70 15 15 `
  --overwrite
```

The resulting layout is:

```text
data/wm811k_cls/
  train/<class_name>/*.png
  val/<class_name>/*.png
  test/<class_name>/*.png
  dataset_summary.json
```

The prepared dataset is large and ignored by Git. Do not enumerate it
recursively in routine checks.

## Quick Commands

Check the resolved pipeline plan without training:

```powershell
D:\anaconda3\envs\pcb_yolo\python.exe scripts\run_wm811k_pipeline.py `
  --config configs\wm811k_cls.yaml `
  --check-config
```

Train a YOLO classification baseline:

```powershell
python scripts/train_wm811k_cls.py `
  --data data/wm811k_cls `
  --model yolov8n-cls.pt `
  --epochs 40 `
  --imgsz 224 `
  --batch 64 `
  --device 0
```

Train the YoloCTM model directly:

```powershell
python scripts/train_wm811k_yoloctm.py `
  --data data/wm811k_cls `
  --weights yolo26m-cls.pt `
  --epochs 40 `
  --imgsz 224 `
  --batch 64 `
  --device 0 `
  --name wm811k_yoloctm
```

Run the YAML-driven pipeline:

```powershell
D:\anaconda3\envs\pcb_yolo\python.exe scripts\run_wm811k_pipeline.py `
  --config configs\wm811k_cls.yaml
```

Run a small CPU smoke test:

```powershell
python scripts/run_wm811k_cls_test.py `
  --data data/wm811k_cls `
  --model yolo26m-cls.pt `
  --epochs 1 `
  --device cpu `
  --fraction 0.05
```

## Pipeline Notes

`scripts/run_wm811k_pipeline.py` supports two algorithm modes:

```yaml
model:
  algorithm: yolo      # Ultralytics YOLO classifier
```

```yaml
model:
  algorithm: yoloctm   # YOLO backbone + CTM recurrent token head
```

For `algorithm: yolo`, the pipeline trains with Ultralytics and later loads:

```text
runs/classify/<run_name>/weights/best.pt
```

For `algorithm: yoloctm`, the pipeline calls
`scripts/train_wm811k_yoloctm.py` and expects:

```text
runs/classify/<run_name>/best_yoloctm.pt
```

Metrics are written under:

```text
runs/classify/<run_name>/metrics/
```

YoloCTM metrics are written by the pipeline. YOLO baseline metrics use
`scripts/evaluate_wm811k_cls.py`.

## Experiment Archive

Use these files to understand or resume the research thread:

- `AutoResearch/results.tsv` - compact table of every logged run.
- `AutoResearch/logs/*.json` - one summary file per run.
- `AutoResearch/configs/*.yaml` - exact candidate configs.
- `AutoResearch/experiment_queue.md` - queued and completed ideas.
- `agent.md` - chronological operational notes and best-result snapshots.
- `research-wiki/yoloctm_nodistill_short_paper_zh.md` - Chinese report draft.

Recent validation-only screening intentionally avoids repeated test-set access.
Only promoted milestones should be evaluated on `test`.

## Development Cautions

- Prefer the local development path `E:\Cjn\PCB_Yolo` for long training jobs.
- The mounted `R:\Cjn\PCB_Yolo` path is fine for editing and inspection.
- Use `D:\anaconda3\envs\pcb_yolo\python.exe` when reproducing recorded
  pipeline commands.
- Keep `workers: 0` on this Windows setup unless you are deliberately testing
  data-loader behavior.
- Do not casually regenerate `data/wm811k_cls`; reproducibility depends on the
  fixed seed, ratios, and `include_none` setting.

