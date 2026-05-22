# AutoResearch

An experiment scaffold for WM-811K research, inspired by `karpathy/autoresearch`.

## Goal

- Keep the dataset split and epoch budget fixed.
- Improve `accuracy`, `precision`, `recall`, and `F1`.
- Reduce parameter count whenever possible.
- Favor architecture changes over hyperparameter fishing.

## What stays fixed

- Dataset source: `data/MIR-WM811K/LSWMD.pkl`
- Prepared dataset: `data/wm811k_cls`
- Split ratios: fixed once the baseline is chosen
- Epoch budget: fixed once the baseline is chosen
- Image size: fixed once the baseline is chosen

## What may change

- YOLO backbone choice
- CTM depth, state size, and token handling
- Where and how the backbone output is pooled
- Parameter sharing inside the CTM block
- Lightweight fusion between YOLO features and CTM state

## Logging

- `runs/classify/<run>/...` holds the real training artifacts.
- `AutoResearch/results.tsv` tracks every experiment in a compact table.
- `AutoResearch/logs/*.json` stores one JSON summary per logged run.

## Development Machine Workflow

- Run training and evaluation from a terminal opened directly on the development machine.
- Treat the project as a local workspace there, usually `E:\Cjn\PCB_Yolo`.
- Prefer explicit Python: `D:\anaconda3\envs\pcb_yolo\python.exe`.
- Use the mounted `R:\Cjn\PCB_Yolo` view only for lightweight editing/inspection; do not launch long training through it.
- Do not wrap long experiments in SSH commands from another machine. Open a terminal on the development machine and run local commands.

## Typical loop

1. Run a fixed-budget experiment.
2. Inspect validation and test metrics.
3. Record the run with `AutoResearch/scripts/log_experiment.py`.
4. Keep a change only if it improves metrics or cuts parameters without hurting metrics.

## Recommended entry points

- `AutoResearch/program.md`
- `AutoResearch/configs/wm811k_autoresearch.yaml`
- `AutoResearch/scripts/log_experiment.py`
