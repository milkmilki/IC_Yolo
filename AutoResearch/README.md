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

## Remote Workspace

- This repository is mounted from a network drive, so the active execution environment may not be local to the files you see here.
- If you need to inspect or run the workspace on the development machine directly, connect by SSH:
  - Host: `10.129.136.178`
  - Username: `du`
  - Command: `ssh du@10.129.136.178`
- Treat paths, runtimes, and long training jobs as remote-first; do not assume a local-only workflow.

## Typical loop

1. Run a fixed-budget experiment.
2. Inspect validation and test metrics.
3. Record the run with `AutoResearch/scripts/log_experiment.py`.
4. Keep a change only if it improves metrics or cuts parameters without hurting metrics.

## Recommended entry points

- `AutoResearch/program.md`
- `AutoResearch/configs/wm811k_autoresearch.yaml`
- `AutoResearch/scripts/log_experiment.py`
