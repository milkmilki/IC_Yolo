# WM811K YoloCTM Experiment Queue

Last updated: 2026-06-06

This queue is for the no-distillation, <=10 epoch, validation-only AutoResearch track. Do not use test metrics for selection or tuning.

## Current Best

- Run: `autoresearch_yoloctm_nodistill_stepcond_class_attention_readout_tau04_e10_20260606_135126`
- Validation macro F1: `0.9115232889273587`
- Protocol: no distillation, 10 epochs, validation-only screening, `test.enabled: false`
- Main idea: step-conditioned adaptive CTM thought steps on top of residual YOLO feature fusion, with class-specific CTM token attention readout.

## Active / Next Runs

1. Class-attention mean-blend readout:
   - Config: `AutoResearch/configs/wm811k_autoresearch_stepcond_class_attention_blend_readout.yaml`
   - Single factor over current best: add one learnable gate per class to blend class-specific attention evidence with mean CTM pooling before the class logit.
   - Rationale: validation-only class deltas show class attention improves Edge-Loc/Center/Loc but slightly hurts Scratch/Near-full/Donut; a per-class mean fallback may preserve spatial gains while protecting fragile classes.

2. Validation-only evidence package for the current best:
   - Current best checkpoint: `runs/classify/autoresearch_yoloctm_nodistill_stepcond_class_attention_readout_tau04_e10_20260606_135126/best_yoloctm.pt`
   - Generated ablation tables at `runs/diagnostics/ablation_table_nodistill`.
   - Generated class-delta reports at `runs/diagnostics/class_delta_class_attention_vs_stepcond`.
   - Generated balanced readout evidence figures at `runs/diagnostics/class_attention_readout_val_figures_balanced` using `--split val --per-class-samples 24`; the selected cases cover all 9 classes.
   - Rationale: the last three readout-side follow-ups were shared attention discard, polar discard, and entropy discard/equal-to-best. Before adding more losses or priors, inspect what the kept class-specific readout actually attends to and which validation classes drive its gain.

## Do Not Rerun

These 2026-06-05 candidates are already completed and recorded:

- `stepcond_deepsup`
- `stepcond_learnedhalt`
- `stepcond_topology_gate`
- `stepcond_consistency`
- `stepcond_attention_readout`
- `stepcond_class_attention_readout`
- `stepcond_class_attention_polar_readout`
- `stepcond_class_attention_entropy_readout`

## Evidence Package Commands

Run these only on validation artifacts. Do not point them at test reports or test exports.

Generate no-distill ablation tables:

```powershell
D:\anaconda3\envs\pcb_yolo\python.exe scripts\summarize_autoresearch_ablation.py `
  --output-dir runs\diagnostics\ablation_table_nodistill
```

Compare current best against the strongest previous no-distill baseline:

```powershell
D:\anaconda3\envs\pcb_yolo\python.exe scripts\compare_classification_reports.py `
  --baseline runs\classify\autoresearch_yoloctm_nodistill_onecycle_lr00125_finaldiv1000_tau04_e10_20260529_125012\metrics\val_classification_report.json `
  --candidate runs\classify\autoresearch_yoloctm_nodistill_stepcond_adaptive_tau04_e10_20260604_175603\metrics\val_classification_report.json `
  --baseline-name onecycle_finaldiv1000 `
  --candidate-name stepcond_adaptive `
  --output-dir runs\diagnostics\class_delta_stepcond_vs_onecycle
```

Build a validation-only readout figure package for a checkpoint:

```powershell
D:\anaconda3\envs\pcb_yolo\python.exe scripts\run_yoloctm_readout_figure_pipeline.py `
  --checkpoint runs\classify\<run_name>\best_yoloctm.pt `
  --output-root runs\diagnostics\<run_name>_readout_figures `
  --device cpu `
  --split val `
  --per-class-samples 24
```

## Run Gating

- Before GPU training: do not set power limits or lock GPU clocks; use the default GPU policy.
- If GPU compute apps show only desktop/browser/remote-control graphics processes and Python process status cannot be confirmed, do not start training.
- If an existing YoloCTM Python training process is active and making progress, monitor only; do not duplicate a run.
- If a machine reboot happens, append `AutoResearch/reboot_events.tsv` with the boot time and candidate-specific reboot count.
