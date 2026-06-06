# WM811K YoloCTM Experiment Queue

Last updated: 2026-06-06

This queue is for the no-distillation, <=10 epoch, validation-only AutoResearch track. Do not use test metrics for selection or tuning.

## Current Best

- Run: `autoresearch_yoloctm_nodistill_stepcond_class_attention_readout_tau04_e10_20260606_135126`
- Validation macro F1: `0.9115232889273587`
- Protocol: no distillation, 10 epochs, validation-only screening, `test.enabled: false`
- Main idea: step-conditioned adaptive CTM thought steps on top of residual YOLO feature fusion, with class-specific CTM token attention readout.

## Active / Next Runs

1. Class-specific attention entropy:
   - Config: `AutoResearch/configs/wm811k_autoresearch_stepcond_class_attention_entropy_readout.yaml`
   - Active run: `autoresearch_yoloctm_nodistill_stepcond_class_attention_entropy_readout_tau04_e10_20260606_160948`
   - Single factor over class attention: add a small training-only entropy minimization loss on class-specific CTM readout weights.
   - Rationale: polar positional bias hurt validation macro F1, but class-specific readout remains the best no-distillation architecture. The next readout-side test should encourage sharper token evidence without adding step-logit losses, distillation, coordinate priors, or inference-time parameters.

## Do Not Rerun

These 2026-06-05 candidates are already completed and recorded:

- `stepcond_deepsup`
- `stepcond_learnedhalt`
- `stepcond_topology_gate`
- `stepcond_consistency`
- `stepcond_attention_readout`
- `stepcond_class_attention_readout`
- `stepcond_class_attention_polar_readout`

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
  --split val
```

## Run Gating

- Before GPU training: do not set power limits or lock GPU clocks; use the default GPU policy.
- If GPU compute apps show only desktop/browser/remote-control graphics processes and Python process status cannot be confirmed, do not start training.
- If an existing YoloCTM Python training process is active and making progress, monitor only; do not duplicate a run.
- If a machine reboot happens, append `AutoResearch/reboot_events.tsv` with the boot time and candidate-specific reboot count.
