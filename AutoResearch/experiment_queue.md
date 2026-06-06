# WM811K YoloCTM Experiment Queue

Last updated: 2026-06-06

This queue is for the no-distillation, <=10 epoch, validation-only AutoResearch track. Do not use test metrics for selection or tuning.

## Current Best

- Run: `autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_tau04_e10_20260607_030102`
- Validation macro F1: `0.912020609416007`
- Protocol: no distillation, 10 epochs, validation-only screening, `test.enabled: false`
- Main idea: step-conditioned adaptive CTM thought steps on top of residual YOLO feature fusion, class-specific CTM token attention readout, and mild deferred LDAM-DRW class-boundary margins.

## Active / Next Runs

1. Validation-only evidence package for the current best:
   - Current best checkpoint: `runs/classify/autoresearch_yoloctm_nodistill_stepcond_class_attention_readout_tau04_e10_20260606_135126/best_yoloctm.pt`
   - Generated ablation tables at `runs/diagnostics/ablation_table_nodistill`.
   - Generated class-delta reports at `runs/diagnostics/class_delta_class_attention_vs_stepcond`.
   - Generated balanced readout evidence figures at `runs/diagnostics/class_attention_readout_val_figures_balanced` using `--split val --per-class-samples 24`; the selected cases cover all 9 classes.
   - Rationale: the last three readout-side follow-ups were shared attention discard, polar discard, and entropy discard/equal-to-best. Before adding more losses or priors, inspect what the kept class-specific readout actually attends to and which validation classes drive its gain.

2. Next active run:
   - Active run: `autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr005_tau04_e10_20260607_050437`.
   - Config: `AutoResearch/configs/wm811k_autoresearch_stepcond_class_attention_ldam_m01_cbr005.yaml`.
   - Recovery wrapper: `scripts/run_stepcond_class_attention_ldam_m01_cbr005.cmd`.
   - Single factor over the new best: keep `ldam_max_margin=0.1` and add a very small deferred classifier-boundary regularizer (`classifier_cbr_weight=0.005`, `classifier_cbr_start_epoch=7`).
   - Rationale: margin 0.05 failed below the margin 0.1 best, so LDAM margin tuning stops. The next test is a qualitatively different class-boundary pressure that regularizes classifier geometry rather than changing per-class margins.
   - Launched on 2026-06-07 05:04 +08:00 via `WM811K_AutoResearch_ClassAttentionLDAMM01CBR005`; initial health check showed epoch 1 progressing, `[cbr] classifier regularization weight=0.0050 ... starts at epoch 7`, and GPU active.
   - Added metadata-only external audit script: `scripts/audit_external_wafer_dataset.py`.
   - Added protocol doc: `research-wiki/external_wafer_robustness_protocol.md`.
   - No external wafer benchmark is currently present under `data/`; do not evaluate external performance until dataset placement, metadata audit, and label mapping are committed.
   - The class-attention mean-blend run is completed and discarded; it helped Scratch but damaged Near-full, Edge-Loc, and Center, so do not continue the mean-fallback/gating direction without a stronger class-conditional rationale.
   - The class-attention halt95 run is completed and discarded/equal-to-best; it increased average thought steps but did not exceed the current validation threshold.
   - The class-attention min5 run is completed and discarded/equal-to-best; it forced average validation thought steps to about `5.076` but did not exceed the current validation threshold.
   - First LDAM launch attempt `autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_tau04_e10_20260607_020207` failed before epoch 1 because the config used invalid loss name `ldam`; corrected to the training script's `ldam_drw`.
   - Relaunched corrected LDAM-DRW on 2026-06-07 02:04 +08:00 as `autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_tau04_e10_20260607_020428`; result: discard, val macro F1 `0.9108497579599639`.
   - LDAM-DRW margin 0.1 completed on 2026-06-07 as `autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_tau04_e10_20260607_030102`; result: keep, val macro F1 `0.912020609416007`.
   - LDAM-DRW margin 0.05 completed on 2026-06-07 as `autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m005_tau04_e10_20260607_040342`; result: discard, val macro F1 `0.9103854226442668`.

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
- `stepcond_class_attention_blend_readout`
- `stepcond_class_attention_halt95`
- `stepcond_class_attention_min5`
- `stepcond_class_attention_ldam`
- `stepcond_class_attention_ldam_m01`
- `stepcond_class_attention_ldam_m005`

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
- Ordinary desktop/browser/remote-control graphics processes do not block training; only obvious unrelated training/compute jobs should make the heartbeat wait and report.
- If an existing YoloCTM Python training process is active and making progress, monitor only; do not duplicate a run.
- If a machine reboot happens, append `AutoResearch/reboot_events.tsv` with the boot time and candidate-specific reboot count.
