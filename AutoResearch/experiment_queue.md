# WM811K YoloCTM Experiment Queue

Last updated: 2026-06-06

This queue is for the no-distillation, <=10 epoch, validation-only AutoResearch track. Do not use test metrics for selection or tuning.

## Current Best

- Run: `autoresearch_yoloctm_nodistill_stepcond_class_attention_readout_tau04_e10_20260606_135126`
- Validation macro F1: `0.9115232889273587`
- Protocol: no distillation, 10 epochs, validation-only screening, `test.enabled: false`
- Main idea: step-conditioned adaptive CTM thought steps on top of residual YOLO feature fusion, with class-specific CTM token attention readout.

## Active / Next Runs

1. Validation-only evidence package for the current best:
   - Current best checkpoint: `runs/classify/autoresearch_yoloctm_nodistill_stepcond_class_attention_readout_tau04_e10_20260606_135126/best_yoloctm.pt`
   - Generated ablation tables at `runs/diagnostics/ablation_table_nodistill`.
   - Generated class-delta reports at `runs/diagnostics/class_delta_class_attention_vs_stepcond`.
   - Generated balanced readout evidence figures at `runs/diagnostics/class_attention_readout_val_figures_balanced` using `--split val --per-class-samples 24`; the selected cases cover all 9 classes.
   - Rationale: the last three readout-side follow-ups were shared attention discard, polar discard, and entropy discard/equal-to-best. Before adding more losses or priors, inspect what the kept class-specific readout actually attends to and which validation classes drive its gain.

2. Next active run:
   - Active run: `autoresearch_yoloctm_nodistill_stepcond_class_attention_min5_tau04_e10_20260607_000637`.
   - Config: `AutoResearch/configs/wm811k_autoresearch_stepcond_class_attention_min5.yaml`.
   - Recovery wrapper: `scripts/run_stepcond_class_attention_min5.cmd`.
   - Single factor: keep the current best class-specific attention architecture and change only `adaptive_min_steps` from `4` to `5`, while keeping `adaptive_confidence_threshold: 0.90`.
   - Rationale: `halt95` proved that asking more low-confidence samples to continue from step 4 to step 6 increases average compute but does not change official validation decisions. `min5` tests the sharper question of whether forcing one extra thought step for every validation sample makes the step-conditioned trajectory useful in a protocol-valid way.
   - Launched on 2026-06-07 00:06 +08:00 via `WM811K_AutoResearch_ClassAttentionMin5`; after launch, old completed class-attention scheduled tasks were disabled to prevent future duplicate relaunches.
   - Added metadata-only external audit script: `scripts/audit_external_wafer_dataset.py`.
   - Added protocol doc: `research-wiki/external_wafer_robustness_protocol.md`.
   - No external wafer benchmark is currently present under `data/`; do not evaluate external performance until dataset placement, metadata audit, and label mapping are committed.
   - The class-attention mean-blend run is completed and discarded; it helped Scratch but damaged Near-full, Edge-Loc, and Center, so do not continue the mean-fallback/gating direction without a stronger class-conditional rationale.
   - The class-attention halt95 run is completed and discarded/equal-to-best; it increased average thought steps but did not exceed the current validation threshold.

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
