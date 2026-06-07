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
   - Prepared config: `AutoResearch/configs/wm811k_autoresearch_stepcond_class_attention_ldam_m01_cbr010_cwp06.yaml`.
   - Planned run name: `autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr010_cwp06_tau04_e10`.
   - Recovery wrapper: `scripts/run_stepcond_class_attention_ldam_m01_cbr010_cwp06.cmd`.
   - Single factor over the current best: keep architecture/readout/sampler fixed, but increase `class_weight_power` from `0.5` to `0.6`.
   - Rationale: none-aware sampling was too disruptive. A modest objective-side increase in inverse-frequency class weighting may improve minority class macro F1 without changing the input distribution.
   - Launched on 2026-06-07 12:19 +08:00 as `autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr010_cwp06_tau04_e10_20260607_121941` via scheduled task `WM811K_AutoResearch_ClassAttentionLDAMM01CBR010CWP06`; task was disabled after manual start to avoid a duplicate 23:59 trigger.
   - Initial health check showed epoch 1 progressing on GPU with `[ldam] max_margin=0.1000 starts at epoch 7` and `[cbr] classifier regularization weight=0.0100 ... starts at epoch 7`.
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
   - LDAM-DRW margin 0.1 + CBR 0.005 completed on 2026-06-07 as `autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr005_tau04_e10_20260607_050437`; result: keep, val macro F1 `0.9131313775284673`.
   - LDAM-DRW margin 0.1 + CBR 0.01 completed on 2026-06-07 as `autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr010_tau04_e10_20260607_060435`; result: keep, val macro F1 `0.9133844769616716`.
   - LDAM-DRW margin 0.1 + CBR 0.02 completed on 2026-06-07 as `autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr020_tau04_e10_20260607_070334`; result: discard, val macro F1 `0.9128871683813008`.
   - LDAM-DRW margin 0.1 + CBR 0.01 + classwise expert fusion completed on 2026-06-07 as `autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr010_expert_tau04_e10_20260607_080419`; result: discard, val macro F1 `0.9109162034388688`.
   - LDAM-DRW margin 0.1 + CBR 0.01 + global log-prob fusion completed on 2026-06-07 as `autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr010_logfusion_tau04_e10_20260607_090408`; result: discard, val macro F1 `0.8899943351556002`.
   - LDAM-DRW margin 0.1 + CBR 0.01 + polar spatial encoding completed on 2026-06-07 as `autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr010_polarenc_tau04_e10_20260607_100351`; result: discard, val macro F1 `0.9010680369633918`.
   - LDAM-DRW margin 0.1 + CBR 0.01 + none-aware sampling completed on 2026-06-07 as `autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr010_noneaware075_tau04_e10_20260607_110426`; result: discard, val macro F1 `0.8958599987513993`.
   - LDAM-DRW margin 0.1 + CBR 0.01 + class weight power 0.6 completed on 2026-06-07 as `autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr010_cwp06_tau04_e10_20260607_121941`; result: discard, val macro F1 `0.8982021938248087`.
   - LDAM-DRW margin 0.1 + CBR 0.01 + class weight power 0.4 completed on 2026-06-07 as `autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr010_cwp04_tau04_e10_20260607_132608`; result: keep, val macro F1 `0.9154644535660744`.
   - Prepared config: `AutoResearch/configs/wm811k_autoresearch_stepcond_class_attention_ldam_m01_cbr010_cwp03.yaml`.
   - Planned run name: `autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr010_cwp03_tau04_e10`.
   - Recovery wrapper: `scripts/run_stepcond_class_attention_ldam_m01_cbr010_cwp03.cmd`.
   - Single factor over the new best: decrease `class_weight_power` from `0.4` to `0.3`; keep natural sampling, architecture, class-specific attention readout, `ldam_max_margin=0.1`, `classifier_cbr_weight=0.01`, no distillation, 10 epochs, validation-only screening, and `test.enabled: false`.
   - If `cwp03` does not beat `0.9154644535660744`, stop class-weight-power tuning and move to a qualitatively different structure/evidence direction.
   - Launched on 2026-06-07 14:20 +08:00 as `autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr010_cwp03_tau04_e10_20260607_142019`; active run, monitor/recover this run first.

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
- `stepcond_class_attention_ldam_m01_cbr005`
- `stepcond_class_attention_ldam_m01_cbr010`
- `stepcond_class_attention_ldam_m01_cbr020`
- `stepcond_class_attention_ldam_m01_cbr010_expert`
- `stepcond_class_attention_ldam_m01_cbr010_logfusion`
- `stepcond_class_attention_ldam_m01_cbr010_polarenc`
- `stepcond_class_attention_ldam_m01_cbr010_noneaware075`
- `stepcond_class_attention_ldam_m01_cbr010_cwp06`
- `stepcond_class_attention_ldam_m01_cbr010_cwp04`

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
