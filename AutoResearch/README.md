# AutoResearch Archive

This directory stores the WM-811K YoloCTM experiment archive. It is inspired by
`karpathy/autoresearch`, but tailored to this local wafer-map classification
project.

## Purpose

- Keep the WM-811K protocol stable while exploring model and loss variants.
- Record every candidate, including failed and crashed runs.
- Prefer validation-only screening for routine iteration.
- Reserve the held-out test split for predeclared milestone checks.
- Keep enough config and metric history to reproduce the research trail from
  the remote repository without committing datasets or checkpoints.

## Fixed Protocol

Recent experiments assume:

- Dataset: `data/wm811k_cls`
- Source pickle: `data/MIR-WM811K/LSWMD.pkl`
- Split: `70 / 15 / 15`
- Classes: 9 with `none` included
- Image size: `224`
- Epoch budget: usually `10` for screening runs
- Batch: `64`
- Workers: `0`

Large artifacts remain local under `runs/`, `data/`, and cache folders; those
paths are ignored by Git.

## Archive Files

- `results.tsv` - main experiment ledger.
- `logs/*.json` - one structured summary per logged run.
- `configs/*.yaml` - candidate pipeline configs.
- `experiment_queue.md` - active and historical candidate queue.
- `program.md` - higher-level research program notes.
- `scripts/log_experiment.py` - helper used to append run summaries.
- `reboot_events.tsv` - workstation restart notes from long experiment cycles.

## Best Recorded Results

Frozen milestone single model:

- `autoresearch_yoloctm_slim_dkd_ema_calselect_priorcal_20260527_175645`
- Params: about `10.525M`
- Test accuracy: `0.98242`
- Test macro precision / recall / F1: `0.91093 / 0.92941 / 0.91878`

Latest no-distillation validation best in the ledger:

- `autoresearch_yoloctm_nodistill_stepcond_class_attention_ldam_m01_cbr010_cwp04_tau04_e10_20260607_132608`
- Params: about `10.527M`
- Validation accuracy: `0.98227`
- Validation macro precision / recall / F1: `0.92595 / 0.90571 / 0.91546`
- Test split: intentionally not evaluated

## Typical Loop

1. Add or select one candidate config.
2. Run `scripts/run_wm811k_pipeline.py --config <config> --check-config`.
3. Launch one training/evaluation job on the development machine.
4. Record the result in `results.tsv` and `logs/*.json`.
5. Mark the candidate `keep`, `discard`, or `crash`.
6. Use validation metrics for ordinary follow-up decisions.

## Development Machine Workflow

For long jobs, work from the development host local path:

```powershell
Set-Location E:\Cjn\PCB_Yolo
D:\anaconda3\envs\pcb_yolo\python.exe scripts\run_wm811k_pipeline.py --config configs\wm811k_cls.yaml
```

The mounted `R:\Cjn\PCB_Yolo` view is useful for editing and inspection, but
long GPU jobs should use the local path recorded above.

