# External Wafer Robustness Protocol

This protocol is for checking whether the current WM811K YoloCTM direction generalizes beyond the local WM811K split. It is intentionally conservative: external data must not become another hidden validation set.

## Current State

- Local data currently present: `data/MIR-WM811K` and `data/wm811k_cls`.
- No independent external wafer benchmark is present in the workspace yet.
- Current no-distillation WM811K validation-only best:
  - run: `autoresearch_yoloctm_nodistill_stepcond_class_attention_readout_tau04_e10_20260606_135126`
  - val macro F1: `0.9115232889273587`
  - checkpoint: `runs/classify/autoresearch_yoloctm_nodistill_stepcond_class_attention_readout_tau04_e10_20260606_135126/best_yoloctm.pt`

## Candidate External Data

MixedWM38 is a plausible external wafer-map benchmark, but it is not label-compatible with WM811K by default. It contains normal, single-defect, and mixed-defect patterns, so it needs an explicit mapping or a declared multi-label protocol before any performance claim.

Do not silently map mixed defects into WM811K single-label classes. Do not tune model choices from repeated external benchmark feedback.

## Step 1: Metadata Audit

Place an external dataset under a separate directory, for example:

```text
data/external/MixedWM38/
```

Supported audit layout is image-folder style:

```text
data/external/<dataset>/<class_name>/*.png
```

or split image-folder style:

```text
data/external/<dataset>/train/<class_name>/*.png
data/external/<dataset>/val/<class_name>/*.png
data/external/<dataset>/test/<class_name>/*.png
```

Run metadata-only audit:

```powershell
D:\anaconda3\envs\pcb_yolo\python.exe scripts\audit_external_wafer_dataset.py `
  --dataset-root data\external\MixedWM38 `
  --output-dir runs\diagnostics\external_audit_mixedwm38
```

By default, folders named `test` are skipped to avoid accidental test feedback. Use `--include-test-metadata` only if the protocol explicitly needs dataset-card counts, and still do not run model scoring from test during model selection.

## Step 2: Declare Label Semantics

Before evaluation, write a small mapping document that states:

- which external labels exactly match WM811K classes;
- which labels are case/format aliases only;
- which labels are mixed defects and therefore not single-label comparable;
- whether the external task is single-label, multi-label, open-set, or abstention-based;
- which split is used for protocol development and which split is reserved.

## Step 3: Frozen Checkpoint Evaluation

Only after Step 1 and Step 2 are committed, evaluate a frozen promoted checkpoint. The first checkpoint should be the current validation-only best listed above. Do not change architecture, calibration, or thresholds based on external evaluation unless a new external-validation protocol is declared first.

## Reporting Rule

External benchmark numbers are robustness evidence, not WM811K model-selection evidence. Report them separately from WM811K validation/test results and include the exact label mapping, ignored labels, and split policy.
