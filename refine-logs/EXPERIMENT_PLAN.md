# EXPERIMENT_PLAN: WM811K YOLO-CTM Improvement

Date: 2026-05-22
Project: E:\Cjn\PCB_Yolo
Python: D:\anaconda3\envs\pcb_yolo\python.exe
Target workflow: /experiment-bridge

## 0. Required context already read

The executor must follow these project documents before editing or running:

- AGENTS.md
- agent.md
- AutoResearch/README.md
- AutoResearch/program.md
- AutoResearch/configs/wm811k_autoresearch.yaml

## 1. Hard constraints

1. Run all training and evaluation from a local foreground terminal on the development machine.
   - Workdir: E:\Cjn\PCB_Yolo
   - Do not wrap long jobs through SSH.
   - Do not launch from a mounted network-drive path.
   - Avoid Start-Process, scheduled tasks, or detached background launches unless the user explicitly requests them.
2. Keep the fixed protocol unchanged.
   - Dataset: data/wm811k_cls
   - Raw source: data/MIR-WM811K
   - Split ratios: [70, 15, 15]
   - include_none: true
   - seed: 42
   - prepare.enabled: false
   - epochs: 10
   - imgsz: 224
   - batch: 64
   - device: "0"
   - workers: 0
3. Do not regenerate or modify the data split.
4. Do not increase epochs or image size.
5. Do not do a large hyperparameter search. Run one meaningful structural/compression variant at a time.
6. Use one GPU sequentially. Do not start a second training run until local process checks show the previous one is gone.
7. After every structural code change, create a git commit before launching the full 10-epoch run.
   - Do not revert unrelated user changes.
   - If unexpected dirty files appear, stop and ask the user.
8. After every completed, discarded, or crashed experiment, write both:
   - AutoResearch/results.tsv
   - AutoResearch/logs/*.json
9. This plan does not authorize long training by itself. Wait for user confirmation before starting any full run.

## 2. Current baseline and target

Reference standard YOLO baseline:

- Run: runs/classify/wm811k_yolo26m_20260518_152306
- Params: about 11.634M
- Test accuracy: 0.97849
- Test macro P/R/F1: 0.91412 / 0.87689 / 0.89345

Best logged YOLO-CTM result so far:

- Run: autoresearch_yoloctm_ctmadapter_priorcal_20260521_214700
- Source checkpoint: autoresearch_yoloctm_ctmadapter_20260521_195218
- Params: 10.525M
- Test accuracy: 0.97934
- Test macro P/R/F1: 0.90362 / 0.89635 / 0.89835
- Key lesson: CTM feature adapter improves recall; validation-selected class-prior logit calibration with tau=0.4 restores precision.

Immediate target:

- Match or exceed test macro F1 0.89835 while reducing parameters below 10.525M.
- Secondary acceptable tradeoff: keep test macro F1 at least 0.89345 while clearly reducing parameters versus 10.525M.

## 3. Preflight checks (no long training)

Run these before any experiment:

```cmd
cd /d E:\Cjn\PCB_Yolo
git status --short
wmic process get ProcessId,Name,CommandLine | findstr /i run_wm811k_pipeline
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
D:\anaconda3\envs\pcb_yolo\python.exe scripts\run_wm811k_pipeline.py --config AutoResearch\configs\wm811k_autoresearch.yaml --check-config
D:\anaconda3\envs\pcb_yolo\python.exe -m py_compile scripts\train_wm811k_yoloctm.py scripts\run_wm811k_pipeline.py AutoResearch\scripts\log_experiment.py
```

Success criteria:

- No active local training process for this project.
- GPU is not occupied by another project run.
- Resolved config still shows epochs=10, imgsz=224, batch=64, workers=0, prepare.enabled=false, algorithm=yoloctm.
- Syntax checks pass.

If current structural code changes for the low-rank adapter are uncommitted and intended, commit them before the first full run. Example commit message:

```cmd
git add scripts\train_wm811k_yoloctm.py scripts\run_wm811k_pipeline.py AutoResearch\configs\wm811k_autoresearch.yaml
git commit -m "Add low-rank CTM adapter AutoResearch experiment"
```

Only include files that are part of the intended structural/workflow change.

## 4. Run order

### M1 - MUST RUN: low-rank CTM feature adapter with prior calibration

Purpose:

- Finish the current intended local experiment properly.
- The prior SSH-based sanity attempt crashed before epoch 1 and is not a valid result.

Hypothesis:

- Replacing the full CTM-to-feature residual adapter with adapter_rank=32 keeps the CTM feature-adapter benefit while shaving parameters.
- Fixed prior_logit_tau=0.4 should preserve the precision/recall balance observed in the best priorcal run.

Config values:

```yaml
model:
  algorithm: yoloctm
  weights: yolo26m-cls.pt
  pretrained: true
  ctm:
    d_model: 96
    steps: 4
    dropout: 0.1
    class_weight_power: 0.5
    adapter_rank: 32
train:
  epochs: 10
  imgsz: 224
  batch: 64
  workers: 0
  name: autoresearch_yoloctm_ctmadapter_lowrank_priorcal
metrics:
  prior_logit_tau: 0.4
logging:
  autoresearch: true
```

Launch command, only after user confirmation:

```cmd
cd /d E:\Cjn\PCB_Yolo && D:\anaconda3\envs\pcb_yolo\python.exe scripts\run_wm811k_pipeline.py --config AutoResearch\configs\wm811k_autoresearch.yaml
```

Health check after launch:

```cmd
wmic process get ProcessId,Name,CommandLine | findstr /i run_wm811k_pipeline
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
```

Also inspect the newest run log locally:

```cmd
powershell -NoProfile -Command "Get-Content -Tail 80 E:\Cjn\PCB_Yolo\runs\classify\<run_name>\pipeline.log"
```

Success criteria:

- Required artifacts exist: best_yoloctm.pt, metrics JSON/CSV, confusion matrices, pipeline.log.
- AutoResearch/results.tsv has one row.
- AutoResearch/logs/ has a JSON summary.
- Keep if test macro F1 >= 0.89835, or if test macro F1 >= 0.89345 with a clear parameter reduction below 10.525M.
- Discard if test macro F1 drops below 0.89345 unless parameter reduction is large and validation metrics justify a follow-up.

### M2 - CONDITIONAL MUST RUN: smaller CTM state with rank-32 adapter

Run only if M1 completes cleanly and the result is competitive.

Purpose:

- Reduce parameter count further by lowering CTM state width while keeping the YOLO head and low-rank adapter path.

Hypothesis:

- d_model=80 with adapter_rank=32 may retain most of the calibrated CTM-adapter benefit while saving CTM projection/transition parameters.

Config delta from M1:

```yaml
model:
  ctm:
    d_model: 80
    steps: 4
    adapter_rank: 32
train:
  name: autoresearch_yoloctm_ctmadapter_d80_rank32_priorcal
metrics:
  prior_logit_tau: 0.4
logging:
  description: "CTM residual feature adapter with d_model=80, rank=32, and class-prior logit calibration tau=0.4"
```

Success criteria:

- Keep if test macro F1 >= 0.89345 and params are lower than M1.
- Strong keep if test macro F1 >= 0.89835 with params lower than M1.
- Discard if both validation macro F1 and test macro F1 are below M1 by more than 0.005 without a meaningful parameter reduction.

### M3 - OPTIONAL COMPRESSION: rank-16 adapter at d_model=96

Run only if M1 is strong but parameter reduction is too small, or if M2 hurts metrics too much.

Purpose:

- Isolate whether adapter rank is the main removable parameter source without changing CTM state width.

Config delta from M1:

```yaml
model:
  ctm:
    d_model: 96
    steps: 4
    adapter_rank: 16
train:
  name: autoresearch_yoloctm_ctmadapter_rank16_priorcal
metrics:
  prior_logit_tau: 0.4
logging:
  description: "More compressed low-rank CTM residual feature adapter (rank=16) with class-prior logit calibration tau=0.4"
```

Success criteria:

- Keep if test macro F1 >= 0.89345 and params are lower than M1.
- Prefer M1 over M3 if M3 saves only a tiny number of parameters but loses more than 0.003 macro F1.

## 5. Evaluation and logging rules

For every run:

1. Use dataset ground-truth labels from data/wm811k_cls/{val,test}; never use another model as ground truth.
2. Checkpoint selection must be by validation macro F1, as implemented in scripts/train_wm811k_yoloctm.py.
3. Report validation and test:
   - accuracy
   - macro precision
   - macro recall
   - macro F1
   - parameter count
4. Write parseable artifacts:
   - runs/classify/<run>/metrics/val_classification_report.json
   - runs/classify/<run>/metrics/test_classification_report.json
   - matching CSV reports and confusion matrices
5. Append AutoResearch/results.tsv and write AutoResearch/logs/*.json.
6. If AutoResearch logging writes TSV but JSON serialization fails, regenerate the JSON with default=str handling and do not duplicate the TSV row.
7. If a run crashes before metrics:
   - verify no local python process is still training
   - append a crash row to AutoResearch/results.tsv
   - write a JSON crash summary under AutoResearch/logs/
   - do not relaunch until the crash is diagnosed

## 6. Budget

- M1: 1 full 10-epoch run, estimated about 0.5-1.0 GPU-hour based on prior local runs.
- M2: 1 conditional full 10-epoch run.
- M3: optional 1 full 10-epoch run.
- Total planned budget after confirmation: 1-3 sequential GPU runs, no parallelism.

## 7. Handoff summary for /experiment-bridge

- Start with M0 preflight only.
- Before any full run, commit intended structural code changes.
- Run M1 first in a local foreground terminal.
- Parse results, update AutoResearch bookkeeping, then decide whether M2 or M3 is justified.
- Do not invent additional variants unless the user approves a new plan update.
