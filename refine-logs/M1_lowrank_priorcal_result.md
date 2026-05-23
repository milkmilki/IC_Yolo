# M1 Result: low-rank CTM adapter with prior calibration

Date: 2026-05-22
Run: runs/classify/autoresearch_yoloctm_ctmadapter_lowrank_priorcal_20260522_155358
Config: AutoResearch/configs/wm811k_autoresearch.yaml
Protocol: fixed WM811K, epochs=10, imgsz=224, batch=64, workers=0, device=0, prior_logit_tau=0.4

## Preflight

- Local development-machine command path was used: E:\Cjn\PCB_Yolo
- No existing python.exe training process was found before launch.
- GPU check before launch: utilization 0%, memory 747 MiB.
- Config check resolved algorithm=yoloctm, epochs=10, run name autoresearch_yoloctm_ctmadapter_lowrank_priorcal_<timestamp>.
- py_compile passed for scripts/train_wm811k_yoloctm.py, scripts/run_wm811k_pipeline.py, and AutoResearch/scripts/log_experiment.py.
- Structural low-rank adapter/config changes were already committed before launch:
  - fc2b0a0 Add-low-rank-CTM-adapter-AutoResearch-experiment
  - 16495a0 Clarify-fixed-prior-calibration-in-AutoResearch-config

## Training outcome

Training completed cleanly for 10 epochs and wrote all required artifacts:

- best_yoloctm.pt
- pipeline.log
- metrics/val_classification_report.json
- metrics/test_classification_report.json
- matching CSV reports and confusion matrices
- AutoResearch/results.tsv row
- AutoResearch/logs/20260522_163017_autoresearch_yoloctm_ctmadapter_lowrank_priorcal_20260522_155358.json

Best validation macro F1 during training occurred at epoch 9: 0.8649 before calibrated final evaluation.

## Metrics

| Split | Accuracy | Macro P | Macro R | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| val | 0.9774128893 | 0.9082037652 | 0.8769646009 | 0.8886174607 |
| test | 0.9772175321 | 0.9040333482 | 0.8584082099 | 0.8779588275 |

Parameter count: 10.496M.

## Decision

Status: discard.

Rationale:

- The run reduced parameters from the prior CTM adapter baseline (10.525M -> 10.496M), but only by about 0.029M.
- Test macro F1 dropped from the prior calibrated best 0.8983463307 to 0.8779588275.
- Test macro F1 is also below the secondary threshold 0.89345 from the standard YOLO baseline.
- The loss is mainly in macro recall (0.8963527099 -> 0.8584082099), especially Loc and several minority classes.

Per EXPERIMENT_PLAN.md, M2 should run only if M1 is competitive. M1 is not competitive, so M2 and M3 are not justified under the current plan.
