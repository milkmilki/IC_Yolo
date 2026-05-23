# Ensemble Result: fixed log-probability YOLO26m + CTM adapters

Date: 2026-05-22
Run: runs/classify/autoresearch_yoloctm_logprob_ensemble_20260522_170000
Script: scripts/evaluate_wm811k_ensemble.py
Protocol: fixed WM811K split, imgsz=224, batch=64-equivalent bookkeeping, eval batch=128, workers=0 for CTM loaders, device=0

## Hypothesis

M1 showed that the low-rank CTM adapter alone lost minority recall, but its errors were not identical to the full CTM adapter or the YOLO26m baseline. A fixed log-probability ensemble can combine YOLO26m precision with CTM adapter recall while leaving the data split, image size, and prior completed training runs unchanged.

## Implementation and review

- Added `scripts/evaluate_wm811k_ensemble.py`.
- Fixed `load_yoloctm_checkpoint` to infer old checkpoints with a feature adapter from `model_state` keys when the stored args lack `feature_adapter`.
- Cross-model review found one critical sample-order issue in the first ensemble implementation; fixed by driving YOLO predictions from `ImageFolder.samples` in the same order as `ImageFolder.targets`.
- A second review confirmed no remaining critical ground-truth, leakage, or metric-ordering issue.
- Structural code commit before evaluation:
  - `567cf0b Add-fixed-WM811K-ensemble-evaluator`

## Command

```cmd
cd /d E:\Cjn\PCB_Yolo && D:\anaconda3\envs\pcb_yolo\python.exe scripts\evaluate_wm811k_ensemble.py --data data\wm811k_cls --run-dir runs\classify\autoresearch_yoloctm_logprob_ensemble_20260522_170000 --device 0 --batch 128 --imgsz 224 --yolo-run runs\classify\wm811k_yolo26m_20260518_152306 --ctm-checkpoint runs\classify\autoresearch_yoloctm_ctmadapter_20260521_195218\best_yoloctm.pt --lowrank-checkpoint runs\classify\autoresearch_yoloctm_ctmadapter_lowrank_priorcal_20260522_155358\best_yoloctm.pt --weights 0.6,0.2,0.2 --prior-tau 0.025 --status keep
```

## Metrics

| Split | Accuracy | Macro P | Macro R | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| val | 0.9813444342 | 0.9237807676 | 0.9017953689 | 0.9109065665 |
| test | 0.9822289041 | 0.9220839838 | 0.9044817596 | 0.9118345879 |

Ensemble inference parameter footprint: 32.655M summed over source models.

## Decision

Status: keep for performance, but not for size.

Rationale:

- Test macro F1 improved from the best single YoloCTM prior-calibrated result 0.8983463307 to 0.9118345879 (+0.01349).
- Test accuracy improved from 0.9793377279 to 0.9822289041.
- This is below the campaign target macro-F1=0.95, but it is the strongest logged performance so far.
- The tradeoff is a much larger inference footprint because all three source models are used.
- AutoResearch/results.tsv and AutoResearch/logs/20260522_203301_autoresearch_yoloctm_logprob_ensemble_20260522_170000.json were written.

## Caveat

The ensemble weights and prior tau were selected on validation metrics. Treat validation numbers as tuned-selection diagnostics; the test row is the held-out result to compare.
