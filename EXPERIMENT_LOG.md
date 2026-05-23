# Experiment Log

## autoresearch_yoloctm_logprob_ensemble_20260522_170000 - 2026-05-22
- **System**: Fixed log-probability ensemble of YOLO26m, full CTM adapter, and low-rank CTM adapter
- **Config**: weights=(0.6, 0.2, 0.2), prior_logit_tau=0.025, imgsz=224, fixed WM811K split, eval batch=128
- **Result**: test macro F1 = 0.9118345879; test accuracy = 0.9822289041
- **Verdict**: positive for accuracy/macro-F1, negative for parameter footprint (32.655M summed inference params)
- **Reproduce**: `D:\anaconda3\envs\pcb_yolo\python.exe scripts\evaluate_wm811k_ensemble.py --data data\wm811k_cls --run-dir runs\classify\autoresearch_yoloctm_logprob_ensemble_20260522_170000 --device 0 --batch 128 --imgsz 224 --yolo-run runs\classify\wm811k_yolo26m_20260518_152306 --ctm-checkpoint runs\classify\autoresearch_yoloctm_ctmadapter_20260521_195218\best_yoloctm.pt --lowrank-checkpoint runs\classify\autoresearch_yoloctm_ctmadapter_lowrank_priorcal_20260522_155358\best_yoloctm.pt --weights 0.6,0.2,0.2 --prior-tau 0.025 --status keep`
