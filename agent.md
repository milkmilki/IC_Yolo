# WM811K Agent Notes

This file records the practical runbook learned from the latest WM811K AutoResearch runs.
Keep it short and operational.

## Autonomous development-host execution

- On 2026-05-26 night, the user explicitly requested that the agent continue experiment cycles autonomously, launch remote commands itself, and recover after crashes without waiting for manual intervention. This supersedes the earlier visible-terminal-only launch preference.
- On 2026-05-27, the user explicitly requested a thread-attached heartbeat for this conversation instead of a standalone workspace automation. A 30-minute heartbeat now owns monitoring, recovery, evaluation, and subsequent experiment launches; do not create a parallel standalone runner.
- The heartbeat must tolerate the mounted `R:` drive disappearing while the development host is powered off: begin each wakeup by attempting SSH to the development host and operate on `E:\Cjn\PCB_Yolo` once it is reachable.
- The automation may inspect the mounted workspace, but every long training/evaluation command must execute through SSH against the development machine's local project path, carried as a Base64 PowerShell `-EncodedCommand` in a locally held foreground SSH process.
- Treat the project as local on the development machine:
  - `E:\Cjn\PCB_Yolo`
- Prefer the project Python explicitly:
  - `D:\anaconda3\envs\pcb_yolo\python.exe`
- If working from `cmd`, use `cd /d E:\Cjn\PCB_Yolo`.
- If working from PowerShell, use `Set-Location E:\Cjn\PCB_Yolo`.
- The agent may inspect artifacts, record results, implement one next candidate, and start/resume one controlled compute job; it must never start a second computation while a run or evaluation is active.

## Stability Notes

- The development host has suffered repeated BugCheck restarts during earlier GPU and CPU phases. A `400 W` power floor and `300-1200 MHz` graphics-clock lock mitigated but did not prove a root-cause fix.
- The completed DKD experiment used effective batch `64` as `micro_batch: 16` with gradient accumulation `4`, plus resumable epoch checkpoints.
- Preserve `last_yoloctm.pt` from interrupted runs; after a process exit or host restart, record the event and autonomously resume the same candidate rather than starting a concurrent duplicate.
- Apply `nvidia-smi -pl 400` and `nvidia-smi -lgc 300,1200` before each GPU training or GPU evaluation phase. The user requested GPU test evaluation for speed; run it only after training has ended and no competing process remains.
- `autoresearch_yoloctm_crossscan_dkd_priorcal_20260526_201638` reached epoch 3 (`val_macro_f1=0.8838`) before the development host was powered off by `RuntimeBroker.exe` at `2026-05-26 23:00:32` (event `1074`, not a BugCheck). It resumed after the `2026-05-27 09:12:53` boot, reached epoch 8 with best observed `val_macro_f1=0.8967` at epoch 6, then was interrupted by `BugCheck 0x50` reported at `2026-05-27 11:00:47`; resume it from `last_yoloctm.pt`.
- If the user interrupts a foreground terminal, always check local process state and `pipeline.log` before restarting.

## Foreground health check

- After starting a foreground training run, verify within 1 minute that a new run directory and `pipeline.log` exist under:
  - `E:\Cjn\PCB_Yolo\runs\classify\<run_name>\pipeline.log`
- Check the first epoch within about 5 minutes from the development machine:
  - `powershell -NoProfile -Command "Get-Content -Tail 80 E:\Cjn\PCB_Yolo\runs\classify\<run_name>\pipeline.log"`
- If no epoch appears after about 5 minutes, run these local checks before waiting longer:
  - `wmic process get ProcessId,Name,CommandLine | findstr /i run_wm811k_pipeline`
  - `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits`
- Treat the run as unhealthy and restart the foreground task when all of these are true:
  - no `Epoch 001` line in `pipeline.log`
  - no active local `python.exe ... run_wm811k_pipeline.py` process, or GPU memory/utilization is idle
  - the run directory has no `best_yoloctm.pt`
- If the local process is alive and GPU is busy, do not restart. Keep waiting and re-check in 3-5 minutes.
- If the run directory was created but the unhealthy run never reached epoch 1, log it as `crash` in `AutoResearch/results.tsv` and write a JSON crash summary before relaunching.
- If `Epoch 001` appears, the run is considered healthy. Let the foreground command continue until validation/test/logging completes.

## Stable WM811K settings

- Keep the fixed protocol unchanged:
  - split: `70/15/15`
  - image size: `224`
  - epochs: `10`
  - batch: `64`
  - `workers: 0`
- `workers > 0` caused hangs around epoch 5 on this Windows setup.
- Keep `data/wm811k_cls` as-is; do not regenerate it unless the source data changes.

## Logging and inspection

- Main pipeline log:
  - `E:\Cjn\PCB_Yolo\runs\classify\<run_name>\pipeline.log`
- YoloCTM checkpoint:
  - `E:\Cjn\PCB_Yolo\runs\classify\<run_name>\best_yoloctm.pt`
- Metrics written by the pipeline:
  - `metrics/val_classification_report.json`
  - `metrics/test_classification_report.json`
  - matching CSV and confusion-matrix files
- AutoResearch bookkeeping:
  - `AutoResearch/results.tsv`
  - `AutoResearch/logs/*.json`

## Known pitfalls

- `AutoResearch/scripts/log_experiment.py` can fail when JSON serialization sees `WindowsPath`.
  - TSV may already be written before the JSON step fails.
  - If that happens, regenerate the JSON with `default=str` handling.
- A missing `pipeline.log` in the mounted network-drive view does not always mean the run never existed.
  - Check the development-machine local path directly.
  - If only `best_yoloctm.pt` exists, the run may have completed training but lost its log trail.
- When a run is still alive, inspect the latest epoch from `pipeline.log`.
  - If it is stuck, check whether the local development-machine `python.exe` process is still present.
- A run can fail before creating its run directory. In that case, first re-run `--check-config`, then relaunch the local foreground command.
- Do not start a second run until process checks show the prior run is gone, otherwise multiple runs can compete for the same GPU.

## Result snapshot

- Baseline to beat: `runs/classify/wm811k_yolo26m_20260518_152306`
  - test acc `0.97849`
  - macro P `0.91412`
  - macro R `0.87689`
  - macro F1 `0.89345`
  - params about `11.634M`
- Best fused experiment so far:
  - `autoresearch_yoloctm_fused_20260520_1930`
  - params `10.476M`
  - test macro F1 `0.86607`
- Smaller low-rank follow-up:
  - `autoresearch_yoloctm_lowrank_20260520_2000`
  - params `9.901M`
  - test macro F1 `0.85581`
  - useful as a size probe, but not a keep
- Best YoloCTM as of 2026-05-21:
  - `autoresearch_yoloctm_yolohead_macroselect_20260521_103941`
  - params `10.476M`
  - test acc `0.97595`
  - test macro P `0.86214`
  - test macro R `0.90279`
  - test macro F1 `0.88016`
  - key lesson: reuse the original YOLO classification head, add CTM residual logits, and select checkpoints by validation macro F1 instead of accuracy.
- Best result against the standard baseline as of 2026-05-21:
  - `autoresearch_yoloctm_ctmadapter_priorcal_20260521_214700`
  - source checkpoint: `autoresearch_yoloctm_ctmadapter_20260521_195218`
  - params `10.525M`
  - test acc `0.97934`
  - test macro P `0.90362`
  - test macro R `0.89635`
  - test macro F1 `0.89835`
  - key lesson: a CTM residual feature adapter can over-boost minority recall, but validation-selected class-prior logit calibration with `tau=0.4` restores precision and beats the standard `yolo26m-cls` baseline while staying smaller.
- Best compact ensemble Pareto point as of 2026-05-26:
  - `autoresearch_yoloctm_slim_logprob_ensemble_20260526_0053`
  - active branches: `0.75*yolo26m + 0.25*ctm_adapter + prior_tau=0.025`
  - params `22.159M`
  - test acc `0.98092`
  - test macro P `0.92015`
  - test macro R `0.89192`
  - test macro F1 `0.90436`
  - key lesson: removing the low-rank third ensemble branch saves `10.496M` parameters while retaining a clear F1 gain over either single model; future compression work should target the two-branch complementarity first.
- Best single YoloCTM as of 2026-05-26:
  - `autoresearch_yoloctm_slim_dkd_priorcal_20260526_120641`
  - teacher: fixed `0.75*yolo26m + 0.25*ctm_adapter + prior_tau=0.025` compact ensemble
  - params `10.525M`
  - test acc `0.98138`
  - test macro P `0.90203`
  - test macro R `0.90150`
  - test macro F1 `0.89990`
  - key lesson: DKD transfers part of the two-branch complementarity into one CTM residual adapter, improving `Scratch`, `Edge-Loc`, `Loc`, and `Random` while surpassing the prior single-model F1 by `0.00155` at the same size.
- Next compression direction selected on 2026-05-26:
  - add Logit Standardization to DKD following Sun et al., CVPR 2024, `https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Logit_Standardization_in_Knowledge_Distillation_CVPR_2024_paper.html`.
  - rationale: standardized teacher/student logits preserve class relations without forcing a compact student to match ensemble confidence magnitude; it adds no inference parameters and directly targets the remaining precision gap.
  - implemented as `distill_mode: dkd_logit_std` with the same compact-ensemble teacher cache and fixed protocol; CPU syntax/config and loss-gradient preflights passed before launching training.
  - result: discard `autoresearch_yoloctm_slim_dkd_logitstd_priorcal_20260526_134214`, params `10.525M`, test acc `0.97814`, macro P `0.90043`, macro R `0.83911`, macro F1 `0.86118`; normalization severely reduced minority recall, especially `Scratch` and `Near-full`.
- Next distillation direction selected after the failed logit-standardization trial:
  - prepare DIST relational distillation following Huang et al., NeurIPS 2022, `https://proceedings.neurips.cc/paper_files/paper/2022/hash/da669dfd3c36c93905a17ddba01eef06-Abstract-Conference.html`.
  - rationale: DIST is designed for stronger teachers and transfers inter-class/intra-class prediction relations rather than forcing exact softened logits; this matches a compact ensemble teacher and avoids the confidence-scale destruction observed above.
  - implemented as `distill_mode: dist` using the existing compact-teacher cache; the Pearson relation loss matches the authors' released `DIST` implementation, and CPU syntax/config/loss-gradient preflights passed. Do not launch alongside another computation.
  - result: discard `autoresearch_yoloctm_slim_dist_priorcal_20260526_162704`, params `10.525M`, test acc `0.97891`, macro P `0.89102`, macro R `0.87230`, macro F1 `0.87851`; relational loss did not retain the DKD student's `Loc`, `Scratch`, and `Near-full` balance.
- Next high-performance candidate selected after DIST:
  - combine the successful compact-ensemble DKD objective with the implemented VMamba-inspired shared cross-scan CTM token mixer (`token_mixer: cross_scan`), following Liu et al., VMamba, `https://arxiv.org/abs/2401.10166`.
  - rationale: both loss replacements degraded minority-class balance; cross-scan instead adds a small spatial structural prior to the best DKD student, and its previous non-DKD trial was interrupted by system crashes before it could be evaluated.
  - keep the controlled `micro_batch: 16` x accumulation `4` launch, `400 W` power limit, `300,1200` GPU clock lock, resumable checkpoints, and user-authorized controlled GPU test/metric evaluation.
  - result: discard `autoresearch_yoloctm_crossscan_dkd_priorcal_20260526_201638`, params `10.535M`, test acc `0.98073`, macro P `0.92305`, macro R `0.85492`, macro F1 `0.88270`; validation macro F1 reached `0.90680`, but the spatial mixer did not generalize and reduced test recall for `Loc`, `Near-full`, and `Scratch`.
- Next high-performance candidate selected on 2026-05-27:
  - combine the successful compact-ensemble DKD objective with the existing frozen trained-YOLO anchor path (`freeze_yolo_anchor: true`, `token_mixer: none`) initialized from `runs/classify/wm811k_yolo26m_20260518_152306/weights/best.pt`.
  - rationale: DKD is the only distillation loss that improved the single model, while cross-scan produced a precision-heavy validation overfit. A frozen precision anchor with a trainable CTM residual correction tests whether ensemble-derived tail recovery can be added without allowing the clean YOLO view to drift.
  - keep the fixed protocol and controlled GPU recovery settings; compare against the single-model DKD test macro F1 `0.899897` at `10.525M`.
  - result: discard `autoresearch_yoloctm_frozenanchor_dkd_priorcal_20260527_120425`, params `10.525M`, test acc `0.97718`, macro P `0.91491`, macro R `0.86384`, macro F1 `0.88648`; freezing retained precision but blocked the `Loc` and `Edge-Loc` corrections learned by trainable DKD.
- Next evidence-driven DKD candidate selected on 2026-05-27:
  - use the successful trainable CTM residual adapter and DKD objective with the stronger three-branch teacher cache `AutoResearch/cache/train_logprob_ensemble_060202_tau0025.npz` (teacher test macro F1 `0.91183`) instead of the slim two-branch teacher (`0.90436`).
  - rationale: ordinary KD from this teacher previously lost recall, but DKD is the only tested transfer loss that improved a single model; this isolates whether stronger teacher complementarity improves the same `10.525M` student without adding inference parameters.
  - logging now supports `status: auto` against the current best single-model test macro F1 `0.899897`, preventing validation-only gains from being marked as keeps.

## Practical workflow

1. Check `AutoResearch/results.tsv` first.
2. If the latest run is unclear, inspect the local development-machine `pipeline.log`.
3. If the run finished but metrics are missing, run validation/test from the development machine on controlled GPU (`--eval-device 0`) after reapplying the power/clock limits and verifying there is no concurrent process.
4. After every run, make sure both:
   - `AutoResearch/results.tsv` has a row
   - `AutoResearch/logs/` has a JSON summary
5. Prefer small structural changes over wider tuning.
6. For this campaign, lower parameter count only counts if metrics stay competitive.
7. When launching a new foreground run, always do the foreground health check above before leaving it unattended.
8. If a YoloCTM run has high macro recall but weak macro precision, run a validation-selected class-prior logit calibration check before changing architecture again.
   - For the current CTM adapter, `tau=0.4` was selected on validation and improved test macro F1 from `0.87726` to `0.89835`.
   - Log calibrated evaluations as separate AutoResearch runs so raw and calibrated metrics stay auditable.
