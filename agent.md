# WM811K Agent Notes

This file records the practical runbook learned from the latest WM811K AutoResearch runs.
Keep it short and operational.

## Local development-host execution

- The user will connect to the development machine and open a terminal there for training.
- Do not launch training by wrapping commands through SSH from the mounted network-drive view.
- Treat the project as local on the development machine:
  - `E:\Cjn\PCB_Yolo`
- Prefer the project Python explicitly:
  - `D:\anaconda3\envs\pcb_yolo\python.exe`
- If working from `cmd`, use `cd /d E:\Cjn\PCB_Yolo`.
- If working from PowerShell, use `Set-Location E:\Cjn\PCB_Yolo`.

## Stable foreground launch

- For WM811K AutoResearch training, run this directly inside a terminal on the development machine:
  - `cd /d E:\Cjn\PCB_Yolo && D:\anaconda3\envs\pcb_yolo\python.exe scripts\run_wm811k_pipeline.py --config AutoResearch\configs\wm811k_autoresearch.yaml`
- Prefer a visible foreground terminal for long GPU training so progress and interruptions are obvious.
- Avoid `Start-Process` background launches unless the user explicitly asks for detached training.
- Avoid Windows scheduled tasks for this training loop unless there is no foreground option. They repeatedly exited early with status `-1073741510` or re-fired unexpectedly.
- If the user interrupts the foreground terminal, always check local process state and `pipeline.log` before restarting.

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

## Practical workflow

1. Check `AutoResearch/results.tsv` first.
2. If the latest run is unclear, inspect the local development-machine `pipeline.log`.
3. If the run finished but metrics are missing, run validation/test from the development machine.
4. After every run, make sure both:
   - `AutoResearch/results.tsv` has a row
   - `AutoResearch/logs/` has a JSON summary
5. Prefer small structural changes over wider tuning.
6. For this campaign, lower parameter count only counts if metrics stay competitive.
7. When launching a new foreground run, always do the foreground health check above before leaving it unattended.
8. If a YoloCTM run has high macro recall but weak macro precision, run a validation-selected class-prior logit calibration check before changing architecture again.
   - For the current CTM adapter, `tau=0.4` was selected on validation and improved test macro F1 from `0.87726` to `0.89835`.
   - Log calibrated evaluations as separate AutoResearch runs so raw and calibrated metrics stay auditable.
