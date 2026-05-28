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
- Best frozen milestone single YoloCTM as of 2026-05-27:
  - `autoresearch_yoloctm_slim_dkd_ema_calselect_priorcal_20260527_175645`
  - selection protocol: promoted using fixed-`tau=0.1` validation macro F1 `0.918524`, then evaluated exactly once on the test lockbox.
  - params `10.525M`
  - test acc `0.98242`
  - test macro P `0.91093`
  - test macro R `0.92941`
  - test macro F1 `0.91878`
  - key lesson: EMA-exported weights substantially stabilized the successful slim-DKD student under the validation-only promotion rule. This test result is frozen for reporting and must not guide routine candidate selection.
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
  - result: discard `autoresearch_yoloctm_fullensemble_dkd_priorcal_20260527_125435`, params `10.525M`, test acc `0.98096`, macro P `0.91789`, macro R `0.88208`, macro F1 `0.89794`; stronger teacher improved tail recall relative to frozen/cross-scan trials but did not exceed slim-teacher DKD.
- Next primary-paper-motivated representation candidate selected on 2026-05-27:
  - add a BCL-inspired class-complement prototype auxiliary objective to the best slim-teacher DKD adapter, based on Zhu et al., CVPR 2022, `https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_Balanced_Contrastive_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2022_paper.html`.
  - implementation scope: normalized pooled CTM embeddings contrast against the existing CTM classifier rows as all-class prototypes (`prototype_bcl_weight: 0.05`, `temperature: 0.1`); training-only objective, zero added inference parameters, and explicitly an adaptation rather than a full dual-view BCL reproduction.
  - rationale: recent failures either over-raised precision or constrained minority correction; a class-complement representation target directly addresses `Loc` and `Random` separation under small micro-batches while retaining successful DKD.
  - result: discard `autoresearch_yoloctm_slim_dkd_prototypebcl_priorcal_20260527_142934`, params `10.525M`, test acc `0.97984`, macro P `0.90716`, macro R `0.88759`, macro F1 `0.89629`; the auxiliary representation objective did not preserve the incumbent DKD minority recall balance.
- Next metric-aligned DKD candidate selected on 2026-05-27:
  - retain the successful slim-teacher DKD architecture and loss, but select the saved checkpoint using a fixed validation class-prior adjustment `selection_prior_logit_tau: 0.1`, matching the calibration selected by the incumbent DKD run.
  - rationale: the incumbent was checkpoint-selected using uncalibrated validation macro F1 and only calibrated afterwards. Selecting against the deployed prediction rule can recover a better epoch without altering the model, distillation teacher, data protocol, or inference parameter count.
  - reward-hacking guard: subsequent development screening must not read the test split on each iteration. This candidate sets `test.enabled: false`, emits only validation metrics, uses the predeclared fixed `metrics.prior_logit_tau: 0.1`, and advances only against the incumbent calibrated validation macro F1 `0.903309`.
  - evaluate the held-out test split only after a candidate is promoted as a milestone under the predeclared validation rule; do not choose follow-up experiments from repeated test comparisons.
  - result: discard `autoresearch_yoloctm_slim_dkd_calselect_priorcal_20260527_162016`, params `10.525M`, calibrated validation macro F1 `0.903309`, exactly tied with the incumbent; no test-set metrics were generated.
- Next validation-only candidate selected on 2026-05-27:
  - add exponential moving average exported weights (`ema_decay: 0.999`) to the successful slim-teacher DKD adapter, following the weight-averaged teacher principle of Tarvainen and Valpola, NeurIPS 2017, `https://papers.nips.cc/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html`.
  - rationale: recent additional representation and spatial priors degraded tail balance, while EMA can smooth late-epoch optimizer noise without adding inference parameters or changing the DKD objective.
  - screen only on fixed calibrated validation macro F1 (`tau=0.1`); require a strict improvement above `0.903309` before any milestone test evaluation.
  - development result: promoted `autoresearch_yoloctm_slim_dkd_ema_calselect_priorcal_20260527_175645`, params `10.525M`, fixed-`tau=0.1` val acc `0.98231`, macro P `0.90717`, macro R `0.93266`, macro F1 `0.918524`; test has not yet been read and may now be evaluated exactly once as a predeclared milestone check.
  - milestone result: test was evaluated once after promotion with fixed `tau=0.1`; acc `0.98242`, macro P `0.91093`, macro R `0.92941`, macro F1 `0.918781`. Freeze this result; future search uses validation only.
- Next validation-only candidate selected after EMA promotion:
  - retain EMA export (`ema_decay: 0.999`), fixed calibrated validation screening (`tau=0.1`), and DKD, while switching only the teacher cache from the slim two-branch teacher to the pre-existing stronger three-branch teacher `AutoResearch/cache/train_logprob_ensemble_060202_tau0025.npz`.
  - rationale: before EMA, the stronger teacher gave lower validation F1 than the slim teacher; EMA's large validation gain justifies re-testing this single teacher factor under the now-stable optimization path. This decision is based on validation evidence available before the lockbox result, not on the frozen test score.
  - require strict improvement above the current development baseline val macro F1 `0.918524`; keep `test.enabled: false` unless promoted.
  - result: discard `autoresearch_yoloctm_fullensemble_dkd_ema_calselect_priorcal_20260527_194358`, params `10.525M`, fixed-`tau=0.1` val acc `0.98165`, macro P `0.90264`, macro R `0.93311`, macro F1 `0.916485`; no test metrics generated.
- Teacher-dependence control selected after user review on 2026-05-27:
  - retain the promoted EMA architecture and fixed validation-only protocol, but disable distillation entirely (`distill_logprobs: null`, `distill_weight: 0.0`) for `autoresearch_yoloctm_nodistill_ema_calselect_priorcal`.
  - rationale: the promoted DKD student may inherit a WM811K-specific teacher bias. A no-distillation EMA control distinguishes gains from EMA/model training versus gains requiring the specialized teacher, without consuming the test lockbox.
  - continue screening against the frozen development baseline val macro F1 `0.918524`; do not test unless it strictly promotes.
  - track baseline result: `autoresearch_yoloctm_nodistill_ema_calselect_priorcal_20260527_204846`, params `10.525M`, fixed-`tau=0.1` val acc `0.97367`, macro P `0.83977`, macro R `0.93206`, macro F1 `0.882153`; no test metrics generated. Per user direction, this now establishes an independent non-distilled development baseline rather than competing directly with the DKD milestone.
- Next non-distilled-track candidate selected on 2026-05-27:
  - keep no-distillation EMA unchanged and adjust only its predeclared validation prior-logit calibration from `tau=0.1` to `tau=0.4`, using prior non-distilled CTM validation evidence rather than test feedback.
  - rationale: the new no-distillation baseline is strongly recall-heavy and precision-limited; the prior non-distilled CTM adapter previously selected `tau=0.4` on validation, making this a track-specific calibration correction rather than a teacher-dependent change.
  - promotion threshold for this track is strict improvement above validation macro F1 `0.882153`; `test.enabled: false` remains mandatory.
  - result: track best `autoresearch_yoloctm_nodistill_ema_tau04_calselect_20260527_214912`, params `10.525M`, fixed-`tau=0.4` val acc `0.97984`, macro P `0.90404`, macro R `0.90317`, macro F1 `0.902814`; no test metrics generated.
- Next non-distilled-track candidate selected after calibration improvement:
  - add a conservative deferred LDAM-inspired target-margin objective (`loss: ldam_drw`, `ldam_max_margin: 0.2`, starts at epoch `7`) to the EMA non-distilled track, based on Cao et al., NeurIPS 2019, `https://papers.nips.cc/paper/2019/hash/621461af90cadfdaf0e8d4cc25129f91-Abstract.html`.
  - implementation is a scoped adaptation: retain the existing weighted cross-entropy and add class-frequency margins only late in training, rather than introducing a teacher or broad sampling change.
  - rationale: after calibrated EMA, remaining validation weakness is concentrated in `Loc` and `Scratch`; deferred minority margins target that deficit while preserving the newly balanced precision/recall operating point.
  - screen only against the non-distilled validation track best `0.902814`; do not generate test metrics unless this independent track later reaches a separately declared milestone.
  - result: discard `autoresearch_yoloctm_nodistill_ema_ldam_tau04_20260527_224641`, params `10.525M`, fixed-`tau=0.4` val acc `0.97892`, macro P `0.89474`, macro R `0.90187`, macro F1 `0.897267`; no test metrics generated. Deferred margins reduced precision and did not beat the non-distilled track best.
- Next non-distilled-track candidate selected after LDAM:
  - revert to the current best no-distillation EMA recipe with fixed `tau=0.4`, and extend training from `10` to `20` epochs.
  - rationale: the no-distillation EMA models still have comparatively low train accuracy and the 10-epoch calibrated run improved through the final epoch, so under-convergence is a plausible bottleneck. This changes only optimization budget, not teacher use, architecture, data split, or test access.
  - screen only against the non-distilled validation track best `0.902814`; do not generate test metrics.
  - result: over-budget diagnostic `autoresearch_yoloctm_nodistill_ema_tau04_e20_20260528_102736`, params `10.525M`, fixed-`tau=0.4` val acc `0.98169`, macro P `0.92070`, macro R `0.91070`, macro F1 `0.915030`; no test metrics generated. Per user correction on 2026-05-28, final non-distilled scoring is constrained to `<=10` epochs, so this does not count as track best.
- Next non-distilled-track candidate selected after e20:
  - keep the same no-distillation EMA `tau=0.4` recipe and extend from `20` to `30` epochs.
  - rationale: validation macro F1 was still at its best on epoch `20`, and train accuracy remained only `0.9713`, so additional training budget is still a plausible non-teacher improvement.
  - screen only against the non-distilled validation track best `0.915030`; do not generate test metrics.
  - result: over-budget diagnostic `autoresearch_yoloctm_nodistill_ema_tau04_e30_20260528_122811`, params `10.525M`, fixed-`tau=0.4` val acc `0.98265`, macro P `0.93090`, macro R `0.91030`, macro F1 `0.919511`; no test metrics generated. This confirms more optimization helps, but it is not eligible for the final `<=10` epoch track.
- Current eligible non-distilled track best after user correction:
  - `autoresearch_yoloctm_nodistill_ema_tau04_calselect_20260527_214912`, params `10.525M`, `10` epochs, fixed-`tau=0.4` val macro F1 `0.902814`; no test metrics generated.
- Next eligible non-distilled-track candidate selected on 2026-05-28:
  - keep the `10`-epoch no-distillation `tau=0.4` recipe and change only EMA decay from `0.999` to `0.9995`.
  - rationale: over-budget diagnostics showed later EMA checkpoints can raise precision substantially; a slower EMA may move some of that smoothing benefit into the fixed 10-epoch budget without changing teachers, data, architecture, or test access.
  - screen only against the eligible non-distilled validation track best `0.902814`; do not generate test metrics.
  - result: discard `autoresearch_yoloctm_nodistill_ema9995_tau04_e10_20260528_153437`, params `10.525M`, fixed-`tau=0.4` val acc `0.97953`, macro P `0.89362`, macro R `0.91191`, macro F1 `0.902032`; no test metrics generated. Slower EMA almost matched the eligible best but lagged slightly, suggesting the 10-epoch budget needs faster optimization rather than additional averaging inertia.
- Next eligible non-distilled-track candidate selected after EMA 0.9995:
  - add a OneCycle/super-convergence learning-rate schedule for the same `10`-epoch no-distillation `tau=0.4`, EMA `0.999` recipe; start with `max_lr=0.002`.
  - rationale: the e20/e30 diagnostics and epoch-10 trend show under-convergence inside the hard 10-epoch budget. OneCycle is a paper-supported way to use a brief high-LR phase plus annealing to improve performance in a fixed short training budget without teachers, data changes, architecture changes, or test access.
  - screen only against the eligible non-distilled validation track best `0.902814`; do not generate test metrics.
  - result: discard `autoresearch_yoloctm_nodistill_onecycle_lr002_tau04_e10_20260528_165003`, params `10.525M`, fixed-`tau=0.4` val acc `0.97980`, macro P `0.90708`, macro R `0.88682`, macro F1 `0.896000`; no test metrics generated. The high-LR phase improved early convergence but hurt recall/stability, so the next OneCycle screen should lower the peak LR.
- Next eligible non-distilled-track candidate selected after OneCycle `max_lr=0.002`:
  - keep the same `10`-epoch no-distillation `tau=0.4`, EMA `0.999`, OneCycle schedule and lower only `max_lr` from `0.002` to `0.0015`.
  - rationale: the `0.002` run reached strong early validation accuracy but unstable macro F1, consistent with an overly aggressive high-LR peak. A milder peak keeps the super-convergence idea while reducing minority-class disruption.
  - screen only against the eligible non-distilled validation track best `0.902814`; do not generate test metrics.
- External generalization track requested on 2026-05-27:
  - the local workspace currently contains only `MIR-WM811K` and prepared `wm811k_cls`; no independent wafer-map dataset is present.
  - candidate external benchmark: `MixedWM38`, reported as an independent public wafer-map dataset with `38,015` maps spanning normal, eight single-defect, and twenty-nine mixed-defect patterns in Micromachines 2024 (`https://www.mdpi.com/2072-666X/15/7/836`) and used alongside WM811K in WMDiff (`https://www.sciencedirect.com/science/article/pii/S095741742403001X`).
  - do not treat MixedWM38 as an interchangeable WM811K test split: its mixed-pattern label space needs a declared mapping or multi-label external-evaluation protocol before any performance claim.

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
9. For iterative model search after 2026-05-27, use validation-only screening with predeclared calibration settings; do not generate or consume test metrics until a milestone candidate is promoted.
