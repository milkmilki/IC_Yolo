# Initial Experiment Results

**Date**: 2026-05-22
**Plan**: refine-logs/EXPERIMENT_PLAN.md

## Results by Milestone

### M1: Low-rank CTM feature adapter with prior calibration

- Run: `runs/classify/autoresearch_yoloctm_ctmadapter_lowrank_priorcal_20260522_155358`
- Status: discard
- Test macro F1: 0.8779588275
- Notes: reduced params only slightly (10.525M -> 10.496M) and lost recall/F1 versus the prior calibrated CTM adapter.

### Follow-up: Fixed log-probability ensemble

- Run: `runs/classify/autoresearch_yoloctm_logprob_ensemble_20260522_170000`
- Status: keep for performance
- Config: YOLO26m + full CTM adapter + low-rank CTM adapter log-prob ensemble, weights=(0.6, 0.2, 0.2), prior_logit_tau=0.025
- Validation metrics: acc 0.9813444342, macro P/R/F1 0.9237807676 / 0.9017953689 / 0.9109065665
- Test metrics: acc 0.9822289041, macro P/R/F1 0.9220839838 / 0.9044817596 / 0.9118345879
- Parameter footprint: 32.655M summed inference parameters
- Notes: strongest logged macro-F1 so far and above the prior best 0.8983463307, but below the target macro-F1 0.95 and not parameter-efficient.

## Summary

- Completed fixed-protocol M1 and one local post-hoc ensemble follow-up.
- Main result is positive for macro-F1/accuracy, negative for size.
- AutoResearch bookkeeping is present for both latest runs:
  - `AutoResearch/results.tsv`
  - `AutoResearch/logs/20260522_163017_autoresearch_yoloctm_ctmadapter_lowrank_priorcal_20260522_155358.json`
  - `AutoResearch/logs/20260522_203301_autoresearch_yoloctm_logprob_ensemble_20260522_170000.json`

## Next Step

Target macro-F1=0.95 is not met. Next iteration should distill or compress the ensemble signal into a single model, or run a reviewer-approved structural change focused on Loc/Edge-Loc/Scratch without increasing inference footprint.
