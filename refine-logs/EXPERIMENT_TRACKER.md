# Experiment Tracker

| Milestone | Run | Status | Key metric | Notes |
| --- | --- | --- | ---: | --- |
| M1 | autoresearch_yoloctm_ctmadapter_lowrank_priorcal_20260522_155358 | discard | test macro F1 0.8779588275 | Low-rank adapter saved 0.029M params but lost recall and macro-F1. |
| Follow-up | autoresearch_yoloctm_logprob_ensemble_20260522_170000 | keep/performance | test macro F1 0.9118345879 | Fixed log-prob ensemble improved F1 but has 32.655M summed inference params. |
