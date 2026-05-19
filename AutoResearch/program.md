# AutoResearch Program

You are running WM-811K classification research with a fixed compute budget.

## Mission

Your job is to improve the tradeoff between performance and size for the CTM + YOLO classifier.
Primary metrics are `accuracy`, `precision`, `recall`, and `F1`.
Secondary metric is parameter count.

## Constraints

- Keep the dataset split fixed for the whole campaign.
- Keep the epoch budget fixed for the whole campaign.
- Keep the image size fixed for the whole campaign.
- Do not add a large hyperparameter search space.
- Prefer structural changes over tuning.
- The workspace is mounted on a network drive, so the actual execution environment may be on a remote development host.
- When remote access is needed, use `ssh du@10.129.136.178` on host `10.129.136.178` with username `du`.

## Allowed edit surface

- `scripts/train_wm811k_yoloctm.py`
- `scripts/run_wm811k_pipeline.py` only if logging or workflow wiring needs it
- `configs/wm811k_cls.yaml` or `AutoResearch/configs/wm811k_autoresearch.yaml`

## Research style

- Start from a clean baseline and log it.
- Make one meaningful structural change at a time.
- Prefer parameter sharing, bottlenecks, and low-rank fusion.
- Prefer deeper CTM-YOLO coupling over bolting a head on top.
- Prefer fewer knobs, not more knobs.

## Good directions

- Shared projections inside CTM blocks
- Reusing the same CTM transition across steps
- Token pooling that preserves wafer geometry
- Channel-efficient feature reduction before CTM
- Lightweight gating between backbone tokens and recurrent state
- Removing redundant layers or projections

## Bad directions

- Large sweeps of learning rates or dropout values
- Many new hyperparameters
- Changes that only help because they increase model size
- Changes that make the pipeline harder to read or reproduce

## Experiment loop

1. Read the current log and identify the best prior run.
2. Propose one architecture change.
3. Run the same fixed-budget training protocol.
4. Evaluate on validation and test.
5. Log the result in `AutoResearch/results.tsv`.
6. Keep the change if it improves metrics or gives a clearly better size/performance tradeoff.

## Logging rule

After every run, append one row to `AutoResearch/results.tsv` and save a JSON summary in `AutoResearch/logs/`.
Each log should capture:

- run directory
- status (`keep`, `discard`, or `crash`)
- model summary
- parameter count
- validation metrics
- test metrics
- short description of the idea
