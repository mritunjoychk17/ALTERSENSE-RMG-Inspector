# Activity Production Plan

## Target labels

For production, Stage 2 should move from the seed `get_put` vs `sew` classifier
to the operational label set:

- `idle`
- `get`
- `put`
- `sew`

These map to business logic as:

- `working`: `get`, `put`, `sew`
- `NPT`: `idle`

## Recommended model choice

There is no single perfect model for this whole task.

Best practical recommendation:

1. `gemini-3.5-flash` for labeling assistance and activity review support
2. local `MobileNetV3-Small` or similar classifier for final per-frame activity inference
3. state-machine logic for NPT and cycle counting

Why not rely on Gemini alone:

- Gemini video understanding samples video at `1 FPS` by default
- quick micro-actions can be missed
- repeated API inference is expensive and harder to control
- cycle counting needs deterministic event logic

## System design

```text
Uploaded video
  -> Stage 1 station presence
  -> keep present crops only
  -> Gemini-assisted review queue
  -> reviewed labels: idle / get / put / sew
  -> local Stage 2 activity classifier
  -> temporal smoothing
  -> cycle state machine
  -> dashboard metrics
```

## Current implementation pieces

- `scripts/prepare_activity_dataset.py`
- `scripts/train_stage2_activity_mobilenet.py`
- `scripts/predict_stage2_activity.py`
- `scripts/build_activity_review_queue.py`
- `scripts/compute_activity_cycle_metrics.py`

## Immediate next milestones

1. build reviewed activity queue from Stage 1 present crops
2. add Gemini-assisted labeling script for that queue
3. expand Stage 2 labels to `idle`, `get`, `put`, `sew`
4. add temporal smoothing and cycle visualization
5. build upload dashboard backend and frontend
