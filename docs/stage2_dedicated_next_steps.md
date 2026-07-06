# Stage 2 Dedicated Next Steps

Stage 2 now has two different levels of support in the repo:

## 1. Seed image baseline

Current usable seed labels:

- `get_put`
- `sew`

Scripts:

- `scripts/prepare_activity_dataset.py`
- `scripts/train_stage2_activity_mobilenet.py`
- `scripts/predict_stage2_activity.py`

## 2. Production reviewed-label path

Production target labels:

- `idle`
- `get`
- `put`
- `sew`

Scripts:

- `scripts/build_activity_review_queue.py`
- `scripts/train_stage2_from_csv.py`
- `scripts/predict_stage2_activity.py`
- `scripts/compute_activity_cycle_metrics.py`

## Recommended order

1. Generate a present-frame activity review queue from Stage 1
2. Fill `final_label` with `idle/get/put/sew`
3. Mark reviewed rows as `review_status=done`
4. Train the reviewed Stage 2 model
5. Predict timeline CSV with smoothing
6. Compute cycle and NPT metrics

## Commands

### Build queue from Stage 1 present frames

```bash
source rmg/bin/activate
python scripts/build_activity_review_queue.py \
  --manifest datasets/interim/roi_crops/manifest.csv \
  --presence-predictions "artifacts/stage1/eval/*.csv" \
  --present-threshold 0.5 \
  --sample-every-present 5 \
  --output datasets/processed/stage2/manifests/activity_review_queue.csv
```

### Train reviewed 4-class Stage 2 model

```bash
source rmg/bin/activate
python scripts/train_stage2_from_csv.py \
  --review-csv datasets/processed/stage2/manifests/activity_review_queue.csv \
  --split-mode video_station \
  --device cuda \
  --output-dir artifacts/stage2/models/mobilenet_reviewed
```

### Predict timeline on reviewed queue or any manifest

```bash
source rmg/bin/activate
python scripts/predict_stage2_activity.py \
  --checkpoint artifacts/stage2/models/mobilenet_reviewed/best.pt \
  --manifest datasets/processed/stage2/manifests/activity_review_queue.csv \
  --smoothing-window 7 \
  --device cuda \
  --output artifacts/stage2/eval/activity_predictions.csv
```

### Compute NPT and cycle metrics

```bash
source rmg/bin/activate
python scripts/compute_activity_cycle_metrics.py \
  --predictions artifacts/stage2/eval/activity_predictions.csv \
  --output artifacts/stage2/eval/activity_cycle_metrics.json
```
