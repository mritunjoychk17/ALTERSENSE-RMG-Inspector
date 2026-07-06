# ROI Crop Labeling Workflow

Build a lightweight CSV queue from extracted ROI crops:

```bash
python3 scripts/build_roi_crop_label_queue.py --video-id cam_39_26_nov_f1_24
```

This creates:

- `datasets/processed/stage1/manifests/roi_crop_label_queue.csv`

Recommended labeling rule for Stage 1:

- `present`: any visible body part is inside the workstation ROI
- `absent`: workstation ROI does not contain the worker
- `notes`: use for ambiguous leaning or transition frames

Suggested review flow:

1. Open the crop images listed in the CSV.
2. Fill `label` with `present` or `absent`.
3. Set `review_status` to `done` after checking.
4. Keep ambiguous frames but mark them in `notes`.

## Faster review with model predictions

After running model inference on the crop manifest:

```bash
python3 scripts/predict_stage1_mobilenet.py \
  --manifest datasets/interim/roi_crops/manifest.csv \
  --output artifacts/stage1/eval/roi_crop_predictions.csv
```

Merge those predictions into a review queue:

```bash
python3 scripts/merge_predictions_into_label_queue.py
```

This creates:

- `datasets/processed/stage1/manifests/roi_crop_review_queue.csv`

Use:

- `model_prediction` and `model_confidence` as the starting suggestion
- `present_confidence` as the main Stage 1 score to inspect
- `final_label` as the corrected label after review

## Optional bootstrap for easy cases

You can pre-fill obvious cases before manual review:

```bash
python3 scripts/bootstrap_review_labels.py
```

This marks very confident rows as `bootstrapped`, but they should still be checked.

## Retrain on reviewed ROI crops

After filling `final_label` and setting `review_status=done` for reviewed rows:

```bash
python3 scripts/train_stage1_from_csv.py \
  --review-csv datasets/processed/stage1/manifests/domain_review_queue.csv \
  --accepted-status done \
  --split-mode video_station \
  --include-seed-images \
  --device cuda
```

Important:

- prefer real reviewed rows over pseudo-labels
- prefer `domain_review_queue.csv` or another human-verified CSV
- avoid random split when training on ROI crops from the same video/session
