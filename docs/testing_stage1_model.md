# Testing The Stage 1 Model

## Check the trained model artifacts

Saved files:

- `artifacts/stage1/models/mobilenet_seed/best.pt`
- `artifacts/stage1/models/mobilenet_seed/history.json`
- `artifacts/stage1/models/mobilenet_seed/split_manifest.csv`

## Test on one labeled seed image

```bash
python3 scripts/predict_stage1_mobilenet.py \
  --image datasets/raw/person/present/07e79729e47b97ee6389a74a39316073d004c6aaeccafdde4fa1c75451691795.jpg \
  --device cuda
```

## Test on extracted ROI crops from one station folder

```bash
python3 scripts/predict_stage1_mobilenet.py \
  --image-dir datasets/interim/roi_crops/cam_39_26_nov_f1_24/station_1 \
  --device cuda
```

## Test on the ROI crop manifest

```bash
python3 scripts/predict_stage1_mobilenet.py \
  --manifest datasets/interim/roi_crops/manifest.csv \
  --device cuda
```

## Test directly on one video station

```bash
python3 scripts/predict_stage1_mobilenet.py \
  --video-id cam_39_26_nov_f1_24 \
  --station-id 1 \
  --max-frames 30 \
  --device cuda
```

This uses the saved ROI and machine mask, extracts sampled masked crops from the
video, and runs the classifier on them.

## Important scoring note

The checkpoint class order is:

- `absent`
- `present`

For Stage 1 review, the most important score is `present_confidence`.
The prediction script now always writes:

- `present_confidence`
- `absent_confidence`

and uses `present_confidence >= --present-threshold` to decide the final label.
