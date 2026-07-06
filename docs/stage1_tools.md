# Stage 1 Tools

## 1. ROI preview tool

Use this after filling workstation polygons:

```bash
python3 scripts/preview_roi_masks.py
```

This writes:

- full-frame overlays with ROI and machine-mask colors
- masked crop previews per workstation

## 2. No-training mask baseline

Use this after ROI annotation to get a fast heuristic signal:

```bash
python3 scripts/run_mask_presence_baseline.py --video-id cam_39_26_nov_f1_24 --station-id 1
```

How it works:

- frame 0 is used as the reference frame
- later frames are compared against the masked ROI
- large appearance change means `present`
- small change means `absent`

This is only a sanity-check baseline, not the final model.

## 3. MobileNetV3 training

Train the first seed-image classifier:

```bash
python3 scripts/train_stage1_mobilenet.py --device cuda
```

This uses the current `datasets/raw/person` images and writes model artifacts into
`artifacts/stage1/models/mobilenet_seed`.
