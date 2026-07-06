# Person Model Pipeline

This repo is currently set up for **Stage 1 only**: worker presence vs absence.

## What we have now

- `100` unique labeled person images
- `50` `present`
- `50` `absent`
- `5` unique raw videos
- `0` content duplicates across the selected person/video assets

## Clean dataset layout

```text
datasets/
  raw/
    person/
      absent/
      present/
    videos/
  interim/
  processed/
  manifests/
```

## Recommended next pipeline

### 1. Use the labeled person images as the Stage 1 bootstrap set

These images are already the fastest path to an initial MobileNetV3 presence model.
We should not wait for the activity pipeline before training the binary classifier.

### 2. Extract validation/training frames from videos only after ROI is defined

The instructions file is right about the main risk: if we extract full frames before
station ROI and machine masking, the model can learn the wrong cues.

For that reason, the correct order is:

1. Define ROI polygon for each video
2. Define machine mask for each video
3. Extract ROI crops from video
4. Label or auto-label those crops
5. Train with leave-one-video-out evaluation

### 3. Keep image-level and video-derived data separate

Use:

- `datasets/raw/person` for manually labeled seed images
- `datasets/interim/roi_crops` for extracted station crops from video
- `datasets/processed/stage1` for train-ready splits and manifests

This separation will help us avoid leakage and keep the provenance of each sample clear.

## My opinion on the plan

The plan in `instructions.md` is strong, but for the current data I would slightly
tighten the execution order:

1. Finish Stage 1 end to end before touching activity classification
2. Build ROI and machine-mask tooling immediately, because that is the biggest source of failure
3. Train a small baseline on the 100 labeled images first, then expand using video crops
4. Treat the 5 videos as the backbone for evaluation, not just extra data

That gives us a fast baseline and a cleaner path to the real production-style pipeline.

## Stage 1 Fix Direction

The corrected Stage 1 protocol is:

1. Station-only ROI as the main annotation unit
2. Fixed machine masking only where needed
3. Seed training on `datasets/raw/person`
4. Clean adaptation on reviewed video-domain ROI crops
5. Group-aware validation split such as `video_station`, not random frame split

## Files added for the next step

- `configs/video_sources.json`: human-readable video inventory with original source names
- `configs/roi_annotations.template.json`: multi-workstation ROI and machine-mask template to fill in
- `scripts/export_annotation_frames.py`: exports one reference frame per video for annotation
- `scripts/extract_stage1_roi_crops.py`: converts annotated videos into ROI-masked crops for each workstation
- `scripts/build_stage1_image_manifest.py`: builds a manifest from the current labeled person images
- `scripts/build_station_seed_manifest.py`: keeps label plus workstation id from the original dataset structure
- `scripts/build_station_contact_sheets.py`: makes workstation-wise visual sheets from the seed images
- `scripts/annotate_station_rois.py`: click-based polygon annotation tool
- `scripts/generate_video_overview_boards.py`: writes station-labeled full-frame guides
- `scripts/preview_roi_masks.py`: visual QA for saved ROIs and machine masks
- `scripts/run_mask_presence_baseline.py`: no-training occupancy-change baseline
- `scripts/train_stage1_mobilenet.py`: first MobileNetV3 Stage 1 trainer

## Immediate workflow

1. Export annotation frames for all five videos.
2. Fill workstation-specific `station_roi_polygon` and `machine_mask_polygons` for each video.
3. Run ROI crop extraction for all annotated workstations.
4. Build the seed image manifest and start Stage 1 training.

## Important update

The person image dataset originally suggested at least `10` workstation-specific ROIs:

- each of `present` and `absent` contains folders `1` through `10`
- each workstation folder has `5` examples per class

So the Stage 1 pipeline should treat each video as a multi-workstation scene, not as a
single station ROI. The exact count can vary by camera and should be driven by the
real frame layout, not by a hard-coded limit.
