# Activity Stage 2 Pipeline

This document defines the next-step pipeline for classifying what a worker is
doing after Stage 1 confirms the worker is present.

## Current activity scope

For now, the activity dataset supports only two classes:

- `sew`
- `get_put`

These labels come from the `Activity model` folder in `Work 2`.

## Dataset reality

The current activity image dataset is very small:

- about `50` labeled activity crops total
- mixed across multiple workstation ids
- image-level labels only, not full temporal sequence labels

So the correct first approach is not a large temporal action-recognition model.
The correct first approach is a staged image-first pipeline, then optional
temporal smoothing.

## Recommended Stage 2 design

### Stage 2 input

Stage 2 should run only when Stage 1 says `present`.

Input should be:

1. workstation ROI crop
2. machine mask applied if useful
3. optional tighter hand/work-area crop later

### Stage 2 output

For now:

- `sew`
- `get_put`
- optional `uncertain`

## Practical pipeline

```text
Raw frame
  -> Stage 1 station ROI
  -> Stage 1 presence classifier
  -> if present:
       Stage 2 activity crop
       -> image classifier
       -> temporal smoothing over short window
       -> final activity label
```

## Why image-first is better here

Because the current activity dataset is tiny, training an LSTM, ST-GCN, or
transformer on sequences right now would be unstable and easy to overfit.

Instead:

1. train an image classifier on `sew` vs `get_put`
2. run it on sampled frames from videos
3. smooth predictions across time
4. later, if enough labeled clips are collected, upgrade to a sequence model

## Recommended model

Use the same family as Stage 1 for simplicity:

- `MobileNetV3-Small` or `EfficientNet-B0`

Why:

- fast
- good for small data
- easy to run on edge hardware
- easy to fine-tune

## Activity-specific feature logic

For these two classes, the key distinction is likely not full-body pose.
It is hand/work-area interaction pattern:

- `sew`: head/body close to machine, sustained working posture
- `get_put`: reaching to place or retrieve fabric, more transitional arm motion

So the right near-term strategy is:

1. station ROI crop
2. optionally crop the machine-side work zone
3. classify frame-level activity
4. smooth over time

## Training protocol

### Phase A: seed activity classifier

Train on the labeled `Activity model` images only.

### Phase B: domain adaptation

Use the real videos:

1. run Stage 1 to keep only present frames
2. extract ROI crops for those frames
3. review and label a subset as `sew` or `get_put`
4. fine-tune the activity classifier on reviewed video-domain crops

### Phase C: temporal stabilization

For inference:

- sample every `N` frames
- predict activity per frame
- majority vote over a short window, such as `5` to `15` predictions

## Validation rule

Do not use random frame split across near-duplicate frames from the same short
video region.

Prefer:

- split by source video
- or split by time block
- or split by workstation and capture session

## What not to do yet

- do not start with pose-only modeling
- do not start with ST-GCN first
- do not trust pseudo-labels as activity ground truth
- do not mix full-frame and ROI-crop distributions without tracking provenance

## Best immediate next implementation

1. prepare a clean `datasets/raw/activity/{sew,get_put}` layout
2. train a two-class image classifier
3. create a video ROI activity review queue
4. fine-tune on reviewed video crops
5. add temporal smoothing at inference

## Future upgrade path

Once enough labeled clips exist:

1. build short clips around present frames
2. add optical flow or motion features
3. test 3D CNN / TimeSformer / TSM / LSTM over frame embeddings
4. optionally fuse object cues and temporal cues
