
# RMG Worker Activity Recognition — Project Pipeline & Integration Plan

**Project:** Garments Worker Presence & Activity Detection  
**Dataset:** 4 videos × ~4 min + labeled images (ceiling-mounted camera)  
**Goal:** Detect worker presence/absence at sewing machines; classify activity type  
**Regime:** Small-data (< 50 labeled images + ~23K extractable video frames)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Camera & Environment Setup](#2-camera--environment-setup)
3. [Stage 1 — Presence / Absence Detection](#3-stage-1--presence--absence-detection)
4. [Stage 2 — Activity Classification](#4-stage-2--activity-classification)
5. [Integration Architecture](#5-integration-architecture)
6. [Data Pipeline](#6-data-pipeline)
7. [Model Training Plan](#7-model-training-plan)
8. [Evaluation Protocol](#8-evaluation-protocol)
9. [Demo & Visualization Plan](#9-demo--visualization-plan)
10. [Critical Watchpoints](#10-critical-watchpoints)
11. [Tech Stack Summary](#11-tech-stack-summary)
12. [Recommended Timeline](#12-recommended-timeline)

---

## 1. Project Overview

This system detects whether a garments worker is present at their sewing workstation and, if present, identifies what type of work they are doing. The pipeline is split into two sequential stages:

- **Stage 1** — Binary presence/absence classification using a fine-tuned MobileNetV3 on ceiling-view chair ROI crops
- **Stage 2** — Activity classification (sewing, cutting, ironing, QC, folding, idle, machine setup) using pose estimation + object context

The entire strategy is built on **transfer learning and foundation models** because the dataset is too small to train from scratch. This is not a limitation — it mirrors how modern industrial computer vision systems are deployed in 2024–2025.

### Activity Labels

| Label | Visual cue (ceiling view) |
|---|---|
| Absent | Empty chair, no head blob |
| Idle | Head present, no arm motion |
| Sewing | Head down, rhythmic arm near machine |
| Cutting | Wide lateral arm sweep |
| Ironing | Back-and-forth arm motion, iron nearby |
| QC Inspection | Arms raised, fabric held up |
| Folding / Packing | Symmetric two-arm motion |
| Machine Setup | Close-to-machine, slow deliberate hands |

---

## 2. Camera & Environment Setup

### Current Setup
- Ceiling-mounted camera, positioned directly above the workstation
- Provides a **top-down view**: head and shoulders visible, full body not visible
- Static camera — does not pan or tilt during a shift

### What This Means for Modeling

The ceiling angle is actually an advantage for presence detection. The "present" signature is a consistent dark oval (head/hair) appearing in the chair zone. However, it creates challenges for pose-based activity classification since standard pose models (trained on front/side views) underperform on top-down angles.

### Setup Checklist

- [ ] Record the camera mounting height (in cm) — needed to normalize head blob size
- [ ] Verify the camera does not shift between recording sessions (check for pixel drift across your 4 videos)
- [ ] Note lighting conditions per video — factory lighting uniformity affects background subtraction
- [ ] Confirm one camera per workstation, or one camera covering multiple stations
- [ ] Define and save workstation ROI polygons (chair bounding area) per camera

---

## 3. Stage 1 — Presence / Absence Detection

### Recommended Approach: Binary Classifier on Chair ROI Crop

**Why not YOLO person detection?** YOLO was trained on front/side-view human images. From a ceiling camera, a person appears as a head blob — YOLO's person class will miss this consistently. A custom binary classifier trained on your own ceiling-view crops will significantly outperform it.

### Pipeline

```
Raw video frame
      │
      ▼
Crop chair ROI polygon
(pre-defined coordinates, static per workstation)
      │
      ▼
Apply sewing machine mask
(zero out machine body pixels inside ROI)
      │
      ▼
Resize crop to 224×224
      │
      ▼
MobileNetV3-Small (fine-tuned)
      │
      ├── confidence < threshold → ABSENT
      │
      └── confidence ≥ threshold → PRESENT → pass to Stage 2
      │
      ▼
Temporal smoothing
(majority vote over 15-frame window)
      │
      ▼
Structured output: { frame_id, timestamp, label, confidence }
```

### Chair ROI Definition

Define a polygon in image coordinates that covers the chair seat and back, excluding the sewing machine body. This is done once per workstation from a representative empty frame.

```python
# Example: save ROI coordinates to a config file
roi_coords = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
```

Use either manual annotation (10 minutes, any image editor) or automated detection with Grounding DINO (`"chair"` prompt) + SAM for a pixel mask.

### Sewing Machine Mask

Create a binary mask image (same resolution as the video frame) where the machine body region is set to 0. Apply before any classifier:

```python
worker_frame = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(machine_mask))
```

This ensures the model never "sees" the machine, preventing false positives when the worker leans over it.

### Confidence Threshold

Start at **0.75**. Frames below this threshold are marked as "uncertain" and resolved by the temporal smoothing window. Tune this on your validation split — if absent frames are being missed, lower to 0.65.

### Temporal Smoothing

Apply a majority vote over a rolling 15-frame window (~0.5 seconds at 30 fps). This suppresses single-frame classification errors caused by motion blur, lighting flicker, or partial occlusion.

---

## 4. Stage 2 — Activity Classification

Stage 2 only runs when Stage 1 confirms the worker is PRESENT.

### Recommended Approach: Pose Estimation + Object Context

```
PRESENT frame (Stage 1 confirmed)
      │
      ▼
YOLO11-Pose
(17 keypoints — top-down view)
      │
      ▼
Extract keypoint sequence
(sliding window of 30 frames)
      │
      ▼
ST-GCN / SkateFormer skeleton classifier
      │
      ▼
Object context stream (parallel)
YOLO head detecting: sewing machine / scissors / iron
      │
      ▼
Fuse skeleton + object context
      │
      ▼
Activity label + confidence
```

### Why Object Context Matters

From a ceiling view, "arm sweeping left-right" is ambiguous between cutting and ironing. The presence of scissors in the frame resolves it to cutting; the presence of an iron resolves it to ironing. Always fuse object detections into the final classification decision.

### Note on Ceiling-View Pose Estimation

Standard YOLO-Pose keypoint models underperform from directly above. After extracting keypoints, the most reliable joints for ceiling-view activity recognition are:

- Wrist positions (hand location relative to machine)
- Elbow positions (arm extension and direction)
- Head position relative to the workstation center
- Shoulder width (indicates body rotation/leaning)

Hip and leg keypoints will be unreliable or invisible — do not include them in your feature vector.

---

## 5. Integration Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Video / Live Stream                   │
└────────────────────────┬────────────────────────────────┘
                         │
              Frame extraction (cv2)
                         │
         ┌───────────────▼───────────────┐
         │        ROI + Mask Layer       │
         │  (static per workstation)     │
         └───────────────┬───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │    Stage 1: MobileNetV3       │
         │    Presence / Absence         │
         └──────┬────────────────┬───────┘
                │                │
            ABSENT            PRESENT
                │                │
         Log + alert      Stage 2: Pose +
                         Activity Classifier
                                 │
                    ┌────────────▼────────────┐
                    │   Output Logger         │
                    │  { frame, time, label,  │
                    │    confidence }         │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   FastAPI Backend        │
                    │   SQLite / PostgreSQL    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   React Dashboard       │
                    │   Live labels + charts  │
                    └─────────────────────────┘
```

### Component Responsibilities

| Component | Role | Technology |
|---|---|---|
| Frame extractor | Pull frames at configurable fps | OpenCV |
| ROI / Mask layer | Crop + blank machine area | OpenCV, NumPy |
| Stage 1 classifier | Present / Absent per frame | PyTorch, MobileNetV3 |
| Stage 2 classifier | Activity label per clip | PyTorch, ST-GCN |
| Object detector | Sewing machine / tools context | YOLOv11 |
| Temporal smoother | Stabilize label sequences | Python deque |
| Output logger | Persist structured events | SQLite / PostgreSQL |
| API layer | Serve labels to dashboard | FastAPI |
| Dashboard | Real-time visualization | React + Recharts |

---

## 6. Data Pipeline

### Step 1 — Extract Frames

```
4 videos × 4 min × 24 fps = ~23,040 raw frames
```

Extract every frame with OpenCV. For training, sample at 1 fps to avoid near-duplicate frames (~960 training candidates across 4 videos).

### Step 2 — Auto-label with Foundation Models

Use Grounding DINO + BLIP-2 to generate coarse labels on extracted frames:

- DINO prompts: `"sewing machine"`, `"garment worker"`, `"scissors"`, `"iron"`
- BLIP-2: generate activity descriptions on cropped worker zones → coarse labels

This bootstraps your dataset. Manually verify and correct ~200–300 samples.

### Step 3 — Augmentation

For Stage 1 (binary classifier, starting from <50 images):

| Augmentation | Parameters |
|---|---|
| Horizontal flip | p=0.5 |
| 90° rotation | all 4 directions |
| Brightness jitter | ±30% |
| Contrast jitter | ±20% |
| Random crop + resize | 80–100% of crop, back to 224×224 |
| Gaussian noise | std=0.02 |

Target: ~500 samples per class (Present / Absent) after augmentation.

For Stage 2 (activity classification):

Target: 800–1,000 samples per activity class after augmentation. This will require frame extraction from videos as the primary source.

### Step 4 — Split Strategy

With only 4 videos, use **leave-one-video-out** cross-validation:
- Train on 3 videos, test on 1
- Repeat for all 4 combinations
- Report mean and standard deviation of accuracy across folds

Do **not** do a random train/test split — frames from the same video are temporally correlated and will leak between splits.

---

## 7. Model Training Plan

### Stage 1: MobileNetV3-Small

```python
model = torchvision.models.mobilenet_v3_small(weights="DEFAULT")

# Freeze all backbone layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier head
model.classifier[3] = torch.nn.Linear(1024, 2)

# Training config
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = torch.nn.CrossEntropyLoss()
epochs = 30
```

**Expected training time:** Under 5 minutes on a laptop CPU, under 1 minute on a GPU.

### Stage 2: ST-GCN on YOLO11-Pose keypoints

- Extract 17-keypoint sequences from 30-frame sliding windows
- Drop hip/leg keypoints (unreliable from ceiling view) — use 11 upper-body keypoints
- Feed sequences into ST-GCN for temporal activity classification
- Alternatively: flatten keypoint sequences and train a simple LSTM or MLP baseline first

### Baseline Comparisons to Run

1. Zero-shot CLIP with prompts like `"worker sewing at machine"` vs `"empty chair"`
2. Background subtraction (MOG2) pixel count threshold — simplest possible baseline
3. MobileNetV3 fine-tuned (recommended approach)
4. YOLOv11 person detection (expected to underperform on ceiling view — useful as ablation)

---

## 8. Evaluation Protocol

### Metrics

| Metric | Why it matters |
|---|---|
| Per-class Precision / Recall | Catch asymmetric errors (missing absent vs false absent) |
| F1 score | Overall classifier quality |
| False absent rate | How often a present worker is wrongly flagged absent |
| Absent detection rate | How often true absences are caught |
| Transition accuracy | Correct label at sit-down / stand-up moments |

### Confusion Matrix

Always generate a confusion matrix. For Stage 1, the critical cell is **Present predicted as Absent** — this is the false alarm that would trigger a wrong management alert.

### Threshold Tuning

Plot Precision-Recall curves for Stage 1. Pick the threshold that gives you the false alarm rate acceptable for your use case. In factory floor monitoring, a false alarm (worker flagged absent when present) is typically more damaging to trust in the system than a missed absence.

---

## 9. Demo & Visualization Plan

### Video Overlay Demo

Side-by-side display with three panels:

1. Raw ceiling frame with ROI polygon drawn in white
2. Cropped + masked worker zone (what the classifier sees)
3. Colored label banner: green for PRESENT, red for ABSENT, amber for uncertain

Add confidence score as percentage text, and a timestamp bar.

### Timeline Bar

Below the video, a horizontal bar that fills green/red as the video plays. Makes absent gaps visually obvious at a glance — useful for presentations and for floor managers reviewing shift footage.

### Live Dashboard (Product Track)

- Grid of N workstations, each showing current label + color
- Per-worker idle % for the current shift
- Alert panel listing recent absent events with timestamps
- Skeleton-only privacy mode (no raw RGB stored — only joint coordinates)

---

## 10. Critical Watchpoints

These are the points where the project most commonly fails. Read carefully.

### Data Quality

- **Temporal correlation:** Frames from the same video second look almost identical. Always use leave-one-video-out splits, never random splits. Random splits will give you inflated accuracy that collapses on new footage.
- **Label imbalance:** If your dataset has many more "present" frames than "absent," the model will learn to always predict present. Check class balance before training and apply `WeightedRandomSampler` or class weights in the loss if imbalanced.
- **Transition frames:** Frames where the worker is mid-sit or mid-stand are genuinely ambiguous. Label these consistently (e.g., always label as "present" if any body part is in the chair zone) and consider excluding them from evaluation metrics.

### Camera & Environment

- **Lighting changes:** Factory lighting often shifts across shifts (morning vs afternoon, indoor vs mixed natural light). If your 4 videos span different times of day, ensure your training set includes samples from all lighting conditions.
- **Camera drift:** Even "fixed" ceiling cameras can shift slightly due to vibration or physical contact. Check that your ROI polygon still covers the chair correctly across all 4 videos. Add a periodic ROI recalibration step in production.
- **Multiple workers in frame:** If the camera FOV is wide enough to catch adjacent workstations, your ROI crop is critical — without it, a worker at a neighboring station will trigger a false "present" classification.
- **Headwear:** Some garments workers wear caps or hair coverings. From a ceiling view, this changes the visual appearance of the head blob. Make sure your training images include workers with and without headwear.

### Model Behavior

- **The mask must be applied before inference, not after.** If the sewing machine body is visible to MobileNetV3, it will learn machine-texture as a "present" feature — leading to false positives on empty-chair-but-machine-visible frames.
- **Confidence threshold calibration:** The default 0.75 threshold may need tuning per workstation. A workstation with unusual lighting or a non-standard chair shape may need a different threshold. Calibrate per camera if deploying across multiple workstations.
- **Temporal smoother window size:** 15 frames (0.5s at 30fps) is a starting point. If your workers frequently lean back briefly (e.g., to inspect finished garments), a shorter window avoids false "absent" labels. If your camera runs at 24fps, adjust accordingly.
- **Stage 2 depends on Stage 1:** If Stage 1 has high false-negative rate (missing absences), Stage 2 will run on frames where nobody is actually there. Always evaluate Stage 1 independently before integrating Stage 2.

### Deployment

- **Model versioning:** Save every model checkpoint with the training data split it was trained on. If you retrain with new data, the old checkpoint and the new one are not interchangeable.
- **Frame rate vs accuracy tradeoff:** Running Stage 1 + Stage 2 on every frame is expensive. For production, run Stage 1 on every frame (it's fast — MobileNetV3-Small runs at 100+ fps on CPU), and run Stage 2 only on confirmed "present" frames at 2–5 fps. Stage 2 operates on 30-frame windows anyway, so per-frame Stage 2 inference is redundant.
- **Clock synchronization:** If you're logging timestamps for multiple cameras across a factory floor, make sure all edge devices are NTP-synchronized. A 5-second clock drift will make per-workstation comparisons meaningless.
- **Privacy and storage:** Do not store raw RGB video frames in production logs. Store only labels, timestamps, and (optionally) skeleton keypoints. This eliminates privacy concerns and reduces storage by 99%.

### Research Track Specific

- **Novelty claim:** Your system's contribution is the first few-shot RMG-domain worker activity recognition benchmark. The closest prior datasets are InHARD (assembly line) and MECCANO (industrial how-to). Neither covers ready-made garments or ceiling-mounted cameras. Document this clearly in your paper.
- **Auto-labeling quality:** Grounding DINO + BLIP-2 coarse labels will have errors. Never use raw auto-labels as ground truth. Always manually verify a random sample of at least 50 images per class before training.
- **Ablation table:** Track the contribution of each component individually — mask layer, temporal smoother, object context stream. A paper without an ablation table will be rejected at IEEE TII or CIE.

---

## 11. Tech Stack Summary

| Layer | Tool | Notes |
|---|---|---|
| Frame extraction | OpenCV (`cv2`) | Standard, runs everywhere |
| ROI / Mask | OpenCV + NumPy | One-time setup per workstation |
| Auto-labeling | Grounding DINO + BLIP-2 | Hugging Face transformers |
| Annotation | CVAT | Free, browser-based |
| Stage 1 model | MobileNetV3-Small | PyTorch + torchvision |
| Stage 2 pose | YOLO11-Pose | Ultralytics |
| Stage 2 activity | ST-GCN or SkateFormer | PyTorch |
| Object detection | YOLOv11 | Ultralytics |
| API backend | FastAPI | Python |
| Database | SQLite (dev) / PostgreSQL + TimescaleDB (prod) | |
| Dashboard | React + Recharts | |
| Deployment | Docker Compose | Single edge server per floor |
| GPU (optional) | RTX 3060 or Jetson Orin | Handles 4–8 cameras at 30fps |

---

## 12. Recommended Timeline

| Week | Milestone |
|---|---|
| 1 | Define chair ROI + machine mask for all 4 videos. Extract all frames. |
| 2 | Augment labeled images. Train Stage 1 MobileNetV3. Evaluate leave-one-out. |
| 3 | Run Grounding DINO + BLIP-2 auto-labeling on extracted frames. Manual verification. |
| 4 | Build video overlay demo for Stage 1. Present presence/absence results. |
| 5–6 | Extract YOLO11-Pose keypoints. Build ST-GCN Stage 2 classifier. |
| 7 | Add object context stream. Fuse with skeleton classifier. |
| 8 | Temporal smoothing integration. End-to-end pipeline test on all 4 videos. |
| 9 | Evaluation: per-class metrics, confusion matrices, ablation table. |
| 10 | Dashboard build (if product track) or paper writing (if research track). |

---

*Document version: 1.0 — based on 4-video RMG ceiling-camera dataset*  
*Covers: Stage 1 presence detection, Stage 2 activity classification, integration architecture, critical deployment watchpoints*