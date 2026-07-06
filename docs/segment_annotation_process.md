# Segment Annotation Process

## Goal

Build a method-study style segment sheet for the two clearest stations:

- `station 4`
- `station 3`

The target output should resemble the expert single-station report:

- `Serial`
- `Start time`
- `End time`
- `Duration`
- `Content`

## Step 1: Build a focused transition queue

```bash
source rmg/bin/activate
python scripts/build_focus_station_queue.py \
  --input-csv datasets/processed/stage2/manifests/transition_review_queue.csv \
  --output-csv datasets/processed/stage2/manifests/segment_focus_station34_queue.csv \
  --stations 3,4
```

## Step 2: Review the focused queue in the UI

```bash
source rmg/bin/activate
python scripts/run_stage2_labeling_ui.py \
  --queue-csv datasets/processed/stage2/manifests/segment_focus_station34_queue.csv
```

Open:

```text
http://127.0.0.1:8010
```

Use the queue to inspect transition anchors with `prev/current/next` context.

## Optional: Qwen2.5-VL segment suggestions

If you install the Qwen2.5-VL runtime stack locally, you can use:

- the expert single-station report as a style guide
- a few reference frames from `Single Station.mp4`

Example command:

```bash
source rmg/bin/activate
python scripts/qwen_segment_suggest.py \
  --queue-csv datasets/processed/stage2/manifests/segment_focus_station34_queue.csv \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --reference-report-text docs/single_station_report_reference.txt \
  --reference-image-glob '/tmp/single_station_ref_*.jpg' \
  --reference-image-limit 4 \
  --limit 12
```

This fills:

- `qwen_segment_decision`
- `qwen_action_label`
- `qwen_segment_text`
- `qwen_confidence`
- `qwen_reason`

## Step 3: Initialize the method-study sheet

```bash
source rmg/bin/activate
python scripts/init_method_study_sheet.py \
  --queue-csv datasets/processed/stage2/manifests/segment_focus_station34_queue.csv \
  --output-csv datasets/processed/stage2/manifests/station34_method_study_sheet.csv
```

## Step 4: Fill segment rows manually

For each real action segment, fill:

- `start_time_sec`
- `end_time_sec`
- `duration_sec`
- `action_label`
- `content`

Optional supporting fields:

- `left_hand_zone`
- `right_hand_zone`
- `object_context`
- `notes`

## Recommended action labels

- `align_fabric`
- `place_on_bed`
- `feed_to_needle`
- `sew`
- `release_fabric`
- `reach_get_zone`
- `pull_fabric`
- `move_to_dispatch`
- `pick_accessory`
- `attach_accessory`
- `reposition_fabric`

## Annotation rule

Do not create a new segment for every frame.
Create a new segment only when the dominant hand/work-area interaction changes in a way that would deserve a new row in the expert report.

## Current recommended stations

- primary: `station 4`
- secondary: `station 3`
