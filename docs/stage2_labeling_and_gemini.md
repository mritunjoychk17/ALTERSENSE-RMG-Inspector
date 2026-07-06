# Stage 2 Labeling And Gemini Assist

## Hand-first labeling rule

For Stage 2, label decisions should emphasize:

1. hand position relative to the work area
2. arm reach direction
3. fabric interaction
4. short-term work motion

Use full-body posture only as secondary context.
Use previous/current/next sampled frames together whenever possible so hand
motion direction is easier to judge.

## Lightweight local labeling UI

Run:

```bash
source rmg/bin/activate
python scripts/run_stage2_labeling_ui.py \
  --queue-csv datasets/processed/stage2/manifests/activity_review_queue.csv
```

Open:

```text
http://127.0.0.1:8010
```

Keyboard shortcuts:

- `1` -> `idle`
- `2` -> `get`
- `3` -> `put`
- `4` -> `sew`
- `5` -> `uncertain`

## Pose-assisted suggestions before manual review

This is the recommended next step for faster Stage 2 labeling.

The local suggestion script fills:

- `pose_label`
- `pose_confidence`
- `pose_reason`

It works in two modes:

1. `motion` fallback:
   usable immediately with the current environment
2. `ultralytics`:
   uses YOLO pose if `ultralytics` is installed and a pose model is available

Run the fallback now:

```bash
source rmg/bin/activate
python scripts/suggest_stage2_pose_labels.py \
  --queue-csv datasets/processed/stage2/manifests/activity_review_queue.csv \
  --backend motion
```

If YOLO pose is installed later, run:

```bash
source rmg/bin/activate
python scripts/suggest_stage2_pose_labels.py \
  --queue-csv datasets/processed/stage2/manifests/activity_review_queue.csv \
  --backend ultralytics \
  --pose-model yolo11n-pose.pt
```

Then open the labeling UI and verify or correct the suggested labels manually.

## Gemini-assisted labeling

Set your API key with an environment variable or `.env` file.

Recommended `.env` file:

```bash
cp .env.example .env
```

Then edit `.env` and set:

```bash
GOOGLE_API_KEY="YOUR_API_KEY"
```

Or export it directly:

```bash
export GOOGLE_API_KEY="YOUR_API_KEY"
```

Then run:

```bash
source rmg/bin/activate
python scripts/gemini_label_stage2_queue.py \
  --queue-csv datasets/processed/stage2/manifests/activity_review_queue.csv \
  --model gemini-3.1-flash-lite \
  --limit 20
```

This fills:

- `gemini_label`
- `gemini_confidence`
- `gemini_reason`

You should still review and finalize:

- `final_label`
- `review_status`

## Suggested workflow

1. build Stage 2 queue from Stage 1 present crops
2. run pose-assisted suggestions
3. optionally run Gemini-assisted suggestions
4. open the local labeling UI
5. confirm or correct labels as `idle/get/put/sew`
6. train reviewed Stage 2 model
