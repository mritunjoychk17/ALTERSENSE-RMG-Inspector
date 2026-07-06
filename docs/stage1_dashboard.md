# Stage 1 Local Dashboard

This dashboard is for **Stage 1 only**.

It lets you:

1. upload a video
2. choose an existing annotated ROI profile
3. run the local Stage 1 presence pipeline
4. view and download the overlay video
5. inspect a simple per-station summary

## Important constraint

The uploaded video must match one of the existing annotated camera layouts in
`configs/roi_annotations.template.json`.

This version does **not** auto-annotate new camera views.

## Run locally

```bash
source rmg/bin/activate
python scripts/run_stage1_dashboard.py
```

Then open:

```text
http://127.0.0.1:8000
```

If you want to expose it on your local network behind your firewall:

```bash
RMG_DASHBOARD_HOST=0.0.0.0 RMG_DASHBOARD_PORT=8000 python scripts/run_stage1_dashboard.py
```

Then open:

```text
http://YOUR_LOCAL_IP:8000
```

## What it uses

- `scripts/render_stage1_overlay_video.py`
- `scripts/predict_stage1_mobilenet.py`
- existing ROI profiles from `configs/roi_annotations.template.json`

## Outputs

The dashboard stores local outputs under:

- `artifacts/webapp/uploads/`
- `artifacts/webapp/jobs/`

Each job contains:

- temporary runtime config
- overlay video
- per-station prediction CSVs
- result summary JSON
