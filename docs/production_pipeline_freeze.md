# ALTERSENSE Production Pipeline Freeze

This is the **single pipeline** we should treat as the current production-ready direction for this repo.

## Chosen pipeline

1. **Stage 1 presence**
   - fixed station ROI annotations
   - MobileNetV3 presence classifier
   - output used only to gate workstation presence and present-time KPIs

2. **Stage 2 activity**
   - clip-based GRU activity model
   - station-specific validation layer over raw clip predictions
   - longer `len12 / stride1` clip overrides for weak stations `4` and `6`

3. **Cycle reporting**
   - `verified_cycle_count` is the production KPI
   - `heuristic_cycle_count` stays for internal audit only
   - `reliability_badge` is required in the UI

## Current report to trust

Use:

`artifacts/altersense/operator_report_cam33_clip_station_1456_station46_len12.json`

This is the best current balance between:

- respecting real cycle closure
- reducing heuristic inflation
- surviving difficult ceiling-view geometry

## What not to use as the production source

Do **not** use the older dense heuristic-heavy report:

`artifacts/altersense/operator_report_cam33_dense_cycle_v1_post.json`

That report over-recovers cycles and is too optimistic for production KPI reporting.

## Deployment architecture

For production reliability:

- **Vercel** should host the dashboard UI
- **Python GPU worker** should run video inference
- the UI should consume the latest verified JSON report from the worker output

This split is more reliable than trying to run full video inference directly inside Vercel serverless functions.
