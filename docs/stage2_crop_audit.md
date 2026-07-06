# Stage 2 Crop Audit

Use this before re-annotating anything.

The goal is to verify whether the current Stage 1 station ROI crops are good
enough for Stage 2 activity recognition.

## What to check

For each station audit image, ask:

1. Is the worker clearly inside the crop when present?
2. Are the hands or work-area interactions visible enough?
3. Is the machine-side work zone visible?
4. Is too much neighboring station or aisle included?
5. When the worker leans or stands, do they leave the crop too easily?

If these fail often, then:

- either re-annotate the station ROI
- or add a dedicated Stage 2 work-area ROI later

## Generate audit boards

```bash
source rmg/bin/activate
python scripts/audit_stage2_crops.py \
  --manifest datasets/interim/roi_crops/manifest.csv \
  --samples-per-station 6
```

Outputs:

- `artifacts/stage2/visualizations/crop_audit/<video_id>/station_<id>_audit.jpg`
- `artifacts/stage2/visualizations/crop_audit/audit_manifest.csv`

## Focus first

Start with the stations that look problematic in Stage 2:

- stations where the worker is partly missing
- stations where hands are rarely visible
- stations with too much empty background
- stations where the worker stands outside the crop
