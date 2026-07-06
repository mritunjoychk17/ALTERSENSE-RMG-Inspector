# Annotation Workflow

This workflow is for Stage 1 only and supports multiple workstations per video.
The exact station count can vary by camera view.

## Files to use

- `datasets/interim/annotation_frames/`: one frame per video for polygon annotation
- `artifacts/stage1/visualizations/station_contact_sheets/`: station-wise present/absent seed references
- `artifacts/stage1/visualizations/video_overviews/`: full-frame guides with station labels
- `configs/roi_annotations.template.json`: where the polygons are saved

## Click-based annotator

Run:

```bash
python3 scripts/annotate_station_rois.py
```

Controls:

- Left click: add point
- `Enter`: save current polygon
- `u`: undo last point
- `x`: clear current polygon
- `c`: annotate station ROI
- `m`: annotate machine mask polygon
- `d`: delete last machine mask for the current station
- `[` and `]`: previous or next station
- `-` and `=`: previous or next video
- `s`: save JSON
- `q`: quit

## Recommended order

1. Open the overview image for a video.
2. Compare each workstation against the station contact sheets.
3. Use the annotator to draw the full station ROI for each actual station in the frame.
4. Add machine mask polygons for sewing machine regions that should be excluded.
5. Repeat for all 5 videos.
6. Run ROI crop extraction after the JSON is filled.
