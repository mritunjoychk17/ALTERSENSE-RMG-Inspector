# Stage 1 Overlay Video

Render per-station Stage 1 predictions onto the source video:

```bash
python3 scripts/render_stage1_overlay_video.py \
  --video-id cam_39_26_nov_f1_24 \
  --checkpoint artifacts/stage1/models/mobilenet_seed/best.pt \
  --present-threshold 0.3 \
  --device cuda \
  --output artifacts/stage1/visualizations/cam39_overlay.mp4
```

What the overlay shows:

- ROI polygon for each annotated station
- station id
- predicted label
- `present_confidence`

Color meaning:

- green: present
- red: absent
- yellow: uncertain, close to threshold

You can also test the cleaner adapted model later:

```bash
python3 scripts/render_stage1_overlay_video.py \
  --video-id cam_39_26_nov_f1_24 \
  --checkpoint artifacts/stage1/models/mobilenet_domain_clean/best.pt \
  --present-threshold 0.3 \
  --device cuda \
  --output artifacts/stage1/visualizations/cam39_overlay_domain_clean.mp4
```
