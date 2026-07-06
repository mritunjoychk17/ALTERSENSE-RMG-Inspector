# Activity Stage 2 Commands

## 1. Extract the labeled activity images

```bash
source rmg/bin/activate
python scripts/prepare_activity_dataset.py
```

This writes:

- `datasets/raw/activity/get_put`
- `datasets/raw/activity/sew`
- `datasets/manifests/activity_extraction_manifest.csv`

## 2. Train the seed Stage 2 classifier

```bash
source rmg/bin/activate
python scripts/train_stage2_activity_mobilenet.py \
  --data-dir datasets/raw/activity \
  --device cuda \
  --output-dir artifacts/stage2/models/mobilenet_seed
```

## 3. Test on the extracted activity images

```bash
source rmg/bin/activate
python scripts/predict_stage2_activity.py \
  --checkpoint artifacts/stage2/models/mobilenet_seed/best.pt \
  --image-dir datasets/raw/activity/sew \
  --device cuda \
  --output artifacts/stage2/eval/sew_seed_predictions.csv
```

```bash
source rmg/bin/activate
python scripts/predict_stage2_activity.py \
  --checkpoint artifacts/stage2/models/mobilenet_seed/best.pt \
  --image-dir datasets/raw/activity/get_put \
  --device cuda \
  --output artifacts/stage2/eval/get_put_seed_predictions.csv
```

## 4. Next planned upgrade

After the seed image model works:

1. run Stage 1 to keep only present station crops
2. build an activity review queue from present ROI crops
3. label those crops as `sew` or `get_put`
4. fine-tune Stage 2 on video-domain activity crops
