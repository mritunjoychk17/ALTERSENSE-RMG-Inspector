#!/usr/bin/env python3
"""Merge model predictions into the ROI crop label queue for review."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", default="datasets/processed/stage1/manifests/roi_crop_label_queue.csv")
    parser.add_argument("--predictions", default="artifacts/stage1/eval/roi_crop_predictions.csv")
    parser.add_argument("--output", default="datasets/processed/stage1/manifests/roi_crop_review_queue.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pred_map = {}
    with open(args.predictions, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pred_map[row["source"]] = row

    rows = []
    with open(args.queue, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pred = pred_map.get(row["crop_path"], {})
            rows.append(
                {
                    **row,
                    "model_prediction": pred.get("predicted_label", ""),
                    "model_confidence": pred.get("confidence", ""),
                    "present_confidence": pred.get("present_confidence", ""),
                    "absent_confidence": pred.get("absent_confidence", ""),
                    "final_label": "",
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "station_id",
                "frame_index",
                "timestamp_sec",
                "crop_path",
                "label",
                "review_status",
                "notes",
                "model_prediction",
                "model_confidence",
                "present_confidence",
                "absent_confidence",
                "final_label",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
