#!/usr/bin/env python3
"""Build a simple labeling queue CSV from extracted ROI crops."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="datasets/interim/roi_crops/manifest.csv")
    parser.add_argument("--video-id", default="")
    parser.add_argument("--output", default="datasets/processed/stage1/manifests/roi_crop_label_queue.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if args.video_id and row["video_id"] != args.video_id:
                continue
            rows.append(
                {
                    "video_id": row["video_id"],
                    "station_id": row["station_id"],
                    "frame_index": row["frame_index"],
                    "timestamp_sec": row["timestamp_sec"],
                    "crop_path": row["crop_path"],
                    "label": "",
                    "review_status": "pending",
                    "notes": "",
                }
            )

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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
