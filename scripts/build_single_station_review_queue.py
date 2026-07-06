#!/usr/bin/env python3
"""Export frames from Single Station video into a Stage 2 review queue for local manual labeling."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


FIELDNAMES = [
    "video_id",
    "station_id",
    "station_role",
    "frame_index",
    "timestamp_sec",
    "crop_path",
    "prev_crop_path",
    "next_crop_path",
    "presence_confidence",
    "pose_label",
    "pose_confidence",
    "pose_reason",
    "gemini_label",
    "gemini_confidence",
    "gemini_reason",
    "final_label",
    "review_status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="Single Station.mp4")
    parser.add_argument("--video-id", default="single_station")
    parser.add_argument("--station-id", default="1")
    parser.add_argument("--station-role", default="sewing")
    parser.add_argument("--sample-every-sec", type=float, default=0.5)
    parser.add_argument("--output-dir", default="datasets/interim/single_station_frames")
    parser.add_argument("--output-csv", default="datasets/processed/stage2/manifests/single_station_review_queue.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sample_every_frames = max(1, int(round(args.sample_every_sec * fps)))
    out_dir = Path(args.output_dir) / args.video_id / f"station_{args.station_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % sample_every_frames != 0:
            frame_index += 1
            continue
        crop_path = out_dir / f"{args.video_id}_station_{args.station_id}_frame_{frame_index:06d}.jpg"
        cv2.imwrite(str(crop_path), frame)
        saved.append(
            {
                "video_id": args.video_id,
                "station_id": args.station_id,
                "station_role": args.station_role,
                "frame_index": str(frame_index),
                "timestamp_sec": f"{frame_index / fps:.3f}",
                "crop_path": str(crop_path),
            }
        )
        frame_index += 1
    cap.release()

    rows = []
    for idx, item in enumerate(saved):
        rows.append(
            {
                **item,
                "prev_crop_path": saved[idx - 1]["crop_path"] if idx > 0 else "",
                "next_crop_path": saved[idx + 1]["crop_path"] if idx < len(saved) - 1 else "",
                "presence_confidence": "",
                "pose_label": "",
                "pose_confidence": "",
                "pose_reason": "",
                "gemini_label": "",
                "gemini_confidence": "",
                "gemini_reason": "",
                "final_label": "",
                "review_status": "pending",
                "notes": "Single-station cycle-focused review queue.",
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
