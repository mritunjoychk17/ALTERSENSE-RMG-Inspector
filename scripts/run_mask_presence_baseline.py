#!/usr/bin/env python3
"""Simple no-training presence baseline using masked ROI occupancy change."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/roi_annotations.template.json")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--station-id", required=True)
    parser.add_argument("--output", default="artifacts/stage1/eval/mask_baseline.csv")
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=18.0)
    return parser.parse_args()


def mask_from_polygons(shape: tuple[int, int], roi_points: list[list[int]], machine_polygons: list[list[list[int]]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(roi_points, dtype=np.int32)], 255)
    machine = np.zeros(shape, dtype=np.uint8)
    for polygon in machine_polygons:
        if polygon:
            cv2.fillPoly(machine, [np.array(polygon, dtype=np.int32)], 255)
    return cv2.bitwise_and(mask, cv2.bitwise_not(machine))


def bbox_from_polygon(points: list[list[int]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def main() -> int:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    video = next((v for v in config["videos"] if v["video_id"] == args.video_id), None)
    if video is None:
        raise ValueError(f"Unknown video_id: {args.video_id}")
    station = next((s for s in video["workstations"] if s["station_id"] == args.station_id), None)
    if station is None:
        raise ValueError(f"Unknown station_id: {args.station_id}")
    if not station["station_roi_polygon"]:
        raise ValueError("Station ROI is empty. Annotate first.")

    cap = cv2.VideoCapture(video["video_path"])
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read first frame from {video['video_path']}")

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    mask = mask_from_polygons(gray0.shape, station["station_roi_polygon"], station["machine_mask_polygons"])
    x1, y1, x2, y2 = bbox_from_polygon(station["station_roi_polygon"])
    reference = gray0.copy()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    frame_index = 0
    saved = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % args.sample_every != 0:
            frame_index += 1
            continue
        if args.max_frames and saved >= args.max_frames:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, reference)
        score = float(diff[mask > 0].mean()) if np.any(mask > 0) else 0.0
        label = "present" if score >= args.threshold else "absent"
        rows.append([args.video_id, args.station_id, frame_index, round(frame_index / fps, 3), round(score, 4), args.threshold, label])
        saved += 1
        frame_index += 1

    cap.release()
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "station_id", "frame_index", "timestamp_sec", "score", "threshold", "predicted_label"])
        writer.writerows(rows)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
