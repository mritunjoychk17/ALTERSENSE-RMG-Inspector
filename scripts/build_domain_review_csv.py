#!/usr/bin/env python3
"""Build a clean reviewed CSV from selected present/absent reference video frames."""

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
    parser.add_argument("--spec", default="configs/domain_adaptation_spec.json")
    parser.add_argument("--output-dir", default="datasets/processed/stage1/domain_reference_crops")
    parser.add_argument("--output-csv", default="datasets/processed/stage1/manifests/domain_review_queue.csv")
    return parser.parse_args()


def mask_from_station(frame_shape: tuple[int, int], station: dict) -> np.ndarray:
    roi = np.zeros(frame_shape, dtype=np.uint8)
    cv2.fillPoly(roi, [np.array(station["station_roi_polygon"], dtype=np.int32)], 255)
    machine = np.zeros(frame_shape, dtype=np.uint8)
    for poly in station["machine_mask_polygons"]:
        if poly:
            cv2.fillPoly(machine, [np.array(poly, dtype=np.int32)], 255)
    return cv2.bitwise_and(roi, cv2.bitwise_not(machine))


def bbox_from_polygon(points: list[list[int]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def main() -> int:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for item in spec["items"]:
        video = next(v for v in config["videos"] if v["video_id"] == item["video_id"])
        stations = video["workstations"]
        if item["stations"] == "annotated":
            stations = [ws for ws in stations if ws["station_roi_polygon"]]
        cap = cv2.VideoCapture(video["video_path"])
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {video['video_path']}")
        for frame_index in item["frame_indices"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            for station in stations:
                mask = mask_from_station(frame.shape[:2], station)
                x1, y1, x2, y2 = bbox_from_polygon(station["station_roi_polygon"])
                crop = cv2.bitwise_and(frame, frame, mask=mask)[y1:y2, x1:x2]
                label_dir = output_dir / item["label"]
                label_dir.mkdir(parents=True, exist_ok=True)
                out_path = label_dir / f"{video['video_id']}_station_{station['station_id']}_frame_{frame_index:06d}.jpg"
                if not cv2.imwrite(str(out_path), crop):
                    raise RuntimeError(f"Could not write {out_path}")
                rows.append(
                    {
                        "video_id": video["video_id"],
                        "station_id": station["station_id"],
                        "frame_index": str(frame_index),
                        "timestamp_sec": str(round(frame_index / (cap.get(cv2.CAP_PROP_FPS) or 1.0), 3)),
                        "crop_path": str(out_path),
                        "label": item["label"],
                        "review_status": "done",
                        "notes": "reference_frame_label",
                        "model_prediction": "",
                        "model_confidence": "",
                        "present_confidence": "",
                        "absent_confidence": "",
                        "final_label": item["label"]
                    }
                )
        cap.release()

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
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
                "final_label"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
