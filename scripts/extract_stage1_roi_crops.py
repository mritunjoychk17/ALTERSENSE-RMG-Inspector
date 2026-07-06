#!/usr/bin/env python3
"""Extract ROI-based crops for Stage 1 presence detection."""

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
    parser.add_argument("--output-dir", default="datasets/interim/roi_crops")
    parser.add_argument("--sample-every", type=int, default=20, help="Keep 1 frame every N frames.")
    parser.add_argument("--max-frames-per-video", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--video-id", help="Extract crops for one video_id only.")
    parser.add_argument(
        "--skip-empty-videos",
        action="store_true",
        help="Skip videos with no annotated station ROIs instead of raising an error.",
    )
    parser.add_argument("--preview", action="store_true", help="Validate config without writing crops.")
    return parser.parse_args()


def polygon_to_array(points: list[list[int]]) -> np.ndarray:
    return np.array(points, dtype=np.int32)


def build_masks(
    frame_shape: tuple[int, int, int],
    roi_points: list[list[int]],
    machine_polygons: list[list[list[int]]],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame_shape[:2]
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [polygon_to_array(roi_points)], 255)

    machine_mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in machine_polygons:
        cv2.fillPoly(machine_mask, [polygon_to_array(polygon)], 255)
    return roi_mask, machine_mask


def crop_bbox_from_polygon(points: list[list[int]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def validate_workstation(video_id: str, workstation: dict) -> None:
    if not workstation["station_roi_polygon"]:
        raise ValueError(f"Missing station ROI polygon for {video_id} station {workstation['station_id']}")


def main() -> int:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    rows: list[list[str | int | float]] = []

    for video in config["videos"]:
        if args.video_id and video["video_id"] != args.video_id:
            continue
        video_path = Path(video["video_path"])
        annotated_workstations = [ws for ws in video["workstations"] if ws["station_roi_polygon"]]
        if not annotated_workstations:
            if args.skip_empty_videos:
                print(f"{video['video_id']}: skipped (no annotated workstations)")
                continue
            raise ValueError(f"No annotated workstations found for {video['video_id']}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video_path}")

        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Could not read first frame from {video_path}")

        workstations = []
        for workstation in video["workstations"]:
            if not workstation["station_roi_polygon"]:
                continue
            validate_workstation(video["video_id"], workstation)
            roi_mask, machine_mask = build_masks(
                frame.shape,
                workstation["station_roi_polygon"],
                workstation["machine_mask_polygons"],
            )
            roi_keep_mask = cv2.bitwise_and(roi_mask, cv2.bitwise_not(machine_mask))
            bbox = crop_bbox_from_polygon(workstation["station_roi_polygon"])
            workstations.append((workstation["station_id"], roi_keep_mask, bbox))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_index = 0
        saved_frames = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % args.sample_every != 0:
                frame_index += 1
                continue
            if args.max_frames_per_video and saved_frames >= args.max_frames_per_video:
                break

            timestamp_sec = frame_index / max(float(video.get("fps", 20.0)), 1.0)
            for station_id, roi_keep_mask, bbox in workstations:
                x1, y1, x2, y2 = bbox
                masked = cv2.bitwise_and(frame, frame, mask=roi_keep_mask)
                crop = masked[y1:y2, x1:x2]

                if not args.preview:
                    station_dir = output_dir / video["video_id"] / f"station_{station_id}"
                    station_dir.mkdir(parents=True, exist_ok=True)
                    out_path = station_dir / f"{video['video_id']}_station_{station_id}_frame_{frame_index:06d}.jpg"
                    if not cv2.imwrite(str(out_path), crop):
                        raise RuntimeError(f"Could not write crop to {out_path}")
                    rows.append(
                        [
                            video["video_id"],
                            station_id,
                            frame_index,
                            round(timestamp_sec, 3),
                            str(video_path),
                            str(out_path),
                            x1,
                            y1,
                            x2,
                            y2,
                        ]
                    )

            saved_frames += 1
            frame_index += 1

        cap.release()
        print(f"{video['video_id']}: saved {saved_frames} sampled frames across {len(workstations)} workstations")

    if not args.preview:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "video_id",
                    "station_id",
                    "frame_index",
                    "timestamp_sec",
                    "video_path",
                    "crop_path",
                    "bbox_x1",
                    "bbox_y1",
                    "bbox_x2",
                    "bbox_y2",
                ]
            )
            writer.writerows(rows)
        print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
