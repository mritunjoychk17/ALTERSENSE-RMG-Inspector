#!/usr/bin/env python3
"""Generate visual previews for ROI and machine masks on annotation frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/roi_annotations.template.json")
    parser.add_argument("--frames-dir", default="datasets/interim/annotation_frames")
    parser.add_argument("--output-dir", default="artifacts/stage1/visualizations/roi_previews")
    parser.add_argument("--video-id", help="Generate previews for one video only.")
    return parser.parse_args()


def polygon_mask(shape: tuple[int, int], points: list[list[int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if points:
        cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return mask


def machine_mask(shape: tuple[int, int], polygons: list[list[list[int]]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        if polygon:
            cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    return mask


def bbox_from_polygon(points: list[list[int]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def main() -> int:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for video in config["videos"]:
        if args.video_id and video["video_id"] != args.video_id:
            continue
        frame_path = Path(args.frames_dir) / f"{video['video_id']}_frame_000000.jpg"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"Could not read {frame_path}")
        full_overlay = frame.copy()
        h, w = frame.shape[:2]

        for workstation in video["workstations"]:
            if not workstation["station_roi_polygon"]:
                continue
            roi = polygon_mask((h, w), workstation["station_roi_polygon"])
            machine = machine_mask((h, w), workstation["machine_mask_polygons"])
            keep = cv2.bitwise_and(roi, cv2.bitwise_not(machine))

            color = (0, 220, 0)
            pts = np.array(workstation["station_roi_polygon"], dtype=np.int32)
            overlay = full_overlay.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.18, full_overlay, 0.82, 0, full_overlay)
            cv2.polylines(full_overlay, [pts], True, color, 2, cv2.LINE_AA)

            for poly in workstation["machine_mask_polygons"]:
                if not poly:
                    continue
                mpts = np.array(poly, dtype=np.int32)
                mover = full_overlay.copy()
                cv2.fillPoly(mover, [mpts], (0, 0, 255))
                cv2.addWeighted(mover, 0.22, full_overlay, 0.78, 0, full_overlay)
                cv2.polylines(full_overlay, [mpts], True, (0, 0, 255), 2, cv2.LINE_AA)

            x1, y1, x2, y2 = bbox_from_polygon(workstation["station_roi_polygon"])
            crop = cv2.bitwise_and(frame, frame, mask=keep)[y1:y2, x1:x2]
            cv2.putText(full_overlay, f"S{workstation['station_id']}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            station_dir = output_dir / video["video_id"]
            station_dir.mkdir(parents=True, exist_ok=True)
            crop_path = station_dir / f"station_{workstation['station_id']}_masked_crop.jpg"
            if not cv2.imwrite(str(crop_path), crop):
                raise RuntimeError(f"Could not write {crop_path}")

        overview_path = output_dir / f"{video['video_id']}_overlay.jpg"
        if not cv2.imwrite(str(overview_path), full_overlay):
            raise RuntimeError(f"Could not write {overview_path}")
        print(overview_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
