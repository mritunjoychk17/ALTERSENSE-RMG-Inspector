#!/usr/bin/env python3
"""Generate simple labeled overview images for each video annotation frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


GRID_POINTS = [
    ("1", 0.13, 0.68),
    ("2", 0.30, 0.72),
    ("3", 0.49, 0.76),
    ("4", 0.69, 0.76),
    ("5", 0.88, 0.73),
    ("6", 0.13, 0.23),
    ("7", 0.31, 0.22),
    ("8", 0.50, 0.20),
    ("9", 0.69, 0.22),
    ("10", 0.89, 0.22)
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/roi_annotations.template.json")
    parser.add_argument("--frames-dir", default="datasets/interim/annotation_frames")
    parser.add_argument("--output-dir", default="artifacts/stage1/visualizations/video_overviews")
    return parser.parse_args()


def annotate_frame(image: np.ndarray, video_id: str) -> np.ndarray:
    canvas = image.copy()
    h, w = canvas.shape[:2]
    overlay = canvas.copy()

    for station_id, fx, fy in GRID_POINTS:
        x = int(w * fx)
        y = int(h * fy)
        cv2.circle(overlay, (x, y), 44, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(overlay, station_id, (x - 18, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.26, canvas, 0.74, 0, canvas)
    cv2.putText(canvas, f"Station mapping guide: {video_id}", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, "Use station contact sheets to verify each workstation before drawing polygons.", (28, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main() -> int:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for video in config["videos"]:
        frame_path = frames_dir / f"{video['video_id']}_frame_000000.jpg"
        image = cv2.imread(str(frame_path))
        if image is None:
            raise RuntimeError(f"Could not read {frame_path}")
        board = annotate_frame(image, video["video_id"])
        out_path = output_dir / f"{video['video_id']}_overview.jpg"
        if not cv2.imwrite(str(out_path), board):
            raise RuntimeError(f"Could not write {out_path}")
        print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
