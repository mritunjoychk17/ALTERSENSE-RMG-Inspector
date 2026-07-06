#!/usr/bin/env python3
"""Export a reference frame from each raw video for ROI and mask annotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/roi_annotations.template.json")
    parser.add_argument("--output-dir", default="datasets/interim/annotation_frames")
    parser.add_argument("--frame-index", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for video in config["videos"]:
        video_path = Path(video["video_path"])
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not read frame {args.frame_index} from {video_path}")
        output_path = output_dir / f"{video['video_id']}_frame_{args.frame_index:06d}.jpg"
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Could not write {output_path}")
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
