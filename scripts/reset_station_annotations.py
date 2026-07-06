#!/usr/bin/env python3
"""Clear saved station ROI polygons and optionally machine masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/roi_annotations.template.json")
    parser.add_argument("--video-id", required=True, help="Video id to reset.")
    parser.add_argument(
        "--keep-machine-masks",
        action="store_true",
        help="Keep machine masks while clearing station ROI polygons.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    video = next((v for v in config["videos"] if v["video_id"] == args.video_id), None)
    if video is None:
        raise ValueError(f"Unknown video_id: {args.video_id}")

    for station in video["workstations"]:
        station["station_roi_polygon"] = []
        if not args.keep_machine_masks:
            station["machine_mask_polygons"] = []

    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Reset annotations for {args.video_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
