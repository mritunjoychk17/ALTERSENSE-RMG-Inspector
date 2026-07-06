#!/usr/bin/env python3
"""Run the simple mask baseline across all annotated stations for one video."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/roi_annotations.template.json")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output-dir", default="artifacts/stage1/eval/batch_mask_baseline")
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=18.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    video = next((v for v in config["videos"] if v["video_id"] == args.video_id), None)
    if video is None:
        raise ValueError(f"Unknown video_id: {args.video_id}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_rows = []

    for ws in video["workstations"]:
        if not ws["station_roi_polygon"]:
            continue
        out_path = output_dir / f"{args.video_id}_station_{ws['station_id']}.csv"
        cmd = [
            sys.executable,
            "scripts/run_mask_presence_baseline.py",
            "--video-id",
            args.video_id,
            "--station-id",
            ws["station_id"],
            "--output",
            str(out_path),
            "--sample-every",
            str(args.sample_every),
            "--threshold",
            str(args.threshold),
        ]
        if args.max_frames:
            cmd.extend(["--max-frames", str(args.max_frames)])
        subprocess.run(cmd, check=True)
        with out_path.open(newline="", encoding="utf-8") as f:
            merged_rows.extend(list(csv.DictReader(f)))

    merged_path = output_dir / f"{args.video_id}_all_stations.csv"
    with merged_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "station_id",
                "frame_index",
                "timestamp_sec",
                "score",
                "threshold",
                "predicted_label",
            ],
        )
        writer.writeheader()
        writer.writerows(merged_rows)
    print(merged_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
