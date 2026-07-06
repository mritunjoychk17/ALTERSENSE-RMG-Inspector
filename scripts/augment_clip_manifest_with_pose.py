#!/usr/bin/env python3
"""Attach frame-level pose suggestions to clip manifests using center timestamp."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-csv", required=True)
    parser.add_argument("--frame-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key_for(row: dict) -> tuple[str, str]:
    station_id = str(row.get("station_id", "")).strip()
    timestamp = f"{float(row.get('timestamp_sec') or 0.0):.3f}"
    return station_id, timestamp


def main() -> int:
    args = parse_args()
    clip_rows = read_rows(Path(args.clip_csv))
    frame_rows = read_rows(Path(args.frame_csv))
    frame_index = {key_for(row): row for row in frame_rows}

    output_rows = []
    for row in clip_rows:
        clone = dict(row)
        match = frame_index.get(key_for(row), {})
        clone["pose_label"] = match.get("pose_label", "")
        clone["pose_confidence"] = match.get("pose_confidence", "")
        clone["pose_reason"] = match.get("pose_reason", "")
        output_rows.append(clone)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()) if output_rows else [])
        if output_rows:
            writer.writeheader()
            writer.writerows(output_rows)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
