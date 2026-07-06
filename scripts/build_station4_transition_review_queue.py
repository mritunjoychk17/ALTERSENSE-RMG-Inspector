#!/usr/bin/env python3
"""Build a focused manual review queue for station 4 transition boundaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDNAMES = [
    "video_id",
    "station_id",
    "station_role",
    "frame_index",
    "timestamp_sec",
    "crop_path",
    "prev_crop_path",
    "next_crop_path",
    "clip_paths",
    "clip_len",
    "pose_label",
    "pose_confidence",
    "predicted_label",
    "smoothed_label",
    "phase_label",
    "clip_validation_note",
    "review_focus",
    "final_label",
    "review_status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validated-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--station-id", default="4")
    parser.add_argument("--window", type=int, default=2, help="Include +/- this many rows around each transition note.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clip_triplet(row: dict) -> tuple[str, str, str]:
    clip_paths = [item.strip() for item in (row.get("clip_paths") or "").split("|") if item.strip()]
    if not clip_paths:
        return "", "", ""
    center = len(clip_paths) // 2
    crop = clip_paths[center]
    prev_crop = clip_paths[max(0, center - 1)] if center > 0 else ""
    next_crop = clip_paths[min(len(clip_paths) - 1, center + 1)] if center + 1 < len(clip_paths) else ""
    return crop, prev_crop, next_crop


def main() -> int:
    args = parse_args()
    rows = read_rows(Path(args.validated_csv))
    station_rows = [row for row in rows if str(row.get("station_id", "")).strip() == args.station_id]
    station_rows.sort(key=lambda row: float(row.get("timestamp_sec") or 0.0))

    selected = set()
    for idx, row in enumerate(station_rows):
        note = (row.get("clip_validation_note") or "").strip()
        if not note:
            continue
        for pos in range(max(0, idx - args.window), min(len(station_rows), idx + args.window + 1)):
            selected.add(pos)

    out_rows = []
    for pos in sorted(selected):
        source = dict(station_rows[pos])
        crop_path, prev_crop_path, next_crop_path = clip_triplet(source)
        out_rows.append(
            {
                "video_id": source.get("video_id", ""),
                "station_id": source.get("station_id", ""),
                "station_role": source.get("station_role", ""),
                "frame_index": source.get("frame_index", ""),
                "timestamp_sec": source.get("timestamp_sec", ""),
                "crop_path": crop_path,
                "prev_crop_path": prev_crop_path,
                "next_crop_path": next_crop_path,
                "clip_paths": source.get("clip_paths", ""),
                "clip_len": source.get("clip_len", ""),
                "pose_label": source.get("pose_label", ""),
                "pose_confidence": source.get("pose_confidence", ""),
                "predicted_label": source.get("predicted_label", ""),
                "smoothed_label": source.get("smoothed_label", ""),
                "phase_label": source.get("phase_label", ""),
                "clip_validation_note": source.get("clip_validation_note", ""),
                "review_focus": "station4_transition",
                "final_label": source.get("final_label", ""),
                "review_status": source.get("review_status", "") or "pending",
                "notes": source.get("notes", ""),
            }
        )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if out_rows:
            writer.writeheader()
            writer.writerows(out_rows)
    print(output_path)
    print(f"rows={len(out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
