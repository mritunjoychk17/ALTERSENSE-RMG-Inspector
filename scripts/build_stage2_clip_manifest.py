#!/usr/bin/env python3
"""Build clip-level manifests from frame-level Stage 2 reviewed queues or ROI manifests."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, help="Reviewed queue or ROI manifest.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--label-column", default="final_label")
    parser.add_argument("--accepted-statuses", default="done,reviewed,approved")
    parser.add_argument("--clip-len", type=int, default=8)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--require-labels", action="store_true", help="Require label_column to be populated.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def infer_label(center_row: dict, label_column: str) -> str:
    return (center_row.get(label_column) or center_row.get("label") or "").strip()


def main() -> int:
    args = parse_args()
    rows = read_rows(Path(args.input_csv))
    accepted = {item.strip() for item in args.accepted_statuses.split(",") if item.strip()}
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if accepted and "review_status" in row and row.get("review_status", "") and row.get("review_status", "") not in accepted:
            continue
        video_id = row.get("video_id", "")
        station_id = str(row.get("station_id", "")).strip()
        crop_path = row.get("crop_path") or row.get("image_path")
        if not crop_path or not station_id:
            continue
        grouped[(video_id, station_id)].append(row)

    out_rows: list[dict] = []
    half = args.clip_len // 2
    for (video_id, station_id), station_rows in sorted(grouped.items(), key=lambda kv: (kv[0][0], int(kv[0][1]) if kv[0][1].isdigit() else kv[0][1])):
        station_rows.sort(key=lambda row: float(row.get("timestamp_sec") or 0.0))
        for center in range(0, len(station_rows), args.stride):
            left = center - half
            right = left + args.clip_len
            if left < 0 or right > len(station_rows):
                continue
            clip_rows = station_rows[left:right]
            center_row = station_rows[center]
            label = infer_label(center_row, args.label_column)
            if args.require_labels and not label:
                continue
            out_rows.append(
                {
                    "video_id": video_id,
                    "station_id": station_id,
                    "station_role": center_row.get("station_role", ""),
                    "frame_index": center_row.get("frame_index", ""),
                    "timestamp_sec": center_row.get("timestamp_sec", ""),
                    "label": label,
                    "clip_len": args.clip_len,
                    "clip_paths": "|".join((row.get("crop_path") or row.get("image_path") or "") for row in clip_rows),
                }
            )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["video_id", "station_id", "station_role", "frame_index", "timestamp_sec", "label", "clip_len", "clip_paths"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} clip rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
