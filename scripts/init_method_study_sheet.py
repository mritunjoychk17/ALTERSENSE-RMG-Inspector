#!/usr/bin/env python3
"""Initialize a method-study segment sheet from a focused review queue."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDNAMES = [
    "serial",
    "station_id",
    "anchor_frame_index",
    "anchor_timestamp_sec",
    "start_time_sec",
    "end_time_sec",
    "duration_sec",
    "action_label",
    "content",
    "left_hand_zone",
    "right_hand_zone",
    "object_context",
    "review_status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queue_path = Path(args.queue_csv)
    output_path = Path(args.output_csv)
    with queue_path.open(newline="", encoding="utf-8") as f:
        queue_rows = list(csv.DictReader(f))

    rows = []
    for idx, row in enumerate(queue_rows, start=1):
        rows.append(
            {
                "serial": idx,
                "station_id": row.get("station_id", ""),
                "anchor_frame_index": row.get("frame_index", ""),
                "anchor_timestamp_sec": row.get("timestamp_sec", ""),
                "start_time_sec": "",
                "end_time_sec": "",
                "duration_sec": "",
                "action_label": "",
                "content": "",
                "left_hand_zone": "",
                "right_hand_zone": "",
                "object_context": "",
                "review_status": "pending",
                "notes": row.get("notes", ""),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
