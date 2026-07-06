#!/usr/bin/env python3
"""Build a station-focused Stage 2 clip training subset."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-csv", required=True, help="Input clip manifest CSV.")
    parser.add_argument(
        "--station-ids",
        required=True,
        nargs="+",
        help="Station IDs to keep in the focused subset.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output CSV path for the focused subset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    station_ids = {str(value).strip() for value in args.station_ids}
    with open(args.clip_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Input clip CSV is empty.")

    output_rows = [row for row in rows if str(row.get("station_id", "")).strip() in station_ids]
    if not output_rows:
        raise ValueError(f"No rows matched station IDs: {sorted(station_ids)}")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    station_counts = Counter(str(row.get("station_id", "")).strip() for row in output_rows)
    label_counts = Counter((row.get("label") or "").strip() for row in output_rows)
    print(f"Wrote {len(output_rows)} rows to {output_path}")
    print(f"station_counts={dict(sorted(station_counts.items(), key=lambda item: item[0]))}")
    print(f"label_counts={dict(sorted(label_counts.items(), key=lambda item: item[0]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
