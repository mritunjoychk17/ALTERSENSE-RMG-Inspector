#!/usr/bin/env python3
"""Merge a base Stage 2 activity CSV with a denser overlay CSV, preferring overlay rows on matching timestamps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--overlay-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def row_key(row: dict) -> tuple[str, str, str]:
    return (row.get("video_id", ""), row.get("station_id", ""), str(row.get("timestamp_sec", "")))


def main() -> int:
    args = parse_args()
    base_rows = list(csv.DictReader(open(args.base_csv, newline="", encoding="utf-8")))
    overlay_rows = list(csv.DictReader(open(args.overlay_csv, newline="", encoding="utf-8")))
    if not base_rows:
        raise ValueError("Base CSV is empty.")
    overlay_map = {row_key(row): row for row in overlay_rows}
    merged: list[dict] = []
    seen = set()
    for row in base_rows:
        key = row_key(row)
        if key in overlay_map:
            merged.append(overlay_map[key])
            seen.add(key)
        else:
            merged.append(row)
            seen.add(key)
    for row in overlay_rows:
        key = row_key(row)
        if key not in seen:
            merged.append(row)
    merged.sort(key=lambda r: (r.get("video_id", ""), int(r.get("station_id", "0") or 0), float(r.get("timestamp_sec", "0") or 0.0)))

    fieldnames: list[str] = []
    for row in merged:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)
    print(f"Wrote {len(merged)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
