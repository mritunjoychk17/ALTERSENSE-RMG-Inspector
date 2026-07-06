#!/usr/bin/env python3
"""Merge a base phase prediction CSV with station-specific override predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--override-csv", required=True)
    parser.add_argument("--station-ids", required=True, help="Comma-separated station IDs to take from override.")
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key_for_row(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("video_id", "")).strip(),
        str(row.get("station_id", "")).strip(),
        str(row.get("frame_index", "")).strip(),
    )


def main() -> int:
    args = parse_args()
    stations = {item.strip() for item in args.station_ids.split(",") if item.strip()}
    base_rows = read_rows(Path(args.base_csv))
    override_rows = read_rows(Path(args.override_csv))
    if not base_rows:
        raise ValueError("Base CSV is empty.")
    if not override_rows:
        raise ValueError("Override CSV is empty.")

    override_index = {
        key_for_row(row): row
        for row in override_rows
        if str(row.get("station_id", "")).strip() in stations
    }

    fieldnames: list[str] = []
    for row in [*base_rows, *override_rows]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    for extra in ["prediction_source"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    out_rows: list[dict] = []
    replaced = 0
    for base_row in base_rows:
        key = key_for_row(base_row)
        station_id = str(base_row.get("station_id", "")).strip()
        if station_id in stations and key in override_index:
            row = dict(override_index[key])
            row["prediction_source"] = "override"
            replaced += 1
        else:
            row = dict(base_row)
            row["prediction_source"] = "base"
        out_rows.append(row)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows to {output_path}")
    print(f"replaced_rows={replaced}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
