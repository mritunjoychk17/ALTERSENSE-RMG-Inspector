#!/usr/bin/env python3
"""Build a station-specific phase training manifest from base and focused clip sets."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", required=True, help="Base phase clip manifest with label and pose columns.")
    parser.add_argument("--focused-csv", action="append", default=[], help="Focused phase clip manifests to oversample.")
    parser.add_argument("--station-ids", required=True, help="Comma-separated station IDs to keep.")
    parser.add_argument("--focused-repeat", type=int, default=1, help="How many times to replicate each focused manifest.")
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_station_rows(rows: list[dict], station_ids: set[str]) -> list[dict]:
    return [row for row in rows if str(row.get("station_id", "")).strip() in station_ids]


def main() -> int:
    args = parse_args()
    station_ids = {item.strip() for item in args.station_ids.split(",") if item.strip()}
    if not station_ids:
        raise ValueError("No station IDs provided.")

    base_rows = filter_station_rows(read_rows(Path(args.base_csv)), station_ids)
    if not base_rows:
        raise ValueError("No base rows matched the requested stations.")

    required = {"clip_paths", "label", "pose_label", "pose_confidence"}
    missing = sorted(key for key in required if key not in base_rows[0])
    if missing:
        raise ValueError(f"Base CSV is missing required columns: {missing}")

    fieldnames: list[str] = list(base_rows[0].keys())
    for extra in ["manifest_source", "replica_index"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    output_rows: list[dict] = []
    for row in base_rows:
        clone = dict(row)
        clone["manifest_source"] = f"base:{Path(args.base_csv).stem}"
        clone["replica_index"] = "0"
        output_rows.append(clone)

    for focused_csv in args.focused_csv:
        focused_rows = filter_station_rows(read_rows(Path(focused_csv)), station_ids)
        if not focused_rows:
            continue
        for row in focused_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        for rep in range(max(1, args.focused_repeat)):
            for row in focused_rows:
                clone = dict(row)
                clone["manifest_source"] = f"focused:{Path(focused_csv).stem}"
                clone["replica_index"] = str(rep)
                output_rows.append(clone)

    if not output_rows:
        raise ValueError("No output rows found after combining base and focused manifests.")

    output_path = Path(args.output_csv)
    write_rows(output_path, output_rows, fieldnames)

    phase_counts = Counter((row.get("label") or "").strip() for row in output_rows)
    source_counts = Counter((row.get("manifest_source") or "").strip() for row in output_rows)
    station_counts = Counter(str(row.get("station_id", "")).strip() for row in output_rows)
    print(f"Wrote {len(output_rows)} rows to {output_path}")
    print(f"station_counts={dict(sorted(station_counts.items(), key=lambda item: item[0]))}")
    print(f"source_counts={dict(sorted(source_counts.items(), key=lambda item: item[0]))}")
    print(f"phase_counts={dict(sorted(phase_counts.items(), key=lambda item: item[0]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
