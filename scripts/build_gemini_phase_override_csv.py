#!/usr/bin/env python3
"""Build a phase override CSV from existing Gemini-labeled rows aligned to a base prediction CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage2_taxonomy import normalize_cycle_phase_for_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--gemini-csv", required=True)
    parser.add_argument("--station-ids", required=True, help="Comma-separated station IDs to keep from Gemini.")
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
    station_ids = {item.strip() for item in args.station_ids.split(",") if item.strip()}
    base_rows = read_rows(Path(args.base_csv))
    gemini_rows = read_rows(Path(args.gemini_csv))
    if not base_rows:
        raise ValueError("Base CSV is empty.")
    if not gemini_rows:
        raise ValueError("Gemini CSV is empty.")

    gemini_index = {
        key_for_row(row): row
        for row in gemini_rows
        if str(row.get("station_id", "")).strip() in station_ids
    }

    output_rows: list[dict] = []
    for base_row in base_rows:
        key = key_for_row(base_row)
        station_id = str(base_row.get("station_id", "")).strip()
        if station_id not in station_ids or key not in gemini_index:
            continue
        gem_row = gemini_index[key]
        phase = normalize_cycle_phase_for_row(gem_row.get("gemini_cycle_phase", ""), gem_row)
        row = dict(base_row)
        row["predicted_phase"] = phase
        row["smoothed_phase"] = phase
        row["prediction_source"] = "gemini_override"
        row["gemini_override_label"] = gem_row.get("gemini_safe_label", "")
        row["gemini_override_phase"] = gem_row.get("gemini_cycle_phase", "")
        row["gemini_override_confidence"] = gem_row.get("gemini_confidence", "")
        output_rows.append(row)

    if not output_rows:
        raise ValueError("No aligned Gemini override rows were found.")

    fieldnames = list(output_rows[0].keys())
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
