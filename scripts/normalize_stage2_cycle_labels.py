#!/usr/bin/env python3
"""Create a reduced, cycle-focused Stage 2 review CSV from reviewed labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--operator-config", default="configs/altersense_operator_profiles.cam33.json")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--target-role", default="sewing", help="Only normalize rows for this station role.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(Path(args.operator_config).read_text(encoding="utf-8"))
    station_role_map = {
        str(item["station_id"]): item.get("station_role", "")
        for item in config.get("stations", [])
    }
    label_map_by_role = config.get("cycle_label_map_by_role", {})

    with Path(args.input_csv).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Input CSV is empty.")

    out_rows: list[dict] = []
    for row in rows:
        station_id = str(row.get("station_id", "")).strip()
        station_role = (row.get("station_role") or station_role_map.get(station_id, "")).strip()
        if args.target_role and station_role != args.target_role:
            continue
        label = (row.get("final_label") or row.get("label") or "").strip()
        if not label:
            continue
        mapped = label_map_by_role.get(station_role, {}).get(label, label)
        clone = dict(row)
        if "manual_final_label" not in clone:
            clone["manual_final_label"] = label
        clone["final_label"] = mapped
        clone["label"] = mapped
        clone["station_role"] = station_role
        out_rows.append(clone)

    fieldnames = list(out_rows[0].keys()) if out_rows else list(rows[0].keys())
    if "manual_final_label" not in fieldnames:
        fieldnames.append("manual_final_label")
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
