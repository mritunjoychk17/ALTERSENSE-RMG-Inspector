#!/usr/bin/env python3
"""Build a seed manifest that preserves label and workstation id from source paths."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="datasets/manifests/extraction_manifest.csv")
    parser.add_argument("--output", default="datasets/processed/stage1/manifests/station_seed_images.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.input, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["kind"] != "person":
                continue
            parts = Path(row["source_path"]).parts
            label = parts[-3].lower()
            station_id = parts[-2]
            rows.append(
                [
                    label,
                    station_id,
                    row["archive"],
                    row["source_path"],
                    f"datasets/{row['output_path']}",
                ]
            )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "station_id", "archive", "source_path", "image_path"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
