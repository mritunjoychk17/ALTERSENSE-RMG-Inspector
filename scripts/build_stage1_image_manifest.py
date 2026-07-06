#!/usr/bin/env python3
"""Build a simple manifest from the seed person-image dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="datasets/raw/person")
    parser.add_argument("--output", default="datasets/processed/stage1/manifests/seed_images.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for label_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        label = label_dir.name
        for image_path in sorted(label_dir.iterdir()):
            if image_path.is_file():
                rows.append([label, str(image_path)])

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "image_path"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
