#!/usr/bin/env python3
"""Combine multiple clip manifest CSV files into one."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, help="Comma-separated clip CSV files.")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    fieldnames = None
    for item in args.inputs.split(","):
        path = Path(item.strip())
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            batch = list(reader)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            rows.extend(batch)
    if not rows or not fieldnames:
        raise ValueError("No rows to combine.")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
