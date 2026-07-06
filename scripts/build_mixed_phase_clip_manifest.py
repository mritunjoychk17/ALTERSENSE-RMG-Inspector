#!/usr/bin/env python3
"""Build a mixed phase clip manifest from a base set plus one or more focused sets."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--focused-csv", action="append", default=[])
    parser.add_argument("--focused-repeat", type=int, default=1)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    args = parse_args()
    base_rows = read_rows(Path(args.base_csv))
    if not base_rows:
        raise ValueError("Base CSV is empty.")

    fieldnames: list[str] = []

    def update_fieldnames(rows: list[dict]) -> None:
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

    update_fieldnames(base_rows)
    out_rows: list[dict] = []
    for row in base_rows:
        clone = dict(row)
        clone["manifest_source"] = "base"
        clone["replica_index"] = "0"
        out_rows.append(clone)

    for focused_csv in args.focused_csv:
        focused_rows = read_rows(Path(focused_csv))
        update_fieldnames(focused_rows)
        for rep in range(max(1, args.focused_repeat)):
            for row in focused_rows:
                clone = dict(row)
                clone["manifest_source"] = f"focused:{Path(focused_csv).stem}"
                clone["replica_index"] = str(rep)
                out_rows.append(clone)

    for extra in ["manifest_source", "replica_index"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    label_counts = Counter((row.get("label") or "").strip() for row in out_rows)
    source_counts = Counter((row.get("manifest_source") or "").strip() for row in out_rows)
    print(f"Wrote {len(out_rows)} rows to {output_path}")
    print(f"source_counts={dict(sorted(source_counts.items(), key=lambda item: item[0]))}")
    print(f"label_counts={dict(sorted(label_counts.items(), key=lambda item: item[0]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
