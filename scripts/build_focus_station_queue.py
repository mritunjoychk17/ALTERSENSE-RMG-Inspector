#!/usr/bin/env python3
"""Filter a review queue down to selected stations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--stations", required=True, help="Comma-separated station ids, e.g. 3,4")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stations = {item.strip() for item in args.stations.split(",") if item.strip()}
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    with input_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    kept = [row for row in rows if row.get("station_id", "") in stations]
    for row in kept:
        row.setdefault("qwen_segment_decision", "")
        row.setdefault("qwen_action_label", "")
        row.setdefault("qwen_segment_text", "")
        row.setdefault("qwen_confidence", "")
        row.setdefault("qwen_reason", "")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(kept[0].keys()) if kept else list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(kept)
    print(f"Wrote {len(kept)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
