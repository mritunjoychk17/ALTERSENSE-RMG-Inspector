#!/usr/bin/env python3
"""Bootstrap review labels from model scores for faster manual correction.

This does not replace review. It only pre-fills easy cases.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="datasets/processed/stage1/manifests/roi_crop_review_queue.csv")
    parser.add_argument("--output", default="datasets/processed/stage1/manifests/roi_crop_review_queue_bootstrapped.csv")
    parser.add_argument("--present-threshold", type=float, default=0.7)
    parser.add_argument("--absent-threshold", type=float, default=0.8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    with open(args.input, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            present = float(row["present_confidence"] or 0.0)
            absent = float(row["absent_confidence"] or 0.0)
            if present >= args.present_threshold:
                row["final_label"] = "present"
                row["review_status"] = "bootstrapped"
            elif absent >= args.absent_threshold:
                row["final_label"] = "absent"
                row["review_status"] = "bootstrapped"
            rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
