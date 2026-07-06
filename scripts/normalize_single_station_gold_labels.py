#!/usr/bin/env python3
"""Normalize the Single Station review queue to the expert PDF gold standard."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-csv", default="datasets/processed/stage2/manifests/single_station_review_queue.csv")
    parser.add_argument("--gold-config", default="configs/single_station_gold_standard.json")
    parser.add_argument("--output-csv", default="datasets/processed/stage2/manifests/single_station_review_queue_gold.csv")
    return parser.parse_args()


def gold_label_for_timestamp(timestamp_sec: float, segments: list[dict]) -> dict:
    for segment in segments:
        if segment["start_sec"] <= timestamp_sec < segment["end_sec"]:
            return segment
    return segments[-1]


def main() -> int:
    args = parse_args()
    gold = json.loads(Path(args.gold_config).read_text(encoding="utf-8"))
    segments = gold["segments"]

    with Path(args.queue_csv).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Queue CSV is empty.")

    fieldnames = list(rows[0].keys())
    for extra in ["manual_final_label", "gold_standard_label", "gold_standard_content", "gold_standard_source"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    normalized = 0
    for row in rows:
        timestamp = float(row.get("timestamp_sec") or 0.0)
        segment = gold_label_for_timestamp(timestamp, segments)
        row["manual_final_label"] = row.get("final_label", "")
        row["gold_standard_label"] = segment["label"]
        row["gold_standard_content"] = segment["content"]
        row["gold_standard_source"] = "single_station_report_pdf"
        row["final_label"] = segment["label"]
        row["review_status"] = "done"
        normalized += 1

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Normalized {normalized} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
