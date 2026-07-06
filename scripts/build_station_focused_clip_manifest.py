#!/usr/bin/env python3
"""Build a station-focused Stage 2 clip manifest with transition emphasis."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--stations", default="1,2,5,6")
    parser.add_argument("--label-column", default="clip_validated_label")
    parser.add_argument("--fallback-label-columns", default="hybrid_postprocessed_label,smoothed_label,predicted_label")
    parser.add_argument("--transition-window", type=int, default=2)
    parser.add_argument("--transition-repeat", type=int, default=3)
    return parser.parse_args()


def choose_label(row: dict, primary: str, fallbacks: list[str]) -> str:
    for key in [primary, *fallbacks]:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def main() -> int:
    args = parse_args()
    stations = {item.strip() for item in args.stations.split(",") if item.strip()}
    fallback_columns = [item.strip() for item in args.fallback_label_columns.split(",") if item.strip()]
    rows = list(csv.DictReader(Path(args.input_csv).open(newline="", encoding="utf-8")))
    grouped: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        station_id = str(row.get("station_id", "")).strip()
        if station_id not in stations:
            continue
        label = choose_label(row, args.label_column, fallback_columns)
        if not label:
            continue
        clone = dict(row)
        clone["label"] = label
        clone["pose_label"] = (row.get("gemini_safe_label") or row.get("smoothed_label") or label or "").strip()
        clone["pose_confidence"] = row.get("confidence", "") or "0.0"
        grouped[station_id].append(clone)

    out_rows: list[dict] = []
    for station_id, station_rows in grouped.items():
        station_rows.sort(key=lambda row: float(row.get("timestamp_sec") or 0.0))
        transition_indices: set[int] = set()
        for idx in range(1, len(station_rows)):
            prev_label = station_rows[idx - 1]["label"]
            curr_label = station_rows[idx]["label"]
            if prev_label != curr_label:
                left = max(0, idx - args.transition_window)
                right = min(len(station_rows), idx + args.transition_window + 1)
                transition_indices.update(range(left, right))

        for idx, row in enumerate(station_rows):
            repeat = args.transition_repeat if idx in transition_indices else 1
            for rep in range(repeat):
                item = dict(row)
                item["sample_weight"] = str(repeat)
                item["transition_boosted"] = "1" if idx in transition_indices else "0"
                item["replica_index"] = str(rep)
                out_rows.append(item)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    counts = Counter((row["station_id"], row["label"]) for row in out_rows)
    print(f"Wrote {len(out_rows)} rows to {output_path}")
    for (station_id, label), count in sorted(counts.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        print(f"station={station_id} label={label} count={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
