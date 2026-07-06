#!/usr/bin/env python3
"""Propagate sparse reviewed Stage 2 labels onto dense ROI manifests by nearest timestamp."""

from __future__ import annotations

import argparse
import csv
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", required=True, help="Comma-separated reviewed CSV files.")
    parser.add_argument("--dense-manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--label-column", default="final_label")
    parser.add_argument("--accepted-statuses", default="done,reviewed,approved")
    parser.add_argument("--max-gap-sec", type=float, default=2.0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    args = parse_args()
    accepted = {item.strip() for item in args.accepted_statuses.split(",") if item.strip()}
    review_rows: list[dict] = []
    for item in args.review_csv.split(","):
        item = item.strip()
        if item:
            review_rows.extend(read_csv(Path(item)))
    dense_rows = read_csv(Path(args.dense_manifest))

    sparse_by_station: dict[tuple[str, str], list[tuple[float, dict]]] = defaultdict(list)
    for row in review_rows:
        status = (row.get("review_status") or "").strip()
        if accepted and status and status not in accepted:
            continue
        label = (row.get(args.label_column) or row.get("label") or "").strip()
        if not label:
            continue
        key = (row.get("video_id", ""), str(row.get("station_id", "")).strip())
        sparse_by_station[key].append((float(row.get("timestamp_sec") or 0.0), row))
    for key in sparse_by_station:
        sparse_by_station[key].sort(key=lambda item: item[0])

    output_rows = []
    for row in dense_rows:
        key = (row.get("video_id", ""), str(row.get("station_id", "")).strip())
        sparse = sparse_by_station.get(key)
        if not sparse:
            continue
        ts = float(row.get("timestamp_sec") or 0.0)
        timestamps = [item[0] for item in sparse]
        idx = bisect_left(timestamps, ts)
        candidates = []
        if idx < len(sparse):
            candidates.append(sparse[idx])
        if idx > 0:
            candidates.append(sparse[idx - 1])
        if not candidates:
            continue
        nearest_ts, nearest_row = min(candidates, key=lambda item: abs(item[0] - ts))
        if abs(nearest_ts - ts) > args.max_gap_sec:
            continue
        clone = dict(row)
        clone["label"] = nearest_row.get(args.label_column) or nearest_row.get("label") or ""
        clone["label_source_timestamp_sec"] = nearest_row.get("timestamp_sec", "")
        clone["label_source_frame_index"] = nearest_row.get("frame_index", "")
        clone["label_gap_sec"] = round(abs(nearest_ts - ts), 3)
        clone["station_role"] = nearest_row.get("station_role", row.get("station_role", ""))
        output_rows.append(clone)

    if not output_rows:
        raise ValueError("No dense labels were propagated. Check station IDs, timestamps, or max-gap-sec.")
    fieldnames = list(output_rows[0].keys())
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} dense labeled rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
