#!/usr/bin/env python3
"""Build a transition-focused Gemini clip queue from an existing clip manifest."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-queue-csv", required=True)
    parser.add_argument("--reference-csv", required=True, help="Frame-based activity or hybrid CSV used to find transition centers.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--label-column", default="hybrid_postprocessed_label")
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--stations", default="1,2,3,4,5,6")
    return parser.parse_args()


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> int:
    args = parse_args()
    target_stations = {item.strip() for item in args.stations.split(",") if item.strip()}
    clip_rows = list(csv.DictReader(open(args.clip_queue_csv, newline="", encoding="utf-8")))
    ref_rows = list(csv.DictReader(open(args.reference_csv, newline="", encoding="utf-8")))
    if not clip_rows:
        raise ValueError("Clip queue CSV is empty.")
    if not ref_rows:
        raise ValueError("Reference CSV is empty.")

    ref_by_station: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in ref_rows:
        sid = str(row.get("station_id", "")).strip()
        if sid not in target_stations:
            continue
        ref_by_station[(row.get("video_id", ""), sid)].append(row)
    for rows in ref_by_station.values():
        rows.sort(key=lambda r: parse_float(r.get("timestamp_sec", "")))

    candidate_windows: dict[tuple[str, str], list[tuple[float, float, str]]] = defaultdict(list)
    for key, rows in ref_by_station.items():
        for idx in range(len(rows) - 1):
            cur = rows[idx]
            nxt = rows[idx + 1]
            cur_label = (cur.get(args.label_column) or cur.get("activity_label") or "").strip()
            next_label = (nxt.get(args.label_column) or nxt.get("activity_label") or "").strip()
            pair = (cur_label, next_label)
            if pair not in {("sew", "align"), ("align", "sew"), ("sew", "idle"), ("get", "sew"), ("put", "sew")}:
                continue
            t0 = parse_float(cur.get("timestamp_sec", ""))
            t1 = parse_float(nxt.get("timestamp_sec", ""))
            lo = min(t0, t1) - args.window_sec
            hi = max(t0, t1) + args.window_sec
            candidate_windows[key].append((lo, hi, f"clip_transition {cur_label}->{next_label}"))

    output_rows = []
    seen = set()
    for row in clip_rows:
        sid = str(row.get("station_id", "")).strip()
        vid = row.get("video_id", "")
        if sid not in target_stations:
            continue
        key = (vid, sid)
        ts = parse_float(row.get("timestamp_sec", ""))
        matched_notes = []
        for lo, hi, note in candidate_windows.get(key, []):
            if lo <= ts <= hi:
                matched_notes.append(note)
        if not matched_notes:
            continue
        dedupe_key = (vid, sid, row.get("frame_index", ""))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        clone = dict(row)
        clone["notes"] = "; ".join(sorted(set(matched_notes)))
        output_rows.append(clone)

    output_rows.sort(key=lambda r: (r.get("video_id", ""), int(r.get("station_id", "0") or 0), parse_float(r.get("timestamp_sec", ""))))
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()) if output_rows else list(clip_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
