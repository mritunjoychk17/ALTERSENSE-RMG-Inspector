#!/usr/bin/env python3
"""Build a station-6-focused review queue around problematic phase boundaries and long pickup bursts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDNAMES = [
    "video_id",
    "station_id",
    "station_role",
    "frame_index",
    "timestamp_sec",
    "crop_path",
    "prev_crop_path",
    "next_crop_path",
    "clip_paths",
    "clip_len",
    "pose_label",
    "pose_confidence",
    "pose_reason",
    "final_label",
    "review_status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-csv", required=True)
    parser.add_argument("--phase-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--station-id", default="6")
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--long-pickup-min-sec", type=float, default=5.0)
    return parser.parse_args()


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def add_row(output_rows: list[dict], seen: set[tuple[str, str, str]], clip_row: dict, note: str) -> None:
    key = (
        str(clip_row.get("video_id", "")).strip(),
        str(clip_row.get("station_id", "")).strip(),
        str(clip_row.get("frame_index", "")).strip(),
    )
    if key in seen:
        return
    seen.add(key)
    output_rows.append(
        {
            "video_id": clip_row.get("video_id", ""),
            "station_id": clip_row.get("station_id", ""),
            "station_role": clip_row.get("station_role", ""),
            "frame_index": clip_row.get("frame_index", ""),
            "timestamp_sec": clip_row.get("timestamp_sec", ""),
            "crop_path": clip_row.get("crop_path", ""),
            "prev_crop_path": clip_row.get("prev_crop_path", ""),
            "next_crop_path": clip_row.get("next_crop_path", ""),
            "clip_paths": clip_row.get("clip_paths", ""),
            "clip_len": clip_row.get("clip_len", ""),
            "pose_label": clip_row.get("pose_label", ""),
            "pose_confidence": clip_row.get("pose_confidence", ""),
            "pose_reason": clip_row.get("pose_reason", ""),
            "final_label": "",
            "review_status": "pending",
            "notes": note,
        }
    )


def main() -> int:
    args = parse_args()
    station_id = str(args.station_id).strip()

    clip_rows = list(csv.DictReader(open(args.clip_csv, newline="", encoding="utf-8")))
    phase_rows = list(csv.DictReader(open(args.phase_csv, newline="", encoding="utf-8")))

    clip_rows = [row for row in clip_rows if str(row.get("station_id", "")).strip() == station_id]
    phase_rows = [row for row in phase_rows if str(row.get("station_id", "")).strip() == station_id]

    if not clip_rows:
        raise ValueError("No clip rows found for requested station.")
    if not phase_rows:
        raise ValueError("No phase rows found for requested station.")

    clip_rows.sort(key=lambda row: parse_float(row.get("timestamp_sec", ""), 0.0))
    phase_rows.sort(key=lambda row: parse_float(row.get("timestamp_sec", ""), 0.0))

    output_rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for idx in range(len(phase_rows) - 1):
        cur = phase_rows[idx]
        nxt = phase_rows[idx + 1]
        cur_label = (cur.get("predicted_phase") or cur.get("smoothed_phase") or "").strip()
        next_label = (nxt.get("predicted_phase") or nxt.get("smoothed_phase") or "").strip()
        pair = (cur_label, next_label)
        if pair not in {
            ("align_phase", "pickup_phase"),
            ("pickup_phase", "sew_phase"),
            ("sew_phase", "pickup_phase"),
        }:
            continue
        center_ts = parse_float(cur.get("timestamp_sec", ""), 0.0)
        lo = center_ts - args.window_sec
        hi = center_ts + args.window_sec
        note = f"station6_boundary {cur_label}->{next_label}"
        for clip_row in clip_rows:
            ts = parse_float(clip_row.get("timestamp_sec", ""), 0.0)
            if lo <= ts <= hi:
                add_row(output_rows, seen, clip_row, note)

    run_start = 0
    while run_start < len(phase_rows):
        run_label = (phase_rows[run_start].get("predicted_phase") or phase_rows[run_start].get("smoothed_phase") or "").strip()
        run_end = run_start
        while run_end + 1 < len(phase_rows):
            next_label = (phase_rows[run_end + 1].get("predicted_phase") or phase_rows[run_end + 1].get("smoothed_phase") or "").strip()
            if next_label != run_label:
                break
            run_end += 1
        if run_label == "pickup_phase":
            start_ts = parse_float(phase_rows[run_start].get("timestamp_sec", ""), 0.0)
            end_ts = parse_float(phase_rows[run_end].get("timestamp_sec", ""), 0.0)
            if end_ts - start_ts >= args.long_pickup_min_sec:
                note = f"station6_long_pickup {start_ts:.1f}->{end_ts:.1f}"
                lo = start_ts - args.window_sec
                hi = end_ts + args.window_sec
                for clip_row in clip_rows:
                    ts = parse_float(clip_row.get("timestamp_sec", ""), 0.0)
                    if lo <= ts <= hi:
                        add_row(output_rows, seen, clip_row, note)
        run_start = run_end + 1

    output_rows.sort(
        key=lambda row: (
            row.get("video_id", ""),
            int(row.get("station_id", "0") or 0),
            parse_float(row.get("timestamp_sec", ""), 0.0),
        )
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
