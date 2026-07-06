#!/usr/bin/env python3
"""Build a second-pass dense Gemini queue for weak stations around transition and long-sew regions."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
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
    "presence_confidence",
    "pose_label",
    "pose_confidence",
    "pose_reason",
    "gemini_label",
    "gemini_confidence",
    "gemini_reason",
    "gemini_protocol_version",
    "gemini_cycle_phase",
    "gemini_motion_direction",
    "gemini_machine_engaged",
    "gemini_hands_on_material",
    "gemini_transition_ok",
    "gemini_safe_label",
    "gemini_schema_error",
    "gemini_json",
    "final_label",
    "review_status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse-csv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stations", default="4,6")
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--long-sew-min-sec", type=float, default=12.0)
    return parser.parse_args()


def load_manifest(path: Path):
    by_station: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_station[(row["video_id"], row["station_id"])].append(row)
    for rows in by_station.values():
        rows.sort(key=lambda r: float(r["timestamp_sec"]))
    return by_station


def compress(rows: list[dict]) -> list[tuple[int, int, str]]:
    if not rows:
        return []
    labels = [(r.get("hybrid_postprocessed_label") or r.get("activity_label") or "").strip() for r in rows]
    out = []
    start = 0
    current = labels[0]
    for idx, label in enumerate(labels[1:], start=1):
        if label != current:
            out.append((start, idx - 1, current))
            start = idx
            current = label
    out.append((start, len(labels) - 1, current))
    return out


def add_window(output_rows, seen, dense_rows, center_idx, window_sec, note, station_role):
    center_ts = float(dense_rows[center_idx]["timestamp_sec"])
    lo = center_ts - window_sec
    hi = center_ts + window_sec
    for j, dense in enumerate(dense_rows):
        ts = float(dense["timestamp_sec"])
        if ts < lo or ts > hi:
            continue
        key = (dense["video_id"], dense["station_id"], dense["frame_index"])
        if key in seen:
            continue
        seen.add(key)
        prev_crop = dense_rows[j - 1]["crop_path"] if j > 0 else ""
        next_crop = dense_rows[j + 1]["crop_path"] if j < len(dense_rows) - 1 else ""
        output_rows.append(
            {
                "video_id": dense["video_id"],
                "station_id": dense["station_id"],
                "station_role": station_role,
                "frame_index": dense["frame_index"],
                "timestamp_sec": dense["timestamp_sec"],
                "crop_path": dense["crop_path"],
                "prev_crop_path": prev_crop,
                "next_crop_path": next_crop,
                "presence_confidence": "",
                "pose_label": "",
                "pose_confidence": "",
                "pose_reason": "",
                "gemini_label": "",
                "gemini_confidence": "",
                "gemini_reason": "",
                "gemini_protocol_version": "",
                "gemini_cycle_phase": "",
                "gemini_motion_direction": "",
                "gemini_machine_engaged": "",
                "gemini_hands_on_material": "",
                "gemini_transition_ok": "",
                "gemini_safe_label": "",
                "gemini_schema_error": "",
                "gemini_json": "",
                "final_label": "",
                "review_status": "pending",
                "notes": note,
            }
        )


def main() -> int:
    args = parse_args()
    target_stations = {item.strip() for item in args.stations.split(",") if item.strip()}
    sparse_rows = list(csv.DictReader(open(args.sparse_csv, newline="", encoding="utf-8")))
    manifest_by_station = load_manifest(Path(args.manifest))

    sparse_by_station: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in sparse_rows:
        if row.get("station_id") not in target_stations:
            continue
        sparse_by_station[(row["video_id"], row["station_id"])].append(row)

    output_rows = []
    seen = set()
    for station_key, rows in sparse_by_station.items():
        rows.sort(key=lambda r: float(r["timestamp_sec"]))
        dense_rows = manifest_by_station.get(station_key, [])
        if not dense_rows:
            continue
        station_role = rows[0].get("station_role", "")
        dense_index = {str(r["timestamp_sec"]): idx for idx, r in enumerate(dense_rows)}

        # Transition neighborhoods from sparse labels.
        for idx in range(len(rows) - 1):
            cur = rows[idx]
            nxt = rows[idx + 1]
            cur_label = (cur.get("hybrid_postprocessed_label") or cur.get("activity_label") or "").strip()
            next_label = (nxt.get("hybrid_postprocessed_label") or nxt.get("activity_label") or "").strip()
            pair = (cur_label, next_label)
            if pair not in {("sew", "align"), ("align", "sew"), ("sew", "idle")}:
                continue
            center_idx = dense_index.get(str(cur["timestamp_sec"]))
            if center_idx is None:
                continue
            add_window(
                output_rows,
                seen,
                dense_rows,
                center_idx,
                args.window_sec,
                f"weak_station_boundary sparse_pair={cur_label}->{next_label}",
                station_role,
            )

        # Long sew bursts from sparse labels.
        segments = compress(rows)
        for start, end, label in segments:
            if label != "sew":
                continue
            start_ts = float(rows[start]["timestamp_sec"])
            end_ts = float(rows[end]["timestamp_sec"])
            if end_ts - start_ts < args.long_sew_min_sec:
                continue
            for sparse_idx in [start, end]:
                center_idx = dense_index.get(str(rows[sparse_idx]["timestamp_sec"]))
                if center_idx is None:
                    continue
                add_window(
                    output_rows,
                    seen,
                    dense_rows,
                    center_idx,
                    args.window_sec,
                    f"weak_station_long_sew burst={start_ts:.1f}->{end_ts:.1f}",
                    station_role,
                )

    output_rows.sort(key=lambda r: (r["video_id"], int(r["station_id"]), float(r["timestamp_sec"])))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
