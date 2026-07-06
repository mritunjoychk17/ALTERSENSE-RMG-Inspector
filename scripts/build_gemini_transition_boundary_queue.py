#!/usr/bin/env python3
"""Build a dense 1-second Gemini queue around sparse align/sew boundaries."""

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
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--stations", default="1,2,3,4,5,6")
    return parser.parse_args()


def load_manifest(path: Path):
    by_station: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_key: dict[tuple[str, str, str], dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["video_id"], row["station_id"], row["frame_index"])
            by_key[key] = row
            by_station[(row["video_id"], row["station_id"])].append(row)
    for rows in by_station.values():
        rows.sort(key=lambda r: float(r["timestamp_sec"]))
    return by_station, by_key


def main() -> int:
    args = parse_args()
    target_stations = {item.strip() for item in args.stations.split(",") if item.strip()}
    sparse_rows = list(csv.DictReader(open(args.sparse_csv, newline="", encoding="utf-8")))
    manifest_by_station, _ = load_manifest(Path(args.manifest))

    sparse_by_station: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in sparse_rows:
        if row.get("station_id") not in target_stations:
            continue
        sparse_by_station[(row["video_id"], row["station_id"])].append(row)

    output_rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for station_key, rows in sparse_by_station.items():
        rows.sort(key=lambda r: float(r["timestamp_sec"]))
        dense_rows = manifest_by_station.get(station_key, [])
        if not dense_rows:
            continue
        for idx in range(len(rows) - 1):
            cur = rows[idx]
            nxt = rows[idx + 1]
            cur_label = (cur.get("hybrid_postprocessed_label") or cur.get("activity_label") or "").strip()
            next_label = (nxt.get("hybrid_postprocessed_label") or nxt.get("activity_label") or "").strip()
            pair = {cur_label, next_label}
            if pair != {"align", "sew"}:
                continue
            t0 = float(cur["timestamp_sec"])
            t1 = float(nxt["timestamp_sec"])
            lo = min(t0, t1) - args.window_sec
            hi = max(t0, t1) + args.window_sec
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
                        "station_role": cur.get("station_role", ""),
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
                        "notes": f"boundary_focus sparse_pair={cur_label}->{next_label} sparse_ts={t0:.1f},{t1:.1f}",
                    }
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
