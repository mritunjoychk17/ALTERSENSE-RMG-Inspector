#!/usr/bin/env python3
"""Build a Stage 2 review queue from an ROI crop manifest and ROI config."""

from __future__ import annotations

import argparse
import csv
import json
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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-per-station", type=int, default=24)
    parser.add_argument("--min-time-gap-sec", type=float, default=5.0)
    return parser.parse_args()


def load_station_map(config_path: Path, video_id: str) -> dict[str, dict]:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    video = next((item for item in data["videos"] if item["video_id"] == video_id), None)
    if video is None:
        raise ValueError(f"Unknown video_id in config: {video_id}")
    return {str(ws["station_id"]): ws for ws in video["workstations"] if ws.get("station_roi_polygon")}


def main() -> int:
    args = parse_args()
    station_map = load_station_map(Path(args.config), args.video_id)
    if not station_map:
        raise ValueError(f"No annotated stations found for {args.video_id}")

    by_station: dict[str, list[dict]] = {}
    with Path(args.manifest).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["video_id"] != args.video_id:
                continue
            station_id = str(row["station_id"])
            if station_id not in station_map:
                continue
            by_station.setdefault(station_id, []).append(row)

    output_rows: list[dict] = []
    for station_id, rows in sorted(by_station.items(), key=lambda kv: int(kv[0])):
        rows.sort(key=lambda r: int(r["frame_index"]))
        selected: list[dict] = []
        for row in rows:
            if args.max_per_station == 0:
                selected.append(row)
                continue
            ts = float(row["timestamp_sec"])
            if any(abs(ts - float(prev["timestamp_sec"])) < args.min_time_gap_sec for prev in selected):
                continue
            selected.append(row)
            if len(selected) >= args.max_per_station:
                break

        station = station_map[station_id]
        frame_to_idx = {row["frame_index"]: idx for idx, row in enumerate(rows)}
        for row in selected:
            idx = frame_to_idx[row["frame_index"]]
            prev_crop = rows[idx - 1]["crop_path"] if idx > 0 else ""
            next_crop = rows[idx + 1]["crop_path"] if idx < len(rows) - 1 else ""
            output_rows.append(
                {
                    "video_id": row["video_id"],
                    "station_id": station_id,
                    "station_role": station.get("station_role", ""),
                    "frame_index": row["frame_index"],
                    "timestamp_sec": row["timestamp_sec"],
                    "crop_path": row["crop_path"],
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
                    "notes": station.get("notes", ""),
                }
            )

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
