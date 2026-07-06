#!/usr/bin/env python3
"""Build a Stage 2 review queue from present station crops."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="datasets/interim/roi_crops/manifest.csv")
    parser.add_argument("--presence-predictions", required=True, help="CSV, directory, or glob from Stage 1 video/station predictions.")
    parser.add_argument("--output", default="datasets/processed/stage2/manifests/activity_review_queue.csv")
    parser.add_argument("--present-threshold", type=float, default=0.5)
    parser.add_argument("--sample-every-present", type=int, default=5, help="Keep every Nth present row to reduce queue size.")
    parser.add_argument("--max-per-station", type=int, default=0, help="Optional cap on queued rows per station after sampling. 0 keeps all sampled rows.")
    parser.add_argument("--video-id")
    parser.add_argument("--station-id")
    return parser.parse_args()


def resolve_prediction_files(value: str) -> list[Path]:
    path = Path(value)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.glob("*.csv") if p.is_file())
    matches = sorted(Path(".").glob(value))
    return [p for p in matches if p.is_file()]


def main() -> int:
    args = parse_args()
    manifest_rows = {}
    station_frame_rows: dict[tuple[str, str], list[dict]] = {}
    with open(args.manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            manifest_rows[(row["video_id"], row["station_id"], row["frame_index"])] = row
            station_frame_rows.setdefault((row["video_id"], row["station_id"]), []).append(row)
    for rows in station_frame_rows.values():
        rows.sort(key=lambda item: int(item["frame_index"]))

    prediction_files = resolve_prediction_files(args.presence_predictions)
    if not prediction_files:
        raise FileNotFoundError(f"No presence prediction CSV files found for: {args.presence_predictions}")

    output_rows = []
    seen_keys: set[tuple[str, str, str]] = set()
    kept = 0
    station_counts: dict[tuple[str, str], int] = {}
    for prediction_file in prediction_files:
        with prediction_file.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "source" not in reader.fieldnames:
                continue
            for row in reader:
                source = row["source"]
                if ":station_" not in source:
                    continue
                video_id, station_part = source.split(":station_")
                station_id = station_part
                if args.video_id and video_id != args.video_id:
                    continue
                if args.station_id and station_id != args.station_id:
                    continue
                present_conf = float(row.get("present_confidence") or 0.0)
                if present_conf < args.present_threshold:
                    continue
                key = (video_id, station_id, row["frame_index"])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                if kept % args.sample_every_present != 0:
                    kept += 1
                    continue
                manifest_row = manifest_rows.get(key)
                kept += 1
                if not manifest_row:
                    continue
                station_key = (video_id, station_id)
                if args.max_per_station and station_counts.get(station_key, 0) >= args.max_per_station:
                    continue
                station_rows = station_frame_rows.get((video_id, station_id), [])
                idx = next((i for i, item in enumerate(station_rows) if item["frame_index"] == row["frame_index"]), -1)
                prev_crop = station_rows[idx - 1]["crop_path"] if idx > 0 else ""
                next_crop = station_rows[idx + 1]["crop_path"] if 0 <= idx < len(station_rows) - 1 else ""
                output_rows.append(
                    {
                        "video_id": video_id,
                        "station_id": station_id,
                        "frame_index": row["frame_index"],
                        "timestamp_sec": manifest_row["timestamp_sec"],
                        "crop_path": manifest_row["crop_path"],
                        "prev_crop_path": prev_crop,
                        "next_crop_path": next_crop,
                        "presence_confidence": row.get("present_confidence", ""),
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
                        "notes": "",
                    }
                )
                station_counts[station_key] = station_counts.get(station_key, 0) + 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "station_id",
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
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
