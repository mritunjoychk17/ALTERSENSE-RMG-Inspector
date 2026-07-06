#!/usr/bin/env python3
"""Build a Gemini labeling queue from a clip manifest."""

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
    parser.add_argument("--clip-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--notes-prefix", default="clip_window")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = list(csv.DictReader(open(args.clip_csv, newline="", encoding="utf-8")))
    if not rows:
        raise ValueError("Clip CSV is empty.")

    output_rows = []
    for row in rows:
        clip_paths = (row.get("clip_paths") or "").strip()
        if not clip_paths:
            continue
        clip_parts = [part for part in clip_paths.split("|") if part]
        if not clip_parts:
            continue
        center_idx = len(clip_parts) // 2
        center_crop = clip_parts[center_idx]
        prev_crop = clip_parts[center_idx - 1] if center_idx > 0 else ""
        next_crop = clip_parts[center_idx + 1] if center_idx < len(clip_parts) - 1 else ""
        output_rows.append(
            {
                "video_id": row.get("video_id", ""),
                "station_id": row.get("station_id", ""),
                "station_role": row.get("station_role", ""),
                "frame_index": row.get("frame_index", ""),
                "timestamp_sec": row.get("timestamp_sec", ""),
                "crop_path": center_crop,
                "prev_crop_path": prev_crop,
                "next_crop_path": next_crop,
                "clip_paths": clip_paths,
                "clip_len": row.get("clip_len", str(len(clip_parts))),
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
                "notes": f"{args.notes_prefix} len={len(clip_parts)}",
            }
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
