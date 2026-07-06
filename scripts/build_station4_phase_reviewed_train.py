#!/usr/bin/env python3
"""Convert reviewed station-4 relabel queue into a phase training manifest."""

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
    "label",
    "clip_len",
    "clip_paths",
    "predicted_label",
    "smoothed_label",
    "confidence",
    "align_confidence",
    "get_confidence",
    "idle_confidence",
    "put_confidence",
    "sew_confidence",
    "postprocessed_label",
    "postprocess_note",
    "gemini_safe_label",
    "gemini_cycle_phase",
    "gemini_motion_direction",
    "gemini_machine_engaged",
    "gemini_hands_on_material",
    "hybrid_postprocessed_label",
    "hybrid_postprocess_note",
    "clip_validated_label",
    "clip_validation_note",
    "phase_label",
    "source_activity_label",
    "pose_label",
    "pose_confidence",
    "sample_weight",
    "transition_boosted",
    "replica_index",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--sample-weight", default="4")
    parser.add_argument("--transition-boosted", default="1")
    parser.add_argument("--station-role-override", default="")
    return parser.parse_args()


def to_phase(label: str) -> str:
    mapping = {
        "align": "align_phase",
        "put": "place_phase",
        "get": "pickup_phase",
        "sew": "sew_phase",
        "idle": "idle_phase",
    }
    return mapping.get(label.strip().lower(), "")


def main() -> int:
    args = parse_args()
    review_path = Path(args.review_csv)
    rows = list(csv.DictReader(review_path.open(newline="", encoding="utf-8")))
    out_rows: list[dict] = []
    station_role_override = (args.station_role_override or "").strip()

    for row in rows:
        if (row.get("review_status") or "").strip() != "done":
            continue
        final_label = (row.get("final_label") or "").strip().lower()
        phase_label = to_phase(final_label)
        if not phase_label:
            continue
        out_rows.append(
            {
                "video_id": row.get("video_id", ""),
                "station_id": row.get("station_id", ""),
                "station_role": station_role_override or row.get("station_role", ""),
                "frame_index": row.get("frame_index", ""),
                "timestamp_sec": row.get("timestamp_sec", ""),
                "label": phase_label,
                "clip_len": row.get("clip_len", ""),
                "clip_paths": row.get("clip_paths", ""),
                "predicted_label": row.get("predicted_label", ""),
                "smoothed_label": row.get("smoothed_label", ""),
                "confidence": "",
                "align_confidence": "",
                "get_confidence": "",
                "idle_confidence": "",
                "put_confidence": "",
                "sew_confidence": "",
                "postprocessed_label": final_label,
                "postprocess_note": "station4_manual_review",
                "gemini_safe_label": "",
                "gemini_cycle_phase": "",
                "gemini_motion_direction": "",
                "gemini_machine_engaged": "",
                "gemini_hands_on_material": "",
                "hybrid_postprocessed_label": final_label,
                "hybrid_postprocess_note": "station4_manual_review",
                "clip_validated_label": final_label,
                "clip_validation_note": row.get("clip_validation_note", "") or "station4_manual_review",
                "phase_label": phase_label,
                "source_activity_label": final_label,
                "pose_label": row.get("pose_label", ""),
                "pose_confidence": row.get("pose_confidence", ""),
                "sample_weight": args.sample_weight,
                "transition_boosted": args.transition_boosted,
                "replica_index": "0",
            }
        )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
