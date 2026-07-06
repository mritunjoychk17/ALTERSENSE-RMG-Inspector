#!/usr/bin/env python3
"""Build a denser station-4-only manual review queue around weak phase boundaries."""

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
    "predicted_label",
    "smoothed_label",
    "phase_label",
    "clip_validation_note",
    "review_focus",
    "final_label",
    "review_status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--clip-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--station-id", default="4")
    parser.add_argument("--boundary-window", type=int, default=3)
    parser.add_argument("--long-sew-min-sec", type=float, default=12.0)
    parser.add_argument("--long-sew-stride", type=int, default=8)
    parser.add_argument("--max-long-sew-samples", type=int, default=60)
    return parser.parse_args()


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add_index(indexes: set[int], start: int, end: int, center: int, window: int) -> None:
    lo = max(start, center - window)
    hi = min(end, center + window)
    for idx in range(lo, hi + 1):
        indexes.add(idx)


def main() -> int:
    args = parse_args()
    station_id = str(args.station_id).strip()
    pred_rows = [
        row for row in read_rows(Path(args.predictions))
        if str(row.get("station_id", "")).strip() == station_id
    ]
    clip_rows = [
        row for row in read_rows(Path(args.clip_csv))
        if str(row.get("station_id", "")).strip() == station_id
    ]
    if not pred_rows:
        raise ValueError("No prediction rows found for the requested station.")
    if not clip_rows:
        raise ValueError("No clip rows found for the requested station.")

    pred_rows.sort(key=lambda row: parse_float(row.get("timestamp_sec", ""), 0.0))
    clip_rows.sort(key=lambda row: parse_float(row.get("timestamp_sec", ""), 0.0))
    pred_by_frame = {str(row.get("frame_index", "")).strip(): row for row in pred_rows}
    clip_by_frame = {str(row.get("frame_index", "")).strip(): row for row in clip_rows}

    selected_indexes: set[int] = set()
    long_sew_samples = 0
    run_start = 0

    for idx in range(len(pred_rows) + 1):
        boundary = idx == len(pred_rows) or pred_rows[idx].get("predicted_phase") != pred_rows[run_start].get("predicted_phase")
        if not boundary:
            continue

        label = pred_rows[run_start].get("predicted_phase", "")
        run_end = idx - 1
        start_ts = parse_float(pred_rows[run_start].get("timestamp_sec", ""), 0.0)
        end_ts = parse_float(pred_rows[run_end].get("timestamp_sec", ""), 0.0)
        duration = end_ts - start_ts + 1.0

        prev_label = pred_rows[run_start - 1].get("predicted_phase", "") if run_start > 0 else ""
        next_label = pred_rows[idx].get("predicted_phase", "") if idx < len(pred_rows) else ""

        if label == "sew_phase" and duration >= args.long_sew_min_sec:
            add_index(selected_indexes, 0, len(pred_rows) - 1, run_start, args.boundary_window)
            add_index(selected_indexes, 0, len(pred_rows) - 1, run_end, args.boundary_window)
            pos = run_start + args.long_sew_stride
            while pos < run_end and long_sew_samples < args.max_long_sew_samples:
                selected_indexes.add(pos)
                long_sew_samples += 1
                pos += args.long_sew_stride

        if label in {"align_phase", "idle_phase"} and duration <= 6.0 and (prev_label == "sew_phase" or next_label == "sew_phase"):
            for center in range(run_start, run_end + 1):
                add_index(selected_indexes, 0, len(pred_rows) - 1, center, args.boundary_window)

        if label in {"pickup_phase", "place_phase"}:
            for center in range(run_start, run_end + 1):
                add_index(selected_indexes, 0, len(pred_rows) - 1, center, args.boundary_window)

        run_start = idx

    out_rows: list[dict] = []
    selected_frames = []
    for idx in sorted(selected_indexes):
        if 0 <= idx < len(pred_rows):
            frame_index = str(pred_rows[idx].get("frame_index", "")).strip()
            if frame_index:
                selected_frames.append(frame_index)

    seen_frames: set[str] = set()
    for frame_index in selected_frames:
        if frame_index in seen_frames:
            continue
        seen_frames.add(frame_index)
        clip_row = clip_by_frame.get(frame_index)
        pred_row = pred_by_frame.get(frame_index, {})
        if not clip_row:
            continue
        clip_paths = [item.strip() for item in (clip_row.get("clip_paths") or "").split("|") if item.strip()]
        center = len(clip_paths) // 2
        crop_path = clip_paths[center] if clip_paths else ""
        prev_crop = clip_paths[max(0, center - 1)] if clip_paths and center > 0 else ""
        next_crop = clip_paths[min(len(clip_paths) - 1, center + 1)] if clip_paths and center + 1 < len(clip_paths) else ""
        out_rows.append(
            {
                "video_id": clip_row.get("video_id", ""),
                "station_id": clip_row.get("station_id", ""),
                "station_role": clip_row.get("station_role", ""),
                "frame_index": clip_row.get("frame_index", ""),
                "timestamp_sec": clip_row.get("timestamp_sec", ""),
                "crop_path": crop_path,
                "prev_crop_path": prev_crop,
                "next_crop_path": next_crop,
                "clip_paths": clip_row.get("clip_paths", ""),
                "clip_len": clip_row.get("clip_len", ""),
                "pose_label": clip_row.get("pose_label", ""),
                "pose_confidence": clip_row.get("pose_confidence", ""),
                "predicted_label": pred_row.get("predicted_phase", ""),
                "smoothed_label": pred_row.get("smoothed_phase", ""),
                "phase_label": pred_row.get("predicted_phase", ""),
                "clip_validation_note": "",
                "review_focus": "station4_dense_boundary",
                "final_label": "",
                "review_status": "pending",
                "notes": f"station4_dense_boundary predicted={pred_row.get('predicted_phase', '')}",
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
