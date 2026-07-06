#!/usr/bin/env python3
"""Compute working time, NPT, cycle count, and cycle times from activity predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="CSV with timestamp_sec and predicted_label.")
    parser.add_argument("--config", default="configs/activity_pipeline.json")
    parser.add_argument("--output", default="artifacts/stage2/eval/activity_cycle_metrics.json")
    return parser.parse_args()


def compress_segments(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    rows = sorted(rows, key=lambda x: float(x["timestamp_sec"]))
    segments = []
    current = {
        "label": rows[0].get("smoothed_label") or rows[0]["predicted_label"],
        "start_sec": float(rows[0]["timestamp_sec"]),
        "end_sec": float(rows[0]["timestamp_sec"]),
    }
    for row in rows[1:]:
        ts = float(row["timestamp_sec"])
        label = row.get("smoothed_label") or row["predicted_label"]
        if label == current["label"]:
            current["end_sec"] = ts
        else:
            segments.append(current)
            current = {"label": label, "start_sec": ts, "end_sec": ts}
    segments.append(current)
    return segments


def main() -> int:
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cycle_cfg = cfg["cycle_logic"]
    cycle_sequence = cycle_cfg["cycle_sequence"]
    npt_labels = set(cycle_cfg["npt_labels"])
    working_labels = set(cycle_cfg["working_labels"])
    max_gap = float(cycle_cfg["max_gap_between_steps_sec"])

    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Prediction CSV not found: {predictions_path}\n"
            "Generate Stage 2 activity predictions first, then pass that CSV here."
        )

    with predictions_path.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("timestamp_sec") not in {"", None}]
    if not rows:
        raise ValueError(
            f"No timestamped activity rows found in {predictions_path}. "
            "The input CSV must contain at least 'timestamp_sec' and 'predicted_label'."
        )
    segments = compress_segments(rows)

    total_npt = 0.0
    total_work = 0.0
    for seg in segments:
        duration = max(0.0, seg["end_sec"] - seg["start_sec"])
        if seg["label"] in npt_labels:
            total_npt += duration
        if seg["label"] in working_labels:
            total_work += duration

    cycles = []
    seq_index = 0
    cycle_start = None
    last_step_time = None
    for seg in segments:
        label = seg["label"]
        if label != cycle_sequence[seq_index]:
            if label == cycle_sequence[0]:
                seq_index = 1
                cycle_start = seg["start_sec"]
                last_step_time = seg["end_sec"]
            continue

        if seq_index == 0:
            cycle_start = seg["start_sec"]
            last_step_time = seg["end_sec"]
            seq_index = 1
            continue

        if last_step_time is not None and seg["start_sec"] - last_step_time > max_gap:
            seq_index = 0
            cycle_start = None
            last_step_time = None
            if label == cycle_sequence[0]:
                cycle_start = seg["start_sec"]
                last_step_time = seg["end_sec"]
                seq_index = 1
            continue

        last_step_time = seg["end_sec"]
        seq_index += 1
        if seq_index == len(cycle_sequence):
            cycles.append(
                {
                    "start_sec": cycle_start,
                    "end_sec": seg["end_sec"],
                    "duration_sec": round(seg["end_sec"] - cycle_start, 3) if cycle_start is not None else None,
                }
            )
            seq_index = 0
            cycle_start = None
            last_step_time = None

    result = {
        "input_csv": args.predictions,
        "labels_seen": sorted({seg["label"] for seg in segments}),
        "total_segments": len(segments),
        "npt_duration_sec": round(total_npt, 3),
        "working_duration_sec": round(total_work, 3),
        "cycle_count": len(cycles),
        "average_cycle_time_sec": round(sum(c["duration_sec"] for c in cycles) / len(cycles), 3) if cycles else 0.0,
        "cycles": cycles,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
