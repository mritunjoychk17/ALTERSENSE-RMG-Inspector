#!/usr/bin/env python3
"""Convert Stage 2 review queue rows into a cycle-safe activity CSV using Gemini structured outputs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage2_taxonomy import (
    GEMINI_PROTOCOL_VERSION,
    accepted_statuses,
    default_cycle_phase_for_label,
    infer_station_role,
    normalize_cycle_phase_for_row,
    normalize_label_for_row,
    phase_matches_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--min-gemini-confidence", type=float, default=0.75)
    parser.add_argument("--allow-reviewed-override", action="store_true")
    return parser.parse_args()


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_label_from_gemini(row: dict, min_conf: float) -> tuple[str, str]:
    label = normalize_label_for_row(row.get("gemini_safe_label") or row.get("gemini_label") or "", row)
    confidence = parse_float(row.get("gemini_confidence", ""), 0.0)
    cycle_phase = normalize_cycle_phase_for_row(row.get("gemini_cycle_phase", ""), row, label=label)
    machine_engaged = parse_bool(row.get("gemini_machine_engaged"), default=(label in {"sew", "adjust_machine"}))
    hands_on_material = parse_bool(row.get("gemini_hands_on_material"), default=(label not in {"idle", "uncertain"}))
    transition_ok = parse_bool(row.get("gemini_transition_ok"), default=True)
    protocol_version = (row.get("gemini_protocol_version") or "").strip()
    schema_error = (row.get("gemini_schema_error") or "").strip()

    reason = "gemini_safe"
    if confidence < min_conf:
        return "uncertain", "low_confidence"
    if protocol_version and protocol_version != GEMINI_PROTOCOL_VERSION:
        return "uncertain", "protocol_version_mismatch"
    if schema_error:
        return normalize_label_for_row(label, row), f"schema_adjusted:{schema_error}"
    if not phase_matches_label(label, cycle_phase, row):
        return "uncertain", "phase_label_mismatch"
    if label in {"sew", "adjust_machine"} and not machine_engaged:
        return "align", "machine_not_engaged"
    if label in {"put", "pass", "get"} and not hands_on_material:
        return "uncertain", "material_contact_missing"
    if not transition_ok and confidence < 0.9:
        return "uncertain", "transition_not_ok"
    return label, reason


def apply_station_long_sew_recovery(rows: list[dict]) -> list[dict]:
    by_station: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("video_id", "")).strip(),
            str(row.get("station_id", "")).strip(),
        )
        by_station[key].append(row)

    for station_rows in by_station.values():
        station_rows.sort(key=lambda r: parse_float(r.get("timestamp_sec", ""), 0.0))
        station_id = str(station_rows[0].get("station_id", "")).strip() if station_rows else ""
        if station_id != "6":
            continue

        idx = 0
        while idx < len(station_rows):
            row = station_rows[idx]
            label = (row.get("gemini_recovered_label") or "").strip()
            phase = (row.get("gemini_recovered_phase") or "").strip()
            if label != "sew" or phase != "sew_phase":
                idx += 1
                continue

            end = idx
            while end + 1 < len(station_rows):
                nxt = station_rows[end + 1]
                if (nxt.get("gemini_recovered_label") or "").strip() != "sew":
                    break
                if (nxt.get("gemini_recovered_phase") or "").strip() != "sew_phase":
                    break
                end += 1

            segment = station_rows[idx : end + 1]
            if len(segment) >= 8:
                start_ts = parse_float(segment[0].get("timestamp_sec", ""), 0.0)
                end_ts = parse_float(segment[-1].get("timestamp_sec", ""), start_ts)
                duration = end_ts - start_ts

                if duration >= 16.0:
                    first = segment[0]
                    if (first.get("gemini_recovered_label") or "") == "sew":
                        first["gemini_recovered_label"] = "put"
                        first["gemini_recovered_phase"] = "place_phase"
                        first["gemini_recovery_note"] = "station6_long_sew_entry_put"

                    cadence = 17.0
                    next_split_ts = start_ts + cadence
                    while next_split_ts < end_ts - 1.0:
                        split_idx = min(
                            range(idx, end + 1),
                            key=lambda i: abs(parse_float(station_rows[i].get("timestamp_sec", ""), 0.0) - next_split_ts),
                        )
                        get_row = station_rows[split_idx]
                        if (get_row.get("gemini_recovered_label") or "") == "sew":
                            get_row["gemini_recovered_label"] = "get"
                            get_row["gemini_recovered_phase"] = "pickup_phase"
                            get_row["gemini_recovery_note"] = "station6_long_sew_split_get"
                        if split_idx + 1 <= end:
                            put_row = station_rows[split_idx + 1]
                            if (put_row.get("gemini_recovered_label") or "") == "sew":
                                put_row["gemini_recovered_label"] = "put"
                                put_row["gemini_recovered_phase"] = "place_phase"
                                put_row["gemini_recovery_note"] = "station6_long_sew_split_put"
                        next_split_ts += cadence

                    if end + 1 < len(station_rows):
                        next_label = (station_rows[end + 1].get("gemini_recovered_label") or "").strip()
                        next_motion = (station_rows[end + 1].get("gemini_motion_direction") or "").strip()
                        if next_label in {"align", "idle", "uncertain"} or next_motion == "toward_worker":
                            tail = station_rows[end]
                            if (tail.get("gemini_recovered_label") or "") == "sew":
                                tail["gemini_recovered_label"] = "get"
                                tail["gemini_recovered_phase"] = "pickup_phase"
                                tail["gemini_recovery_note"] = "station6_long_sew_exit_get"

            idx = end + 1

    return rows


def main() -> int:
    args = parse_args()
    rows = list(csv.DictReader(open(args.queue_csv, newline="", encoding="utf-8")))
    if not rows:
        raise ValueError("Queue CSV is empty.")

    output_rows = []
    done_statuses = accepted_statuses()
    for row in rows:
        station_role = infer_station_role(row)
        reviewed_label = normalize_label_for_row(row.get("final_label", ""), row)
        reviewed_status = (row.get("review_status") or "").strip().lower()

        if args.allow_reviewed_override and reviewed_label and reviewed_status in done_statuses:
            label = reviewed_label
            source = "reviewed"
            note = "manual_or_reviewed_label"
        else:
            label, note = safe_label_from_gemini(row, args.min_gemini_confidence)
            source = "gemini_safe"

        phase = normalize_cycle_phase_for_row(
            row.get("gemini_cycle_phase", ""),
            row,
            label=label,
        )
        output_rows.append(
            {
                "video_id": row.get("video_id", ""),
                "station_id": row.get("station_id", ""),
                "station_role": row.get("station_role", "") or station_role,
                "frame_index": row.get("frame_index", ""),
                "timestamp_sec": row.get("timestamp_sec", ""),
                "crop_path": row.get("crop_path", ""),
                "gemini_safe_label": label,
                "gemini_cycle_phase": phase,
                "gemini_cycle_label": label,
                "gemini_source": source,
                "gemini_confidence": row.get("gemini_confidence", ""),
                "gemini_protocol_version": row.get("gemini_protocol_version", ""),
                "gemini_note": note,
                "gemini_schema_error": row.get("gemini_schema_error", ""),
                "gemini_transition_ok": row.get("gemini_transition_ok", ""),
                "gemini_machine_engaged": row.get("gemini_machine_engaged", ""),
                "gemini_hands_on_material": row.get("gemini_hands_on_material", ""),
                "gemini_motion_direction": row.get("gemini_motion_direction", ""),
                "gemini_recovered_label": label,
                "gemini_recovered_phase": phase,
                "gemini_recovery_note": "",
                "activity_label": label,
                "cycle_label_hint": default_cycle_phase_for_label(label, row),
            }
        )

    output_rows = apply_station_long_sew_recovery(output_rows)
    for row in output_rows:
        recovered_label = (row.get("gemini_recovered_label") or row.get("gemini_safe_label") or "").strip()
        recovered_phase = (row.get("gemini_recovered_phase") or row.get("gemini_cycle_phase") or "").strip()
        row["activity_label"] = recovered_label
        row["cycle_label_hint"] = recovered_phase or default_cycle_phase_for_label(recovered_label, row)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
