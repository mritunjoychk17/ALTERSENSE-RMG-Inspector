#!/usr/bin/env python3
"""Build a station-aware clip-validated Stage 2 activity CSV for stricter cycle audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


TARGET_STATIONS = {"1", "4", "5", "6"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-csv", required=True)
    parser.add_argument("--gemini-csv", default="")
    parser.add_argument("--operator-config", default="configs/altersense_operator_profiles.cam33.json")
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bool(value: object, default: bool = False) -> bool:
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def base_label(row: dict) -> str:
    for key in ["postprocessed_label", "smoothed_label", "predicted_label", "label"]:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return "idle"


def confidence(row: dict, label: str) -> float:
    return parse_float(row.get(f"{label}_confidence", ""), default=0.0)


def merge_gemini(clip_rows: list[dict], gemini_rows: list[dict]) -> list[dict]:
    gemini_index = {}
    for row in gemini_rows:
        key = (str(row.get("station_id", "")).strip(), round(parse_float(row.get("timestamp_sec", "")), 3))
        gemini_index[key] = row
    merged = []
    for row in clip_rows:
        key = (str(row.get("station_id", "")).strip(), round(parse_float(row.get("timestamp_sec", "")), 3))
        clone = dict(row)
        extra = gemini_index.get(key, {})
        for field in [
            "gemini_safe_label",
            "gemini_cycle_phase",
            "gemini_motion_direction",
            "gemini_machine_engaged",
            "gemini_hands_on_material",
            "hybrid_postprocessed_label",
            "hybrid_postprocess_note",
        ]:
            clone[field] = extra.get(field, clone.get(field, ""))
        merged.append(clone)
    return merged


def assign_station_roles(rows: list[dict], config: dict) -> None:
    role_map = {str(item["station_id"]): item.get("station_role", "") for item in config.get("stations", [])}
    for row in rows:
        if not (row.get("station_role") or "").strip():
            row["station_role"] = role_map.get(str(row.get("station_id", "")).strip(), "")


def validate_station(rows: list[dict], station_id: str) -> list[dict]:
    rows = sorted(rows, key=lambda r: parse_float(r.get("timestamp_sec", "")))
    labels = [base_label(r) for r in rows]
    notes = ["" for _ in rows]

    for idx, row in enumerate(rows):
        label = labels[idx]
        gem_phase = (row.get("gemini_cycle_phase") or "").strip()
        gem_motion = (row.get("gemini_motion_direction") or "").strip()
        machine = parse_bool(row.get("gemini_machine_engaged"), default=(label == "sew"))
        hands = parse_bool(row.get("gemini_hands_on_material"), default=(label != "idle"))
        put_c = confidence(row, "put")
        get_c = confidence(row, "get")
        align_c = confidence(row, "align")
        sew_c = confidence(row, "sew")

        if station_id in {"4", "6"}:
            if label == "get" and machine and sew_c >= max(0.8, get_c + 0.1):
                labels[idx] = "sew"
                notes[idx] = "weak_station_get_demoted_to_sew"
            elif label == "put" and machine and sew_c >= max(0.85, put_c + 0.1):
                labels[idx] = "sew"
                notes[idx] = "weak_station_put_demoted_to_sew"
            elif label == "align" and machine and sew_c >= 0.85:
                labels[idx] = "sew"
                notes[idx] = "weak_station_align_demoted_to_sew"

        if station_id == "5":
            if label == "align" and gem_phase == "sew_phase" and machine and sew_c >= 0.75:
                labels[idx] = "sew"
                notes[idx] = "station5_align_promoted_to_sew"
            elif label == "put" and sew_c > put_c and machine:
                labels[idx] = "sew"
                notes[idx] = "station5_put_promoted_to_sew"

        if station_id == "1":
            if label == "get" and machine and sew_c >= 0.8 and gem_phase == "sew_phase":
                labels[idx] = "sew"
                notes[idx] = "station1_get_demoted_to_sew"
            elif label == "put" and not machine and put_c < 0.55 and sew_c < 0.55:
                labels[idx] = "align"
                notes[idx] = "station1_put_demoted_to_align"

        if labels[idx] == "align" and not hands and align_c < 0.6:
            labels[idx] = "idle"
            notes[idx] = notes[idx] or "no_hands_align_to_idle"

        if labels[idx] == "get" and gem_motion == "machine_side" and machine and sew_c > get_c:
            labels[idx] = "sew"
            notes[idx] = notes[idx] or "machine_side_get_to_sew"

        # Station 4 often expresses cycle boundary as a brief idle/align break
        # rather than a clean get->align transition.
        if station_id == "4":
            if labels[idx] == "idle" and machine and sew_c >= 0.35:
                labels[idx] = "sew"
                notes[idx] = notes[idx] or "station4_machine_idle_to_sew"
            elif labels[idx] == "idle" and not machine and hands:
                labels[idx] = "align"
                notes[idx] = notes[idx] or "station4_idle_to_align"

    # Segment-aware recovery after confidence cleanup.
    start = 0
    while start < len(rows):
        end = start
        while end + 1 < len(rows) and labels[end + 1] == labels[start]:
            end += 1
        label = labels[start]
        seg_len = end - start + 1

        if label == "sew":
            if station_id in {"4", "6"} and seg_len >= 10:
                prev_idx = start - 1
                next_idx = end + 1
                if prev_idx >= 0 and labels[prev_idx] == "align":
                    labels[start] = "put"
                    notes[start] = "weak_station_long_sew_entry_put"
                if next_idx < len(rows):
                    motion = (rows[next_idx].get("gemini_motion_direction") or "").strip()
                    if labels[next_idx] in {"align", "idle"} or motion == "toward_worker":
                        labels[next_idx] = "get"
                        notes[next_idx] = "weak_station_long_sew_exit_get"
            elif station_id == "5" and seg_len >= 4:
                prev_idx = start - 1
                next_idx = end + 1
                if prev_idx >= 0 and labels[prev_idx] == "align":
                    labels[prev_idx] = "put"
                    notes[prev_idx] = "station5_align_to_put_before_sew"
                if next_idx < len(rows):
                    motion = (rows[next_idx].get("gemini_motion_direction") or "").strip()
                    if motion == "toward_worker" or labels[next_idx] == "align":
                        labels[next_idx] = "get"
                        notes[next_idx] = "station5_post_sew_get"
            elif station_id == "1" and seg_len >= 3:
                prev_idx = start - 1
                next_idx = end + 1
                if prev_idx >= 0 and labels[prev_idx] == "align" and confidence(rows[prev_idx], "put") >= 0.45:
                    labels[prev_idx] = "put"
                    notes[prev_idx] = "station1_align_to_put_before_sew"
                if next_idx < len(rows):
                    motion = (rows[next_idx].get("gemini_motion_direction") or "").strip()
                    if labels[next_idx] in {"align", "idle"} and motion in {"toward_worker", "stationary", ""}:
                        labels[next_idx] = "get"
                        notes[next_idx] = "station1_post_sew_get"

        if station_id == "4":
            # Dedicated recovery for station 4: very long sew runs are broken by
            # short idle/align pauses, so we convert those pauses into explicit
            # get->align restarts when they sit between sew-dominant regions.
            if label in {"align", "idle"} and seg_len <= 4:
                prev_idx = start - 1
                next_idx = end + 1
                prev_is_sew = prev_idx >= 0 and labels[prev_idx] == "sew"
                next_is_sew = next_idx < len(rows) and labels[next_idx] == "sew"
                if prev_is_sew and next_is_sew:
                    labels[start] = "get"
                    notes[start] = "station4_break_start_to_get"
                    for j in range(start + 1, end + 1):
                        labels[j] = "align"
                        notes[j] = "station4_break_fill_align"
                    if next_idx < len(rows) and labels[next_idx] == "sew":
                        labels[next_idx] = "put"
                        notes[next_idx] = "station4_restart_put"

            # If we have a long sew run after an align block but no explicit put,
            # make the first sew sample the placement step.
            if label == "align":
                next_idx = end + 1
                if next_idx < len(rows) and labels[next_idx] == "sew":
                    labels[next_idx] = "put"
                    notes[next_idx] = notes[next_idx] or "station4_align_followed_by_put"

        start = end + 1

    out = []
    for row, label, note in zip(rows, labels, notes):
        clone = dict(row)
        clone["clip_validated_label"] = label
        clone["clip_validation_note"] = note
        out.append(clone)
    return out


def main() -> int:
    args = parse_args()
    clip_rows = read_rows(Path(args.clip_csv))
    gemini_rows = read_rows(Path(args.gemini_csv)) if args.gemini_csv else []
    config = json.loads(Path(args.operator_config).read_text(encoding="utf-8"))
    rows = merge_gemini(clip_rows, gemini_rows) if gemini_rows else clip_rows
    assign_station_roles(rows, config)

    fieldnames = list(rows[0].keys())
    for extra in ["clip_validated_label", "clip_validation_note"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    by_station: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_station[str(row.get("station_id", "")).strip()].append(row)

    output_rows: list[dict] = []
    for station_id, station_rows in sorted(by_station.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        if station_id in TARGET_STATIONS:
            validated = validate_station(station_rows, station_id)
        else:
            validated = []
            for row in sorted(station_rows, key=lambda r: parse_float(r.get("timestamp_sec", ""))):
                clone = dict(row)
                clone["clip_validated_label"] = base_label(row)
                clone["clip_validation_note"] = "pass_through"
                validated.append(clone)
        output_rows.extend(validated)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
