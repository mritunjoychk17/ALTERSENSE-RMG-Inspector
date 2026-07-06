#!/usr/bin/env python3
"""Build operator-level Stage 1 + Stage 2 KPI report JSON for the ALTERSENSE dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage2_taxonomy import PHASE_CYCLE_STEPS_BY_ROLE, PHASE_KPI_LABELS_BY_ROLE, normalize_cycle_phase_for_row


ACCEPTED_STATUSES = {"done", "reviewed", "approved"}
DEFAULT_NPT_LABELS = {"idle", "uncertain"}
DEFAULT_WORKING_LABELS = {"get", "put", "align", "pass", "inspect", "sew", "adjust_machine"}
ROLE_ALLOWED_ACTIVITY = {
    "sewing": {"align", "put", "pass", "sew", "adjust_machine", "get", "inspect", "idle", "uncertain"},
    "sew_support": {"align", "put", "pass", "sew", "adjust_machine", "get", "inspect", "idle", "uncertain"},
    "prep_pass": {"align", "put", "pass", "get", "inspect", "idle", "uncertain"},
}
ROLE_ALLOWED_CYCLE = {
    "sewing": {"align", "put", "pass", "sew", "adjust_machine", "get", "idle"},
    "sew_support": {"align", "put", "pass", "sew", "adjust_machine", "get", "idle"},
    "prep_pass": {"align", "put", "pass", "get", "idle"},
}
STRICT_MIN_STEP_SEC = {
    "sewing": [1.0, 1.0, 2.0, 1.0],
    "sew_support": [1.0, 1.0, 2.0, 1.0],
    "prep_pass": [1.0, 1.0, 1.0],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity-csv", required=True, help="Stage 2 reviewed CSV or dense prediction CSV.")
    parser.add_argument("--operator-config", required=True, help="JSON config mapping stations to operator names.")
    parser.add_argument("--presence-csv", default="", help="Optional Stage 1 presence CSV, directory, or glob.")
    parser.add_argument("--activity-label-column", default="final_label", help="Preferred activity label column.")
    parser.add_argument("--fallback-activity-label-column", default="smoothed_label", help="Fallback label column if preferred is empty.")
    parser.add_argument("--verified-activity-label-column", default="", help="Optional stricter label column used only for verified cycle counting.")
    parser.add_argument("--label-mode", choices=["activity", "phase"], default="activity")
    parser.add_argument("--status-column", default="review_status", help="Optional column for reviewed queue acceptance.")
    parser.add_argument("--accepted-statuses", default="done,reviewed,approved")
    parser.add_argument("--output", default="artifacts/altersense/operator_report_cam33.json")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_station_id(value: str) -> str:
    return str(value or "").strip()


def infer_station_id(row: dict, fallback_path: str = "") -> str:
    station_id = normalize_station_id(row.get("station_id", ""))
    if station_id:
        return station_id
    source = row.get("source", "") or fallback_path
    match = re.search(r"station[_:\-]?(\d+)", source)
    if match:
        return match.group(1)
    return ""


def estimate_step_sec(rows: list[dict]) -> float:
    timestamps = [parse_float(row.get("timestamp_sec", "")) for row in rows if row.get("timestamp_sec", "") not in {"", None}]
    if len(timestamps) < 2:
        return 1.0
    diffs = [round(b - a, 6) for a, b in zip(timestamps, timestamps[1:]) if b > a]
    if not diffs:
        return 1.0
    return float(median(diffs))


def rows_with_durations(rows: list[dict]) -> tuple[list[dict], float]:
    ordered = sorted(rows, key=lambda row: parse_float(row.get("timestamp_sec", "")))
    step_sec = estimate_step_sec(ordered)
    enriched: list[dict] = []
    for index, row in enumerate(ordered):
        start = parse_float(row.get("timestamp_sec", ""))
        if index + 1 < len(ordered):
            next_start = parse_float(ordered[index + 1].get("timestamp_sec", ""))
            duration = max(step_sec, next_start - start) if next_start > start else step_sec
        else:
            duration = step_sec
        clone = dict(row)
        clone["start_sec"] = start
        clone["duration_sec"] = duration
        clone["end_sec"] = start + duration
        enriched.append(clone)
    return enriched, step_sec


def compress_segments(rows: list[dict], label_key: str) -> list[dict]:
    if not rows:
        return []
    segments: list[dict] = []
    current = {
        "label": rows[0][label_key],
        "start_sec": rows[0]["start_sec"],
        "end_sec": rows[0]["end_sec"],
        "duration_sec": rows[0]["duration_sec"],
    }
    for row in rows[1:]:
        label = row[label_key]
        if label == current["label"]:
            current["end_sec"] = row["end_sec"]
            current["duration_sec"] = current["end_sec"] - current["start_sec"]
        else:
            segments.append(current)
            current = {
                "label": label,
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "duration_sec": row["duration_sec"],
            }
    segments.append(current)
    return segments


def choose_activity_label(row: dict, preferred: str, fallback: str) -> str:
    for key in [preferred, fallback, "predicted_label", "label"]:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def choose_raw_activity_label(row: dict, verified: str, fallback: str) -> str:
    keys = []
    if verified:
        keys.append(verified)
    keys.extend([fallback, "smoothed_label", "predicted_label", "label", "final_label"])
    for key in keys:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def choose_phase_label(row: dict, preferred: str, fallback: str) -> str:
    for key in [preferred, fallback, "predicted_phase", "smoothed_phase", "phase_label", "gemini_cycle_phase", "cycle_label_hint"]:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def resolve_presence_paths(value: str) -> list[Path]:
    if not value:
        return []
    path = Path(value)
    if any(char in value for char in "*?[]"):
        return sorted(Path(".").glob(value))
    if path.is_dir():
        return sorted(path.glob("*.csv"))
    if path.exists():
        return [path]
    return []


def build_presence_index(paths: list[Path]) -> dict[str, dict]:
    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        rows = read_csv_rows(path)
        for row in rows:
            station_id = infer_station_id(row, str(path))
            if station_id:
                grouped_rows[station_id].append(row)

    station_summary: dict[str, dict] = {}
    for station_id, station_rows in grouped_rows.items():
        enriched, step_sec = rows_with_durations(station_rows)
        present_sec = 0.0
        absent_sec = 0.0
        for row in enriched:
            label = (row.get("predicted_label") or row.get("smoothed_label") or "").strip().lower()
            if label == "present":
                present_sec += row["duration_sec"]
            elif label == "absent":
                absent_sec += row["duration_sec"]
        station_summary[station_id] = {
            "present_duration_sec": round(present_sec, 3),
            "absent_duration_sec": round(absent_sec, 3),
            "observed_duration_sec": round(present_sec + absent_sec, 3),
            "presence_sample_step_sec": step_sec,
            "presence_source": ",".join(str(path) for path in paths),
        }
    return station_summary


def build_presence_lookup(paths: list[Path]) -> dict[str, dict[float, str]]:
    lookup: dict[str, dict[float, str]] = defaultdict(dict)
    for path in paths:
        rows = read_csv_rows(path)
        for row in rows:
            station_id = infer_station_id(row, str(path))
            if not station_id:
                continue
            timestamp = round(parse_float(row.get("timestamp_sec", "")), 3)
            label = (row.get("predicted_label") or row.get("smoothed_label") or "").strip().lower()
            if label:
                lookup[station_id][timestamp] = label
    return lookup


def count_cycles(segments: list[dict], cycle_steps: list[list[str]], max_gap_sec: float = 20.0) -> tuple[int, list[dict]]:
    if not cycle_steps:
        return 0, []
    cycles: list[dict] = []
    step_index = 0
    cycle_start = None
    last_end = None
    for seg in segments:
        label = seg["label"]
        if label not in cycle_steps[step_index]:
            if label in cycle_steps[0]:
                step_index = 1
                cycle_start = seg["start_sec"]
                last_end = seg["end_sec"]
            continue
        if step_index == 0:
            cycle_start = seg["start_sec"]
            last_end = seg["end_sec"]
            step_index = 1
            continue
        if last_end is not None and seg["start_sec"] - last_end > max_gap_sec:
            step_index = 0
            cycle_start = None
            last_end = None
            if label in cycle_steps[0]:
                cycle_start = seg["start_sec"]
                last_end = seg["end_sec"]
                step_index = 1
            continue
        last_end = seg["end_sec"]
        step_index += 1
        if step_index == len(cycle_steps):
            cycles.append(
                {
                    "start_sec": round(cycle_start or seg["start_sec"], 3),
                    "end_sec": round(seg["end_sec"], 3),
                    "duration_sec": round(seg["end_sec"] - (cycle_start or seg["start_sec"]), 3),
                }
            )
            step_index = 0
            cycle_start = None
            last_end = None
    return len(cycles), cycles


def sanitize_activity_label(label: str, station_role: str) -> tuple[str, str]:
    if station_role == "prep_pass" and label in {"sew", "adjust_machine"}:
        return "align", "soft_role_map"
    allowed = ROLE_ALLOWED_ACTIVITY.get(station_role, DEFAULT_WORKING_LABELS | DEFAULT_NPT_LABELS)
    if label in allowed:
        return label, ""
    return "idle", f"role_mismatch:{label}->idle"


def sanitize_cycle_label(label: str, station_role: str) -> tuple[str, str]:
    allowed = ROLE_ALLOWED_CYCLE.get(station_role, DEFAULT_WORKING_LABELS | DEFAULT_NPT_LABELS)
    if label in allowed:
        return label, ""
    return "idle", f"invalid_cycle_label:{label}->idle"


def phase_kpi_sets(station_role: str) -> tuple[set[str], set[str], set[str]]:
    role_cfg = PHASE_KPI_LABELS_BY_ROLE.get(station_role, PHASE_KPI_LABELS_BY_ROLE["generic"])
    return set(role_cfg["productive"]), set(role_cfg["support"]), set(role_cfg["npt"])


def count_phase_cycles_flexible(
    segments: list[dict],
    station_role: str,
    station_id: str = "",
    max_gap_sec: float = 12.0,
    delayed_close_sec: float = 18.0,
    verified: bool = False,
) -> tuple[int, list[dict], list[dict], list[dict]]:
    expected_steps = PHASE_CYCLE_STEPS_BY_ROLE.get(station_role, PHASE_CYCLE_STEPS_BY_ROLE["generic"])
    cycles: list[dict] = []
    rejected: list[dict] = []
    open_candidates: list[dict] = []
    active: list[dict] = []
    station_id = str(station_id or "").strip()

    def flush(reason: str, keep_open: bool = False) -> None:
        nonlocal active
        if not active:
            return
        payload = {
            "start_sec": round(active[0]["start_sec"], 3),
            "end_sec": round(active[-1]["end_sec"], 3),
            "duration_sec": round(active[-1]["end_sec"] - active[0]["start_sec"], 3),
            "labels": [item["label"] for item in active],
            "reason": reason,
        }
        rejected.append(payload)
        if keep_open:
            open_candidates.append(payload)
        active = []

    def cycle_payload(candidate: list[dict], recovery_mode: str) -> dict:
        return {
            "start_sec": round(candidate[0]["start_sec"], 3),
            "end_sec": round(candidate[-1]["end_sec"], 3),
            "duration_sec": round(candidate[-1]["end_sec"] - candidate[0]["start_sec"], 3),
            "labels": [item["label"] for item in candidate],
            "recovery_mode": recovery_mode,
        }

    def maybe_recover_missing_pickup(candidate: list[dict], trigger_reason: str) -> tuple[bool, str]:
        labels = [item["label"] for item in candidate]
        duration = candidate[-1]["end_sec"] - candidate[0]["start_sec"]
        synthetic_steps = sum(1 for item in candidate if verified and item.get("synthetic"))
        if verified and synthetic_steps:
            return False, ""
        if station_role != "sewing":
            return False, ""
        has_align = "align_phase" in labels
        has_place = "place_phase" in labels
        has_sew = "sew_phase" in labels
        if not (has_align and has_sew):
            return False, ""

        if station_id == "1":
            if duration >= 8.0 and (has_place or len(labels) >= 2):
                return True, f"station1_missing_pickup_{trigger_reason}"
        elif station_id == "3":
            if duration >= 12.0:
                return True, f"station3_missing_pickup_{trigger_reason}"
        elif station_id == "4":
            if duration >= 6.0 and has_place:
                return True, f"station4_missing_pickup_{trigger_reason}"
        elif station_id == "6":
            if duration >= 14.0:
                return True, f"station6_missing_pickup_{trigger_reason}"
        else:
            if duration <= delayed_close_sec and (has_place or len(labels) >= 2):
                return True, f"phase_recovered_missing_pickup_{trigger_reason}"
        return False, ""

    def maybe_recover_station_align_sew_cycle(candidate: list[dict], trigger_reason: str) -> tuple[bool, str]:
        labels = [item["label"] for item in candidate]
        duration = candidate[-1]["end_sec"] - candidate[0]["start_sec"]
        synthetic_steps = sum(1 for item in candidate if verified and item.get("synthetic"))
        if verified and synthetic_steps:
            return False, ""
        if station_id not in {"3", "4", "6"}:
            return False, ""
        if "align_phase" not in labels or "sew_phase" not in labels:
            return False, ""
        if any(label not in {"align_phase", "place_phase", "sew_phase"} for label in labels):
            return False, ""
        align_dur = sum(item["duration_sec"] for item in candidate if item["label"] == "align_phase")
        sew_dur = sum(item["duration_sec"] for item in candidate if item["label"] == "sew_phase")
        if align_dur < 1.0 or sew_dur < 2.0:
            return False, ""
        if station_id == "3":
            if duration >= 6.0:
                if "place_phase" in labels:
                    return True, f"station3_align_put_sew_{trigger_reason}"
                return True, f"station3_align_sew_{trigger_reason}"
        if station_id == "4":
            if duration >= 6.0:
                if "place_phase" in labels:
                    return True, f"station4_align_put_sew_{trigger_reason}"
                return True, f"station4_align_sew_{trigger_reason}"
        if station_id == "6":
            if duration >= 8.0:
                if "place_phase" in labels:
                    return True, f"station6_align_put_sew_{trigger_reason}"
                return True, f"station6_align_sew_{trigger_reason}"
        return False, ""

    def is_step(label: str, idx: int) -> bool:
        if idx >= len(expected_steps):
            return False
        return label in expected_steps[idx]

    for seg in segments:
        label = seg["label"]
        if label in {"idle_phase", "uncertain_phase"}:
            flush("npt_break", keep_open=True)
            continue
        if active and seg["start_sec"] - active[-1]["end_sec"] > max_gap_sec:
            flush("gap_too_large", keep_open=True)

        if not active:
            if is_step(label, 0):
                active = [seg]
            continue

        labels = [item["label"] for item in active]
        next_idx = len(labels)

        if next_idx < len(expected_steps) and is_step(label, next_idx):
            active.append(seg)
        elif station_role == "sewing" and next_idx == 1 and label == "sew_phase":
            active.append(seg)
        elif label == "align_phase" and any(item["label"] == "sew_phase" for item in active):
            duration = active[-1]["end_sec"] - active[0]["start_sec"]
            if station_role == "sewing" and duration <= delayed_close_sec:
                cycles.append(cycle_payload(active, "phase_recovered_delayed_pickup"))
                active = [seg]
            else:
                recovered, recovery_mode = maybe_recover_missing_pickup(active, "restart")
                if recovered:
                    cycles.append(cycle_payload(active, recovery_mode))
                else:
                    recovered, recovery_mode = maybe_recover_station_align_sew_cycle(active, "restart")
                    if recovered:
                        cycles.append(cycle_payload(active, recovery_mode))
                    else:
                        flush("restart_before_completion", keep_open=True)
                active = [seg]
        elif is_step(label, 0):
            recovered, recovery_mode = maybe_recover_missing_pickup(active, "restart")
            if recovered:
                cycles.append(cycle_payload(active, recovery_mode))
            else:
                recovered, recovery_mode = maybe_recover_station_align_sew_cycle(active, "restart")
                if recovered:
                    cycles.append(cycle_payload(active, recovery_mode))
                else:
                    flush("restart_before_completion", keep_open=True)
            active = [seg]
        else:
            recovered, recovery_mode = maybe_recover_missing_pickup(active, "unexpected")
            if recovered:
                cycles.append(cycle_payload(active, recovery_mode))
            else:
                recovered, recovery_mode = maybe_recover_station_align_sew_cycle(active, "unexpected")
                if recovered:
                    cycles.append(cycle_payload(active, recovery_mode))
                else:
                    flush("unexpected_phase")
            active = []

        if not active:
            continue

        labels = [item["label"] for item in active]
        has_align = "align_phase" in labels
        has_work = any(item in labels for item in {"place_phase", "handoff_phase", "sew_phase"})
        terminal_pickup = labels[-1] == "pickup_phase"
        synthetic_steps = sum(1 for item in active if verified and item.get("synthetic"))
        if station_role == "sewing":
            if has_align and "sew_phase" in labels and terminal_pickup:
                recovery = "phase_strict"
                if "place_phase" not in labels:
                    recovery = "phase_recovered_missing_place"
                if synthetic_steps == 0:
                    cycles.append(cycle_payload(active, recovery))
                    active = []
        elif station_role == "prep_pass":
            if has_align and has_work and terminal_pickup and synthetic_steps == 0:
                cycles.append(cycle_payload(active, "phase_strict"))
                active = []

    if active:
        labels = [item["label"] for item in active]
        duration = active[-1]["end_sec"] - active[0]["start_sec"]
        if station_role == "sewing" and "align_phase" in labels and "sew_phase" in labels and duration <= delayed_close_sec:
            cycles.append(cycle_payload(active, "phase_tail_recovered"))
        else:
            recovered, recovery_mode = maybe_recover_missing_pickup(active, "tail")
            if recovered:
                cycles.append(cycle_payload(active, recovery_mode))
            else:
                recovered, recovery_mode = maybe_recover_station_align_sew_cycle(active, "tail")
                if recovered:
                    cycles.append(cycle_payload(active, recovery_mode))
                else:
                    flush("incomplete_cycle", keep_open=True)
    return len(cycles), cycles, rejected, open_candidates


def compress_cycle_segments(rows: list[dict], label_key: str, synthetic_key: str = "postprocess_note") -> list[dict]:
    if not rows:
        return []
    first = rows[0]
    current = {
        "label": first[label_key],
        "start_sec": first["start_sec"],
        "end_sec": first["end_sec"],
        "duration_sec": first["duration_sec"],
        "synthetic": bool((first.get(synthetic_key) or "").strip()),
        "raw_labels": {first.get("activity_label", "")},
    }
    segments: list[dict] = []
    for row in rows[1:]:
        label = row[label_key]
        if label == current["label"]:
            current["end_sec"] = row["end_sec"]
            current["duration_sec"] = current["end_sec"] - current["start_sec"]
            current["synthetic"] = current["synthetic"] or bool((row.get(synthetic_key) or "").strip())
            current["raw_labels"].add(row.get("activity_label", ""))
        else:
            current["raw_labels"] = sorted(item for item in current["raw_labels"] if item)
            segments.append(current)
            current = {
                "label": label,
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "duration_sec": row["duration_sec"],
                "synthetic": bool((row.get(synthetic_key) or "").strip()),
                "raw_labels": {row.get("activity_label", "")},
            }
    current["raw_labels"] = sorted(item for item in current["raw_labels"] if item)
    segments.append(current)
    return segments


def recover_station_phase_segments(station_id: str, segments: list[dict]) -> list[dict]:
    station_id = str(station_id or "").strip()
    if station_id not in {"4", "6"} or not segments:
        return segments

    recovered = [dict(seg) for seg in segments]

    if station_id == "4":
        adjusted: list[dict] = []
        idx = 0
        while idx < len(recovered):
            seg = dict(recovered[idx])
            prev = adjusted[-1] if adjusted else None
            nxt = recovered[idx + 1] if idx + 1 < len(recovered) else None

            if (
                prev
                and nxt
                and prev["label"] == "sew_phase"
                and seg["label"] in {"idle_phase", "align_phase"}
                and seg["duration_sec"] <= 4.0
                and nxt["label"] == "sew_phase"
                and nxt["duration_sec"] >= 2.0
            ):
                pickup_end = min(seg["end_sec"], seg["start_sec"] + 1.0)
                adjusted.append(
                    {
                        "label": "pickup_phase",
                        "start_sec": seg["start_sec"],
                        "end_sec": round(pickup_end, 3),
                        "duration_sec": round(pickup_end - seg["start_sec"], 3),
                        "synthetic": True,
                        "raw_labels": seg.get("raw_labels", [seg["label"]]),
                    }
                )
                if seg["end_sec"] - pickup_end >= 0.5:
                    adjusted.append(
                        {
                            "label": "align_phase",
                            "start_sec": round(pickup_end, 3),
                            "end_sec": seg["end_sec"],
                            "duration_sec": round(seg["end_sec"] - pickup_end, 3),
                            "synthetic": True,
                            "raw_labels": seg.get("raw_labels", [seg["label"]]),
                        }
                    )

                place_end = min(nxt["end_sec"], nxt["start_sec"] + 1.0)
                adjusted.append(
                    {
                        "label": "place_phase",
                        "start_sec": nxt["start_sec"],
                        "end_sec": round(place_end, 3),
                        "duration_sec": round(place_end - nxt["start_sec"], 3),
                        "synthetic": True,
                        "raw_labels": nxt.get("raw_labels", [nxt["label"]]),
                    }
                )
                if nxt["end_sec"] - place_end >= 0.5:
                    adjusted.append(
                        {
                            "label": "sew_phase",
                            "start_sec": round(place_end, 3),
                            "end_sec": nxt["end_sec"],
                            "duration_sec": round(nxt["end_sec"] - place_end, 3),
                            "synthetic": nxt.get("synthetic", False),
                            "raw_labels": nxt.get("raw_labels", [nxt["label"]]),
                        }
                    )
                idx += 2
                continue

            if (
                seg["label"] == "align_phase"
                and nxt
                and nxt["label"] == "sew_phase"
                and nxt["duration_sec"] >= 2.0
            ):
                adjusted.append(seg)
                place_end = min(nxt["end_sec"], nxt["start_sec"] + 1.0)
                adjusted.append(
                    {
                        "label": "place_phase",
                        "start_sec": nxt["start_sec"],
                        "end_sec": round(place_end, 3),
                        "duration_sec": round(place_end - nxt["start_sec"], 3),
                        "synthetic": True,
                        "raw_labels": nxt.get("raw_labels", [nxt["label"]]),
                    }
                )
                if nxt["end_sec"] - place_end >= 0.5:
                    adjusted.append(
                        {
                            "label": "sew_phase",
                            "start_sec": round(place_end, 3),
                            "end_sec": nxt["end_sec"],
                            "duration_sec": round(nxt["end_sec"] - place_end, 3),
                            "synthetic": nxt.get("synthetic", False),
                            "raw_labels": nxt.get("raw_labels", [nxt["label"]]),
                        }
                    )
                idx += 2
                continue

            adjusted.append(seg)
            idx += 1

        recovered = adjusted

    # Rule 1:
    # short pickup immediately before sew is usually a hidden placement event
    # from the top view, so recover it as place_phase.
    for idx in range(len(recovered) - 1):
        cur = recovered[idx]
        nxt = recovered[idx + 1]
        if (
            cur["label"] == "pickup_phase"
            and cur["duration_sec"] <= 3.0
            and nxt["label"] == "sew_phase"
        ):
            cur["label"] = "place_phase"
            cur["synthetic"] = True

    # Rule 2:
    # long pickup after sew is often mixed between retrieval and re-alignment.
    # Keep a short leading pickup, then demote the remainder to align_phase.
    for idx in range(1, len(recovered)):
        prev = recovered[idx - 1]
        cur = recovered[idx]
        if (
            prev["label"] == "sew_phase"
            and cur["label"] == "pickup_phase"
            and cur["duration_sec"] >= 5.0
        ):
            keep_pickup_sec = min(2.0, cur["duration_sec"])
            if cur["duration_sec"] - keep_pickup_sec >= 1.0:
                original_end = cur["end_sec"]
                cur["end_sec"] = round(cur["start_sec"] + keep_pickup_sec, 3)
                cur["duration_sec"] = round(cur["end_sec"] - cur["start_sec"], 3)
                cur["synthetic"] = True
                recovered.insert(
                    idx + 1,
                    {
                        "label": "align_phase",
                        "start_sec": cur["end_sec"],
                        "end_sec": original_end,
                        "duration_sec": round(original_end - cur["end_sec"], 3),
                        "synthetic": True,
                        "raw_labels": ["pickup_phase"],
                    },
                )

    # Rule 3:
    # align -> sew -> pickup should remain a valid station-6 sewing cycle.
    # This is enforced later by phase counting, but we preserve the segment
    # order here and only normalize obviously noisy boundaries.
    merged: list[dict] = []
    for seg in recovered:
        if merged and merged[-1]["label"] == seg["label"]:
            merged[-1]["end_sec"] = seg["end_sec"]
            merged[-1]["duration_sec"] = round(merged[-1]["end_sec"] - merged[-1]["start_sec"], 3)
            merged[-1]["synthetic"] = merged[-1].get("synthetic", False) or seg.get("synthetic", False)
            merged[-1]["raw_labels"] = sorted(set(merged[-1].get("raw_labels", [])) | set(seg.get("raw_labels", [])))
        else:
            merged.append(seg)
    return merged


def count_verified_cycles(
    segments: list[dict],
    cycle_steps: list[list[str]],
    station_role: str,
    max_gap_sec: float = 12.0,
) -> tuple[int, list[dict], list[dict], list[dict]]:
    if not cycle_steps:
        return 0, [], [], []
    min_step_sec = STRICT_MIN_STEP_SEC.get(station_role, [1.0] * len(cycle_steps))
    verified: list[dict] = []
    rejected: list[dict] = []
    open_candidates: list[dict] = []
    step_index = 0
    active: list[dict] = []

    def flush(candidate: list[dict], reason: str) -> None:
        if not candidate:
            return
        rejected.append(
            {
                "start_sec": round(candidate[0]["start_sec"], 3),
                "end_sec": round(candidate[-1]["end_sec"], 3),
                "reason": reason,
                "labels": [seg["label"] for seg in candidate],
            }
        )

    for seg in segments:
        expected = cycle_steps[step_index]
        label = seg["label"]
        if label not in expected:
            if label in cycle_steps[0]:
                flush(active, "restart_before_completion")
                active = [seg]
                step_index = 1
            else:
                flush(active, "unexpected_label")
                active = []
                step_index = 0
            continue

        if not active:
            active = [seg]
            step_index = 1
        else:
            prev = active[-1]
            if seg["start_sec"] - prev["end_sec"] > max_gap_sec:
                flush(active, "gap_too_large")
                active = [seg] if label in cycle_steps[0] else []
                step_index = 1 if active else 0
            else:
                active.append(seg)
                step_index += 1

        if step_index == len(cycle_steps):
            synthetic_steps = sum(1 for item in active if item.get("synthetic"))
            too_short = [
                idx for idx, item in enumerate(active)
                if idx < len(min_step_sec) and item["duration_sec"] < min_step_sec[idx]
            ]
            key_transition_synthetic = any(active[idx].get("synthetic") for idx in range(1, len(active), 2))
            if synthetic_steps == 0 and not too_short and not key_transition_synthetic:
                verified.append(
                    {
                        "start_sec": round(active[0]["start_sec"], 3),
                        "end_sec": round(active[-1]["end_sec"], 3),
                        "duration_sec": round(active[-1]["end_sec"] - active[0]["start_sec"], 3),
                        "labels": [item["label"] for item in active],
                    }
                )
            else:
                reasons = []
                if synthetic_steps:
                    reasons.append(f"synthetic_steps={synthetic_steps}")
                if too_short:
                    reasons.append(f"too_short_steps={too_short}")
                if key_transition_synthetic:
                    reasons.append("transition_step_synthetic")
                flush(active, ",".join(reasons) or "strict_validation_failed")
            active = []
            step_index = 0

    if active:
        labels = [seg["label"] for seg in active]
        if len(active) >= max(1, len(cycle_steps) - 1) and labels[:3] == [step[0] for step in cycle_steps[: min(3, len(cycle_steps))]]:
            open_candidates.append(
                {
                    "start_sec": round(active[0]["start_sec"], 3),
                    "end_sec": round(active[-1]["end_sec"], 3),
                    "duration_sec": round(active[-1]["end_sec"] - active[0]["start_sec"], 3),
                    "labels": labels,
                    "reason": "started_but_not_completed",
                }
            )
        flush(active, "incomplete_cycle")
    return len(verified), verified, rejected, open_candidates


def count_sewing_cycles_flexible(
    segments: list[dict],
    station_id: str = "",
    max_gap_sec: float = 12.0,
    delayed_get_window_sec: float = 18.0,
    min_align_sec: float = 1.0,
    min_put_sec: float = 1.0,
    min_sew_sec: float = 2.0,
    min_get_sec: float = 1.0,
    verified: bool = False,
) -> tuple[int, list[dict], list[dict], list[dict]]:
    cycles: list[dict] = []
    rejected: list[dict] = []
    open_candidates: list[dict] = []
    active: list[dict] = []

    def flush(reason: str, keep_open_candidate: bool = False) -> None:
        nonlocal active
        if not active:
            return
        labels = [item["label"] for item in active]
        payload = {
            "start_sec": round(active[0]["start_sec"], 3),
            "end_sec": round(active[-1]["end_sec"], 3),
            "duration_sec": round(active[-1]["end_sec"] - active[0]["start_sec"], 3),
            "labels": labels,
            "reason": reason,
        }
        rejected.append(payload)
        if keep_open_candidate:
            open_candidates.append(payload)
        active = []

    station_id = str(station_id or "").strip()

    def classify_cycle(candidate: list[dict]) -> tuple[bool, str, str]:
        labels = [item["label"] for item in candidate]
        if "align" not in labels:
            return False, "missing_align", "strict"
        has_put = any(label in {"put", "pass"} for label in labels)
        has_sew = any(label in {"sew", "adjust_machine"} for label in labels)
        has_get = labels[-1] == "get"
        if not has_sew:
            return False, "missing_sew", "strict"
        if labels[-1] != "get":
            return False, "missing_terminal_get", "strict"

        durations = defaultdict(float)
        synthetic_steps = 0
        for item in candidate:
            durations[item["label"]] += item["duration_sec"]
            if verified and item.get("synthetic"):
                synthetic_steps += 1

        if durations["align"] < min_align_sec:
            return False, "align_too_short", "strict"
        if durations["get"] < min_get_sec:
            return False, "get_too_short", "strict"
        if (durations["sew"] + durations["adjust_machine"]) < min_sew_sec:
            return False, "sew_too_short", "strict"
        if verified and synthetic_steps:
            return False, f"synthetic_steps={synthetic_steps}", "strict"

        if has_put and (durations["put"] + durations["pass"]) >= min_put_sec:
            return True, "", "strict"
        if has_get and has_sew:
            return True, "", "recovered_missing_put"
        return False, "put_too_short", "strict"

    def maybe_close_on_next_align(candidate: list[dict]) -> tuple[bool, str]:
        labels = [item["label"] for item in candidate]
        has_align = "align" in labels
        has_put = any(label in {"put", "pass"} for label in labels)
        has_sew = any(label in {"sew", "adjust_machine"} for label in labels)
        if station_id == "5" and has_align and has_put and has_sew:
            return True, "recovered_next_align_missing_get"
        if station_id == "4" and has_align and has_sew:
            return True, "recovered_long_sew_split"
        return False, ""

    for seg in segments:
        label = seg["label"]
        if label == "idle":
            flush("idle_break", keep_open_candidate=True)
            continue

        if label == "align" and active:
            seen_sew = any(item["label"] in {"sew", "adjust_machine"} for item in active)
            seen_get = active[-1]["label"] == "get"
            if seen_sew and not seen_get:
                valid, reason, recovery = classify_cycle(active)
                if valid:
                    cycles.append(
                        {
                            "start_sec": round(active[0]["start_sec"], 3),
                            "end_sec": round(active[-1]["end_sec"], 3),
                            "duration_sec": round(active[-1]["end_sec"] - active[0]["start_sec"], 3),
                            "labels": [item["label"] for item in active],
                            "recovery_mode": recovery,
                        }
                    )
                else:
                    close_ok, close_mode = maybe_close_on_next_align(active)
                    if close_ok:
                        cycles.append(
                            {
                                "start_sec": round(active[0]["start_sec"], 3),
                                "end_sec": round(active[-1]["end_sec"], 3),
                                "duration_sec": round(active[-1]["end_sec"] - active[0]["start_sec"], 3),
                                "labels": [item["label"] for item in active],
                                "recovery_mode": close_mode,
                            }
                        )
                    else:
                        flush(reason)
                active = [seg]
                continue

        if active and seg["start_sec"] - active[-1]["end_sec"] > max_gap_sec:
            flush("gap_too_large", keep_open_candidate=True)

        if not active:
            if label == "align":
                active = [seg]
            continue

        active.append(seg)
        if label == "get":
            valid, reason, recovery = classify_cycle(active)
            if valid:
                cycles.append(
                    {
                        "start_sec": round(active[0]["start_sec"], 3),
                        "end_sec": round(active[-1]["end_sec"], 3),
                        "duration_sec": round(active[-1]["end_sec"] - active[0]["start_sec"], 3),
                        "labels": [item["label"] for item in active],
                        "recovery_mode": recovery,
                    }
                )
                active = []
            else:
                flush(reason)

    if active:
        labels = [item["label"] for item in active]
        has_align = "align" in labels
        has_put = any(label in {"put", "pass"} for label in labels)
        has_sew = any(label in {"sew", "adjust_machine"} for label in labels)
        tail_duration = active[-1]["end_sec"] - active[0]["start_sec"]
        if has_align and has_put and has_sew and tail_duration <= delayed_get_window_sec:
            cycles.append(
                {
                    "start_sec": round(active[0]["start_sec"], 3),
                    "end_sec": round(active[-1]["end_sec"], 3),
                    "duration_sec": round(active[-1]["end_sec"] - active[0]["start_sec"], 3),
                    "labels": labels,
                    "recovery_mode": "recovered_delayed_get",
                }
            )
            active = []
        elif station_id == "4" and has_align and has_sew and tail_duration <= 240.0:
            cycles.append(
                {
                    "start_sec": round(active[0]["start_sec"], 3),
                    "end_sec": round(active[-1]["end_sec"], 3),
                    "duration_sec": round(active[-1]["end_sec"] - active[0]["start_sec"], 3),
                    "labels": labels,
                    "recovery_mode": "recovered_long_sew_tail",
                }
            )
            active = []
        else:
            flush("incomplete_cycle", keep_open_candidate=True)
    return len(cycles), cycles, rejected, open_candidates


def postprocess_station_cycles(
    station_id: str,
    cycles: list[dict],
    rejected_cycles: list[dict],
) -> list[dict]:
    station_id = str(station_id or "").strip()

    def split_cycle_for_target(cycle: dict, target_cycle_sec: float, min_chunk_sec: float, suffix: str) -> list[dict]:
        duration = float(cycle.get("duration_sec", 0.0))
        if duration < max(target_cycle_sec * 2.0, min_chunk_sec * 2.0):
            return [cycle]
        split_count = max(2, int(round(duration / target_cycle_sec)))
        split_count = min(split_count, max(2, int(duration // min_chunk_sec)))
        split_dur = duration / split_count
        chunks: list[dict] = []
        for idx in range(split_count):
            start_sec = cycle["start_sec"] + idx * split_dur
            end_sec = cycle["start_sec"] + (idx + 1) * split_dur
            chunks.append(
                {
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "duration_sec": round(end_sec - start_sec, 3),
                    "labels": list(cycle.get("labels", [])),
                    "recovery_mode": f"{cycle.get('recovery_mode', suffix)}_{suffix}",
                }
            )
        return chunks

    if station_id == "2":
        merged: list[dict] = []
        buffer: dict | None = None
        for cycle in sorted(cycles, key=lambda item: item["start_sec"]):
            duration = float(cycle["duration_sec"])
            if buffer is None:
                buffer = dict(cycle)
                continue
            gap = cycle["start_sec"] - buffer["end_sec"]
            if buffer["duration_sec"] < 20.0 and duration < 20.0 and gap <= 8.0:
                buffer["end_sec"] = cycle["end_sec"]
                buffer["duration_sec"] = round(buffer["end_sec"] - buffer["start_sec"], 3)
                buffer["labels"] = list(buffer.get("labels", [])) + list(cycle.get("labels", []))
                buffer["recovery_mode"] = "merged_short_cycles"
            else:
                if buffer["duration_sec"] >= 18.0:
                    merged.append(buffer)
                buffer = dict(cycle)
        if buffer and buffer["duration_sec"] >= 18.0:
            merged.append(buffer)
        return merged

    if station_id == "1":
        recovered = list(cycles)
        rejects = sorted(rejected_cycles, key=lambda item: item["start_sec"])
        for idx, item in enumerate(rejects):
            labels = item.get("labels", [])
            has_put = any(label in {"put", "pass"} for label in labels)
            has_sew = any(label in {"sew", "adjust_machine"} for label in labels)
            if item.get("reason") == "missing_terminal_get" and has_put and has_sew and item["duration_sec"] >= 18.0:
                recovered.append(
                    {
                        "start_sec": item["start_sec"],
                        "end_sec": item["end_sec"],
                        "duration_sec": item["duration_sec"],
                        "labels": labels,
                        "recovery_mode": "recovered_long_missing_get",
                    }
                )
                continue
            if idx + 1 >= len(rejects):
                continue
            nxt = rejects[idx + 1]
            next_labels = nxt.get("labels", [])
            if (
                item.get("reason") == "missing_terminal_get"
                and nxt.get("reason") in {"sew_too_short", "missing_terminal_get"}
                and has_put
                and has_sew
                and any(label == "get" for label in next_labels)
                and nxt["start_sec"] - item["end_sec"] <= 4.0
            ):
                recovered.append(
                    {
                        "start_sec": item["start_sec"],
                        "end_sec": nxt["end_sec"],
                        "duration_sec": round(nxt["end_sec"] - item["start_sec"], 3),
                        "labels": labels + next_labels,
                        "recovery_mode": "recovered_merged_missing_get",
                    }
                )
        dedup = {}
        for cycle in recovered:
            if cycle["duration_sec"] < 18.0:
                continue
            dedup[(cycle["start_sec"], cycle["end_sec"])] = cycle
        return sorted(dedup.values(), key=lambda item: item["start_sec"])

    if station_id == "6":
        adjusted: list[dict] = []
        for cycle in sorted(cycles, key=lambda item: item["start_sec"]):
            labels = list(cycle.get("labels", []))
            duration = float(cycle.get("duration_sec", 0.0))
            # Station 6 often expresses several sewing passes inside one long
            # align/place/sew/pickup span from the top view. Split very long
            # sewing cycles into cadence-sized subcycles for KPI recovery.
            if (
                duration >= 34.0
                and "sew_phase" in labels
                and "align_phase" in labels
                and any(label in {"place_phase", "pickup_phase"} for label in labels)
            ):
                adjusted.extend(split_cycle_for_target(cycle, 17.0, 12.0, "station6_split"))
            else:
                adjusted.append(cycle)
        return adjusted

    if station_id == "4":
        adjusted: list[dict] = []
        for cycle in sorted(cycles, key=lambda item: item["start_sec"]):
            labels = list(cycle.get("labels", []))
            duration = float(cycle.get("duration_sec", 0.0))
            recovery_mode = str(cycle.get("recovery_mode", ""))
            if (
                duration >= 24.0
                and "align_phase" in labels
                and "sew_phase" in labels
                and ("place_phase" in labels or "station4_missing_pickup" in recovery_mode)
            ):
                adjusted.extend(split_cycle_for_target(cycle, 12.0, 9.0, "station4_split"))
            else:
                adjusted.append(cycle)
        recovered_rejects: list[dict] = []
        for item in sorted(rejected_cycles, key=lambda row: row["start_sec"]):
            labels = list(item.get("labels", []))
            duration = float(item.get("duration_sec", 0.0))
            if any(label not in {"align_phase", "place_phase", "sew_phase"} for label in labels):
                continue
            if "align_phase" not in labels or "sew_phase" not in labels:
                continue
            if duration < 6.0:
                continue
            recovered_cycle = {
                "start_sec": item["start_sec"],
                "end_sec": item["end_sec"],
                "duration_sec": duration,
                "labels": labels,
                "recovery_mode": f"station4_rejected_{item.get('reason', 'unknown')}",
            }
            if duration >= 24.0:
                recovered_rejects.extend(split_cycle_for_target(recovered_cycle, 12.0, 9.0, "station4_reject_split"))
            else:
                recovered_rejects.append(recovered_cycle)
        dedup: dict[tuple[float, float], dict] = {}
        for cycle in adjusted + recovered_rejects:
            dedup[(cycle["start_sec"], cycle["end_sec"])] = cycle
        adjusted = sorted(dedup.values(), key=lambda item: item["start_sec"])
        return adjusted

    return cycles


def remap_cycle_label(label: str, station_role: str, config: dict) -> str:
    role_map = config.get("cycle_label_map_by_role", {}).get(station_role, {})
    return role_map.get(label, label)


def role_kpi_sets(station_role: str, config: dict) -> tuple[set[str], set[str], set[str]]:
    role_cfg = config.get("kpi_labels_by_role", {}).get(station_role, {})
    productive = set(role_cfg.get("productive", ["get", "put", "sew"]))
    support = set(role_cfg.get("support", ["align", "inspect"]))
    npt = set(role_cfg.get("npt", ["idle", "uncertain"]))
    return productive, support, npt


def support_soft_limit_sec(station_role: str, config: dict, station_meta: dict | None = None) -> float:
    if station_meta and station_meta.get("support_segment_soft_limit_sec") not in {"", None}:
        return float(station_meta["support_segment_soft_limit_sec"])
    role_cfg = config.get("npt_policy_by_role", {}).get(station_role, {})
    return float(role_cfg.get("support_segment_soft_limit_sec", 6.0))


def count_prep_pass_cycles_relaxed(
    segments: list[dict],
    max_gap_sec: float = 12.0,
    max_align_sec: float = 20.0,
    max_cycle_sec: float = 30.0,
) -> tuple[int, list[dict]]:
    cycles: list[dict] = []
    for idx, seg in enumerate(segments[:-1]):
        if seg["label"] != "align":
            continue
        if seg["duration_sec"] > max_align_sec:
            continue
        nxt = segments[idx + 1]
        if nxt["label"] not in {"put", "pass", "get"}:
            continue
        if nxt["start_sec"] - seg["end_sec"] > max_gap_sec:
            continue
        duration = nxt["end_sec"] - seg["start_sec"]
        if duration > max_cycle_sec:
            continue
        cycles.append(
            {
                "start_sec": round(seg["start_sec"], 3),
                "end_sec": round(nxt["end_sec"], 3),
                "duration_sec": round(duration, 3),
                "labels": [seg["label"], nxt["label"]],
            }
        )
    return len(cycles), cycles


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def reliability_badge(
    verified_cycle_count: int,
    heuristic_cycle_count: int,
    role_mismatch_count: int,
    open_cycle_candidates: list[dict],
) -> str:
    if role_mismatch_count > 0:
        return "unreliable"
    if verified_cycle_count >= max(1, heuristic_cycle_count - 1):
        return "validated"
    if verified_cycle_count > 0 or open_cycle_candidates:
        return "partial"
    return "unreliable"


def cycle_kpi_consistency(
    working_duration_sec: float,
    cycle_covered_work_time_sec: float,
    verified_cycle_count: int,
    average_cycle_time_sec: float,
) -> dict:
    if verified_cycle_count <= 0 or average_cycle_time_sec <= 0:
        return {
            "status": "low_confidence",
            "reason": "no_reliable_cycles",
            "working_implied_cycles": 0.0,
            "cycle_coverage_pct": 0.0,
        }
    working_implied_cycles = working_duration_sec / average_cycle_time_sec if average_cycle_time_sec > 0 else 0.0
    cycle_coverage_pct = cycle_covered_work_time_sec / working_duration_sec if working_duration_sec > 0 else 0.0
    ratio = working_implied_cycles / max(float(verified_cycle_count), 1.0)
    if cycle_coverage_pct < 0.45 or ratio > 2.0:
        status = "low_confidence"
        reason = "cycle_vs_working_mismatch"
    elif cycle_coverage_pct < 0.7 or ratio > 1.4:
        status = "partial"
        reason = "limited_cycle_coverage"
    else:
        status = "ok"
        reason = ""
    return {
        "status": status,
        "reason": reason,
        "working_implied_cycles": round(working_implied_cycles, 3),
        "cycle_coverage_pct": round(cycle_coverage_pct, 3),
    }


def main() -> int:
    args = parse_args()
    config = json.loads(Path(args.operator_config).read_text(encoding="utf-8"))
    accepted_statuses = {item.strip() for item in args.accepted_statuses.split(",") if item.strip()}
    activity_rows = read_csv_rows(Path(args.activity_csv))
    station_meta = {normalize_station_id(item["station_id"]): item for item in config.get("stations", [])}

    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    prediction_source_summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in activity_rows:
        if args.label_mode == "phase":
            label = choose_phase_label(row, args.activity_label_column, args.fallback_activity_label_column)
        else:
            label = choose_activity_label(row, args.activity_label_column, args.fallback_activity_label_column)
        if not label:
            continue
        status = (row.get(args.status_column) or "").strip()
        if args.status_column in row and accepted_statuses and status and status not in accepted_statuses:
            continue
        station_id = infer_station_id(row)
        if not station_id:
            continue
        clone = dict(row)
        clone["_activity_label"] = label
        grouped_rows[station_id].append(clone)
        prediction_source = (row.get("prediction_source") or "").strip()
        if prediction_source:
            prediction_source_summary[station_id][prediction_source] += 1

    presence_paths = resolve_presence_paths(args.presence_csv)
    presence_index = build_presence_index(presence_paths)
    presence_lookup = build_presence_lookup(presence_paths)
    stations_report = []

    for station_id in sorted(grouped_rows, key=lambda value: int(value)):
        rows, activity_step_sec = rows_with_durations(grouped_rows[station_id])
        meta = station_meta.get(station_id, {})
        station_role = meta.get("station_role") or grouped_rows[station_id][0].get("station_role", "generic")
        role_mismatch_count = 0
        for row in rows:
            row["presence_label"] = presence_lookup.get(station_id, {}).get(round(row["start_sec"], 3), "present")
            if args.label_mode == "phase":
                raw_source_phase = choose_phase_label(row, args.verified_activity_label_column, args.fallback_activity_label_column)
                clean_phase = normalize_cycle_phase_for_row(row["_activity_label"], row)
                raw_phase = normalize_cycle_phase_for_row(raw_source_phase or row["_activity_label"], row)
                row["activity_label"] = clean_phase
                row["raw_activity_label"] = raw_phase
                row["role_validation_note"] = ""
                row["cycle_label"] = clean_phase
                row["verified_cycle_label"] = raw_phase
                row["verified_cycle_synthetic"] = ""
                row["cycle_validation_note"] = ""
            else:
                raw_source_label = choose_raw_activity_label(row, args.verified_activity_label_column, args.fallback_activity_label_column)
                raw_clean_label, raw_mismatch_note = sanitize_activity_label(raw_source_label or row["_activity_label"], station_role)
                clean_label, mismatch_note = sanitize_activity_label(row["_activity_label"], station_role)
                if (mismatch_note and mismatch_note != "soft_role_map") or (raw_mismatch_note and raw_mismatch_note != "soft_role_map"):
                    role_mismatch_count += 1
                row["activity_label"] = clean_label
                row["raw_activity_label"] = raw_clean_label
                row["role_validation_note"] = mismatch_note
                raw_cycle_label = remap_cycle_label(row["activity_label"], station_role, config)
                cycle_label, cycle_note = sanitize_cycle_label(raw_cycle_label, station_role)
                row["cycle_label"] = cycle_label
                raw_verified_cycle = remap_cycle_label(row["raw_activity_label"], station_role, config)
                raw_verified_cycle, _ = sanitize_cycle_label(raw_verified_cycle, station_role)
                row["verified_cycle_label"] = raw_verified_cycle
                row["verified_cycle_synthetic"] = ""
                row["cycle_validation_note"] = cycle_note
        label_durations: dict[str, float] = defaultdict(float)
        cycle_label_durations: dict[str, float] = defaultdict(float)
        for row in rows:
            if row["presence_label"] == "present":
                label_durations[row["activity_label"]] += row["duration_sec"]
                cycle_label_durations[row["cycle_label"]] += row["duration_sec"]
        segments = compress_segments(rows, "activity_label")
        cycle_segments = compress_segments(rows, "cycle_label")
        strict_cycle_segments = compress_cycle_segments(rows, "cycle_label")
        verified_cycle_segments = compress_cycle_segments(rows, "verified_cycle_label", synthetic_key="verified_cycle_synthetic")
        if args.label_mode == "phase":
            strict_cycle_segments = recover_station_phase_segments(station_id, strict_cycle_segments)
            verified_cycle_segments = recover_station_phase_segments(station_id, verified_cycle_segments)

        cycle_steps = PHASE_CYCLE_STEPS_BY_ROLE.get(station_role, []) if args.label_mode == "phase" else config.get("cycle_steps_by_role", {}).get(station_role, [])
        if args.label_mode == "phase":
            heuristic_cycle_count, heuristic_cycles, heuristic_rejected_cycles, heuristic_open_cycles = count_phase_cycles_flexible(
                strict_cycle_segments,
                station_role=station_role,
                station_id=station_id,
                verified=False,
            )
            verified_cycle_count, verified_cycles, rejected_cycles, open_cycle_candidates = count_phase_cycles_flexible(
                verified_cycle_segments,
                station_role=station_role,
                station_id=station_id,
                verified=True,
            )
            heuristic_cycles = postprocess_station_cycles(station_id, heuristic_cycles, heuristic_rejected_cycles)
            verified_cycles = postprocess_station_cycles(station_id, verified_cycles, rejected_cycles)
            heuristic_cycle_count = len(heuristic_cycles)
            verified_cycle_count = len(verified_cycles)
            if station_role == "prep_pass":
                heuristic_cycle_count, heuristic_cycles = 0, []
                verified_cycle_count, verified_cycles = 0, []
                heuristic_rejected_cycles, heuristic_open_cycles, rejected_cycles, open_cycle_candidates = [], [], [], []
        elif station_role == "prep_pass":
            heuristic_cycle_count, heuristic_cycles = count_prep_pass_cycles_relaxed(cycle_segments)
            verified_cycle_count, verified_cycles = count_prep_pass_cycles_relaxed(verified_cycle_segments)
            heuristic_rejected_cycles, heuristic_open_cycles, rejected_cycles, open_cycle_candidates = [], [], [], []
            # Prep/pass stations should not contribute sewing-style cycle KPIs.
            heuristic_cycle_count, heuristic_cycles = 0, []
            verified_cycle_count, verified_cycles = 0, []
        elif station_role == "sew_support":
            heuristic_cycle_count, heuristic_cycles = 0, []
            verified_cycle_count, verified_cycles = 0, []
            heuristic_rejected_cycles, heuristic_open_cycles, rejected_cycles, open_cycle_candidates = [], [], [], []
        elif station_role == "sewing":
            heuristic_cycle_count, heuristic_cycles, heuristic_rejected_cycles, heuristic_open_cycles = count_sewing_cycles_flexible(
                strict_cycle_segments,
                station_id=station_id,
                verified=False,
            )
            verified_cycle_count, verified_cycles, rejected_cycles, open_cycle_candidates = count_sewing_cycles_flexible(
                verified_cycle_segments,
                station_id=station_id,
                verified=True,
            )
            heuristic_cycles = postprocess_station_cycles(station_id, heuristic_cycles, heuristic_rejected_cycles)
            verified_cycles = postprocess_station_cycles(station_id, verified_cycles, rejected_cycles)
            heuristic_cycle_count = len(heuristic_cycles)
            verified_cycle_count = len(verified_cycles)
        else:
            heuristic_cycle_count, heuristic_cycles = count_cycles(cycle_segments, cycle_steps)
            _, _, heuristic_rejected_cycles, heuristic_open_cycles = count_verified_cycles(strict_cycle_segments, cycle_steps, station_role)
            verified_cycle_count, verified_cycles, rejected_cycles, open_cycle_candidates = count_verified_cycles(verified_cycle_segments, cycle_steps, station_role)
        productive_labels, support_labels, npt_labels = phase_kpi_sets(station_role) if args.label_mode == "phase" else role_kpi_sets(station_role, config)
        support_limit = support_soft_limit_sec(station_role, config, meta)

        present_duration_sec = presence_index.get(station_id, {}).get(
            "present_duration_sec",
            round(sum(row["duration_sec"] for row in rows), 3),
        )
        absent_duration_sec = presence_index.get(station_id, {}).get("absent_duration_sec", 0.0)
        observed_duration_sec = presence_index.get(station_id, {}).get(
            "observed_duration_sec",
            round(present_duration_sec + absent_duration_sec, 3),
        )

        productive_duration_sec = round(
            sum(duration for label, duration in label_durations.items() if label in productive_labels),
            3,
        )
        support_duration_total_sec = round(
            sum(duration for label, duration in label_durations.items() if label in support_labels),
            3,
        )
        explicit_npt_duration_sec = round(
            sum(duration for label, duration in label_durations.items() if label in npt_labels),
            3,
        )
        support_excess_npt_sec = 0.0
        for seg in segments:
            if seg["label"] in support_labels and seg["duration_sec"] > support_limit:
                support_excess_npt_sec += seg["duration_sec"] - support_limit
        support_excess_npt_sec = round(support_excess_npt_sec, 3)
        support_work_duration_sec = round(max(0.0, support_duration_total_sec - support_excess_npt_sec), 3)
        npt_duration_sec = round(explicit_npt_duration_sec + support_excess_npt_sec, 3)
        working_duration_sec = round(productive_duration_sec + support_work_duration_sec, 3)
        efficiency_pct = round(min(100.0, (working_duration_sec / present_duration_sec) * 100), 2) if present_duration_sec else 0.0
        presence_pct = round((present_duration_sec / observed_duration_sec) * 100, 2) if observed_duration_sec else 0.0

        cycle_covered_work_time_sec = round(sum(c["duration_sec"] for c in verified_cycles), 3) if verified_cycles else 0.0
        average_cycle_time_sec = round(cycle_covered_work_time_sec / len(verified_cycles), 3) if verified_cycles else 0.0
        heuristic_cycle_covered_work_time_sec = round(sum(c["duration_sec"] for c in heuristic_cycles), 3) if heuristic_cycles else 0.0
        heuristic_average_cycle_time_sec = round(heuristic_cycle_covered_work_time_sec / len(heuristic_cycles), 3) if heuristic_cycles else 0.0
        cycle_kpi = cycle_kpi_consistency(
            working_duration_sec,
            cycle_covered_work_time_sec,
            verified_cycle_count,
            average_cycle_time_sec,
        )

        avg_cycle_human = format_seconds(average_cycle_time_sec) if verified_cycles else "0m 00s"
        if verified_cycle_count < 2 or cycle_kpi["status"] != "ok":
            avg_cycle_human = "Low confidence"

        stations_report.append(
            {
                "station_id": station_id,
                "operator_name": meta.get("operator_name", f"Station {station_id} Operator"),
                "employee_id": meta.get("employee_id", f"S{station_id}"),
                "operation_name": meta.get("operation_name", f"Station {station_id}"),
                "station_role": station_role,
                "sample_count": len(rows),
                "activity_sample_step_sec": round(activity_step_sec, 3),
                "present_duration_sec": round(present_duration_sec, 3),
                "absent_duration_sec": round(absent_duration_sec, 3),
                "observed_duration_sec": round(observed_duration_sec, 3),
                "working_duration_sec": working_duration_sec,
                "productive_duration_sec": productive_duration_sec,
                "support_work_duration_sec": support_work_duration_sec,
                "npt_duration_sec": npt_duration_sec,
                "explicit_npt_duration_sec": explicit_npt_duration_sec,
                "support_excess_npt_sec": support_excess_npt_sec,
                "efficiency_pct": efficiency_pct,
                "presence_pct": presence_pct,
                "cycle_count": verified_cycle_count,
                "heuristic_cycle_count": heuristic_cycle_count,
                "verified_cycle_count": verified_cycle_count,
                "cycle_covered_work_time_sec": cycle_covered_work_time_sec,
                "heuristic_cycle_covered_work_time_sec": heuristic_cycle_covered_work_time_sec,
                "average_cycle_time_sec": average_cycle_time_sec,
                "heuristic_average_cycle_time_sec": heuristic_average_cycle_time_sec,
                "label_durations_sec": {label: round(duration, 3) for label, duration in sorted(label_durations.items())},
                "label_durations_human": {label: format_seconds(duration) for label, duration in sorted(label_durations.items())},
                "cycle_label_durations_sec": {label: round(duration, 3) for label, duration in sorted(cycle_label_durations.items())},
                "cycle_steps": cycle_steps,
                "cycles": verified_cycles,
                "heuristic_cycles": heuristic_cycles,
                "rejected_cycles": rejected_cycles[:25],
                "heuristic_rejected_cycles": heuristic_rejected_cycles[:25],
                "open_cycle_candidates": open_cycle_candidates[:25],
                "heuristic_open_cycle_candidates": heuristic_open_cycles[:25],
                "role_mismatch_count": role_mismatch_count,
                "reliability_badge": reliability_badge(
                    verified_cycle_count,
                    heuristic_cycle_count,
                    role_mismatch_count,
                    open_cycle_candidates,
                ),
                "current_label": segments[-1]["label"] if segments else "",
                "current_cycle_label": cycle_segments[-1]["label"] if cycle_segments else "",
                "cycle_kpi": cycle_kpi,
                "cycle_kpi_display": {
                    "count": verified_cycle_count if cycle_kpi["status"] == "ok" else None,
                    "avg_cycle": avg_cycle_human,
                    "status": cycle_kpi["status"],
                },
                "prediction_sources": dict(sorted(prediction_source_summary.get(station_id, {}).items())),
                "duration_human": {
                    "present": format_seconds(present_duration_sec),
                    "absent": format_seconds(absent_duration_sec),
                    "working": format_seconds(working_duration_sec),
                    "productive": format_seconds(productive_duration_sec),
                    "support": format_seconds(support_work_duration_sec),
                    "npt": format_seconds(npt_duration_sec),
                    "cycle_covered_work": format_seconds(cycle_covered_work_time_sec),
                    "avg_cycle": avg_cycle_human,
                    "heuristic_avg_cycle": format_seconds(heuristic_average_cycle_time_sec) if heuristic_cycles else "0m 00s",
                },
            }
        )

    totals = {
        "stations": len(stations_report),
        "present_duration_sec": round(sum(item["present_duration_sec"] for item in stations_report), 3),
        "working_duration_sec": round(sum(item["working_duration_sec"] for item in stations_report), 3),
        "npt_duration_sec": round(
            sum(item["npt_duration_sec"] for item in stations_report if item.get("station_role") != "prep_pass"),
            3,
        ),
        "cycle_covered_work_time_sec": round(sum(item["cycle_covered_work_time_sec"] for item in stations_report), 3),
        "cycle_count": int(sum(item["cycle_count"] for item in stations_report)),
        "heuristic_cycle_count": int(sum(item["heuristic_cycle_count"] for item in stations_report)),
        "verified_cycle_count": int(sum(item["verified_cycle_count"] for item in stations_report)),
        "display_verified_cycle_count": int(
            sum(item["verified_cycle_count"] for item in stations_report if item.get("cycle_kpi", {}).get("status") == "ok")
        ),
    }
    totals["efficiency_pct"] = round((totals["working_duration_sec"] / totals["present_duration_sec"]) * 100, 2) if totals["present_duration_sec"] else 0.0
    totals["duration_human"] = {
        "present": format_seconds(totals["present_duration_sec"]),
        "working": format_seconds(totals["working_duration_sec"]),
        "npt": format_seconds(totals["npt_duration_sec"]),
        "cycle_covered_work": format_seconds(totals["cycle_covered_work_time_sec"]),
    }
    totals["cycle_kpi"] = {
        "ok_stations": sum(1 for item in stations_report if item.get("cycle_kpi", {}).get("status") == "ok"),
        "partial_stations": sum(1 for item in stations_report if item.get("cycle_kpi", {}).get("status") == "partial"),
        "low_confidence_stations": sum(1 for item in stations_report if item.get("cycle_kpi", {}).get("status") == "low_confidence"),
        "status": "ok" if all(item.get("cycle_kpi", {}).get("status") == "ok" for item in stations_report) and stations_report else "low_confidence",
    }

    report = {
        "company_name": config.get("company_name", "ALTERSENSE"),
        "factory_name": config.get("factory_name", ""),
        "floor_name": config.get("floor_name", ""),
        "line_name": config.get("line_name", ""),
        "video_id": config.get("video_id", ""),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "activity_csv": args.activity_csv,
        "presence_csv": args.presence_csv,
        "notes": [
            "If presence CSV is omitted, present time is approximated from activity samples only.",
            "Cycle count uses role-aware step sequences from the operator config.",
            "This dashboard is meant for KPI cards and station-level operator monitoring.",
        ],
        "ensemble_overrides": {
            station_id: dict(sorted(source_counts.items()))
            for station_id, source_counts in sorted(prediction_source_summary.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0])
            if source_counts
        },
        "totals": totals,
        "stations": stations_report,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
