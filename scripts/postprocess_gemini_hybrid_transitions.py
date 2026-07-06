#!/usr/bin/env python3
"""Hybrid Gemini-aware Stage 2 postprocessor for recovering put/get transitions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--operator-config", default="configs/altersense_operator_profiles.cam33.json")
    parser.add_argument("--label-column", default="activity_label")
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_bool(value: object, default: bool = False) -> bool:
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def compress_indices(labels: list[str]) -> list[tuple[int, int, str]]:
    if not labels:
        return []
    segments = []
    start = 0
    current = labels[0]
    for idx, label in enumerate(labels[1:], start=1):
        if label != current:
            segments.append((start, idx - 1, current))
            start = idx
            current = label
    segments.append((start, len(labels) - 1, current))
    return segments


def recover_station(rows: list[dict], label_column: str, station_role: str, station_id: str) -> tuple[list[str], list[str]]:
    labels = [(row.get(label_column) or "").strip() for row in rows]
    out = labels[:]
    notes = ["" for _ in rows]

    if station_role != "sewing":
        return out, notes

    # Rule 1: align before sew at machine side -> put
    for idx in range(len(rows) - 1):
        cur = rows[idx]
        nxt = rows[idx + 1]
        cur_label = out[idx]
        next_label = out[idx + 1]
        motion = (cur.get("gemini_motion_direction") or "").strip()
        machine = parse_bool(cur.get("gemini_machine_engaged"), default=False)
        hands = parse_bool(cur.get("gemini_hands_on_material"), default=False)
        if (
            cur_label == "align"
            and next_label == "sew"
            and hands
            and (motion == "machine_side" or machine)
        ):
            out[idx] = "put"
            notes[idx] = "align->put before sew using machine_side/hands"

    # Rule 2: sew followed by align/idle with material contact -> get
    for idx in range(1, len(rows)):
        prev_label = out[idx - 1]
        cur_label = out[idx]
        motion = (rows[idx].get("gemini_motion_direction") or "").strip()
        hands = parse_bool(rows[idx].get("gemini_hands_on_material"), default=False)
        if (
            prev_label == "sew"
            and cur_label in {"align", "idle"}
            and hands
            and motion in {"toward_worker", "stationary", "unclear"}
        ):
            out[idx] = "get"
            notes[idx] = "post-sew recovery to get"

    # Rule 3: long sew bursts get split into put/get on boundaries.
    segments = compress_indices(out)
    for start, end, label in segments:
        if label != "sew":
            continue
        seg_len = end - start + 1
        if seg_len >= 3:
            before_label = out[start - 1] if start > 0 else ""
            after_label = out[end + 1] if end + 1 < len(out) else ""
            if before_label not in {"put", "sew"}:
                out[start] = "put"
                notes[start] = "long sew burst leading edge -> put"
            if after_label in {"align", "idle"}:
                out[end + 1] = "get"
                notes[end + 1] = "long sew burst trailing edge -> get"

    # Rule 3b: weak stations 4/6 need more aggressive burst splitting because
    # Gemini compresses nearly all dense windows into sew.
    if station_id in {"4", "6"}:
        segments = compress_indices(out)
        for start, end, label in segments:
            if label != "sew":
                continue
            seg_len = end - start + 1
            if seg_len < 5:
                continue
            # Force a simple recoverable pattern: put on entry, get on exit,
            # and if the burst is very long, inject one midpoint get->put pair.
            if out[start] == "sew":
                out[start] = "put"
                notes[start] = "weak station long sew entry -> put"
            if end < len(out) - 1 and out[end + 1] in {"align", "idle", "sew"}:
                out[end] = "get"
                notes[end] = "weak station long sew exit -> get"
            if seg_len >= 8:
                mid = start + seg_len // 2
                if out[mid] == "sew":
                    out[mid] = "get"
                    notes[mid] = "weak station midpoint -> get"
                if mid + 1 <= end and out[mid + 1] == "sew":
                    out[mid + 1] = "put"
                    notes[mid + 1] = "weak station midpoint -> put"

    # Rule 4: align after place/sew with toward_worker motion -> get
    for idx in range(1, len(rows)):
        prev_label = out[idx - 1]
        cur_label = out[idx]
        motion = (rows[idx].get("gemini_motion_direction") or "").strip()
        hands = parse_bool(rows[idx].get("gemini_hands_on_material"), default=False)
        if (
            cur_label == "align"
            and prev_label in {"put", "pass", "sew"}
            and hands
            and motion == "toward_worker"
        ):
            out[idx] = "get"
            notes[idx] = "align->get using toward_worker motion"

    return out, notes


def main() -> int:
    args = parse_args()
    config = json.loads(Path(args.operator_config).read_text(encoding="utf-8"))
    role_map = {str(item["station_id"]): item.get("station_role", "") for item in config.get("stations", [])}
    rows = read_rows(Path(args.input_csv))
    if not rows:
        raise ValueError("Input CSV is empty.")

    fieldnames = list(rows[0].keys())
    for extra in ["hybrid_postprocessed_label", "hybrid_postprocess_note"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    by_station: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_station[str(row.get("station_id", "")).strip()].append(row)

    output_rows: list[dict] = []
    for station_id, station_rows in sorted(by_station.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        station_rows.sort(key=lambda r: float(r.get("timestamp_sec") or 0.0))
        processed, notes = recover_station(station_rows, args.label_column, role_map.get(station_id, ""), station_id)
        for row, new_label, note in zip(station_rows, processed, notes):
            clone = dict(row)
            clone["hybrid_postprocessed_label"] = new_label
            clone["hybrid_postprocess_note"] = note
            output_rows.append(clone)

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
