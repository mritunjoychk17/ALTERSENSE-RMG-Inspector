#!/usr/bin/env python3
"""Recover likely cycle transitions from dense Stage 2 predictions for sewing stations."""

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
    parser.add_argument("--label-column", default="smoothed_label")
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def postprocess_station(labels: list[str], station_role: str) -> list[str]:
    if station_role != "sewing":
        return labels[:]
    out = labels[:]
    segments = compress_indices(out)

    # Recover missing `put` before short sew bursts preceded by align.
    for idx, (start, end, label) in enumerate(segments):
        if label != "sew":
            continue
        prev_seg = segments[idx - 1] if idx > 0 else None
        if prev_seg and prev_seg[2] == "align":
            out[start] = "put"

    # Recover missing `get` after sew bursts when followed by align/idle.
    segments = compress_indices(out)
    for idx, (start, end, label) in enumerate(segments):
        if label != "sew":
            continue
        next_seg = segments[idx + 1] if idx + 1 < len(segments) else None
        if not next_seg:
            continue
        if next_seg[2] in {"align", "idle"}:
            next_start = next_seg[0]
            out[next_start] = "get"

    # Recover `get` after explicit pass if no get is present next.
    segments = compress_indices(out)
    for idx, (start, end, label) in enumerate(segments):
        if label != "pass":
            continue
        next_seg = segments[idx + 1] if idx + 1 < len(segments) else None
        if next_seg and next_seg[2] == "align":
            out[next_seg[0]] = "get"

    return out


def main() -> int:
    args = parse_args()
    config = json.loads(Path(args.operator_config).read_text(encoding="utf-8"))
    role_map = {str(item["station_id"]): item.get("station_role", "") for item in config.get("stations", [])}

    rows = read_rows(Path(args.input_csv))
    if not rows:
        raise ValueError("Input CSV is empty.")
    fieldnames = list(rows[0].keys())
    for extra in ["postprocessed_label", "postprocess_note"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    by_station: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_station[str(row.get("station_id", "")).strip()].append(row)

    output_rows: list[dict] = []
    for station_id, station_rows in sorted(by_station.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        station_rows.sort(key=lambda r: float(r.get("timestamp_sec") or 0.0))
        labels = [(row.get(args.label_column) or row.get("predicted_label") or "").strip() for row in station_rows]
        processed = postprocess_station(labels, role_map.get(station_id, ""))
        for row, new_label, old_label in zip(station_rows, processed, labels):
            clone = dict(row)
            clone["postprocessed_label"] = new_label
            clone["postprocess_note"] = "" if new_label == old_label else f"{old_label}->{new_label}"
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
