#!/usr/bin/env python3
"""Build a phase-labeled Stage 2 clip manifest for temporal training and prediction."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage2_taxonomy import (
    accepted_statuses,
    default_cycle_phase_for_label,
    normalize_cycle_phase_for_row,
    normalize_label_for_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, help="Reviewed queue, validated clip CSV, or Gemini-safe Stage 2 CSV.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--status-column", default="review_status")
    parser.add_argument("--accepted-statuses", default="done,reviewed,approved")
    parser.add_argument("--label-column", default="final_label")
    parser.add_argument("--fallback-label-columns", default="clip_validated_label,hybrid_postprocessed_label,smoothed_label,predicted_label,label,activity_label,gemini_safe_label")
    parser.add_argument("--phase-column", default="gemini_cycle_phase")
    parser.add_argument("--allow-unreviewed", action="store_true")
    return parser.parse_args()


def choose_value(row: dict, keys: list[str]) -> str:
    for key in keys:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def main() -> int:
    args = parse_args()
    rows = list(csv.DictReader(open(args.input_csv, newline="", encoding="utf-8")))
    if not rows:
        raise ValueError("Input CSV is empty.")

    allowed_statuses = {item.strip().lower() for item in args.accepted_statuses.split(",") if item.strip()}
    fallback_columns = [item.strip() for item in args.fallback_label_columns.split(",") if item.strip()]
    out_rows = []

    for row in rows:
        if not args.allow_unreviewed and args.status_column in row:
            status = (row.get(args.status_column) or "").strip().lower()
            if status and status not in allowed_statuses:
                continue

        label_value = choose_value(row, [args.label_column] + fallback_columns)
        label = normalize_label_for_row(label_value, row)
        phase_value = (row.get(args.phase_column) or "").strip()
        phase = normalize_cycle_phase_for_row(phase_value, row, label=label)

        if not phase_value:
            phase = default_cycle_phase_for_label(label, row)

        clip_paths = (row.get("clip_paths") or "").strip()
        if not clip_paths:
            continue

        clone = dict(row)
        clone["label"] = phase
        clone["phase_label"] = phase
        clone["source_activity_label"] = label
        out_rows.append(clone)

    if not out_rows:
        raise ValueError("No phase-labeled rows found after filtering.")

    fieldnames = list(out_rows[0].keys())
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
