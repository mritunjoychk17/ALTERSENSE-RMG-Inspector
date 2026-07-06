#!/usr/bin/env python3
"""Summarize cycle-failure patterns for selected stations from strict and hybrid reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-report", required=True)
    parser.add_argument("--hybrid-report", required=True)
    parser.add_argument("--activity-csv", required=True)
    parser.add_argument("--stations", default="3,4,5")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    station_ids = [item.strip() for item in args.stations.split(",") if item.strip()]

    strict = load_report(Path(args.strict_report))
    hybrid = load_report(Path(args.hybrid_report))
    strict_map = {item["station_id"]: item for item in strict["stations"]}
    hybrid_map = {item["station_id"]: item for item in hybrid["stations"]}

    label_counts: dict[str, Counter] = {sid: Counter() for sid in station_ids}
    with Path(args.activity_csv).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = str(row.get("station_id", "")).strip()
            if sid not in label_counts:
                continue
            label = (row.get("final_label") or row.get("smoothed_label") or row.get("predicted_label") or "").strip()
            if label:
                label_counts[sid][label] += 1

    rows = []
    for sid in station_ids:
        strict_station = strict_map[sid]
        hybrid_station = hybrid_map[sid]
        strict_reasons = Counter(item["reason"] for item in strict_station.get("rejected_cycles", []))
        hybrid_reasons = Counter(item["reason"] for item in hybrid_station.get("rejected_cycles", []))
        counts = label_counts[sid]
        rows.append(
            {
                "station_id": sid,
                "strict_verified_cycles": strict_station["verified_cycle_count"],
                "hybrid_verified_cycles": hybrid_station["verified_cycle_count"],
                "strict_top_reasons": "; ".join(f"{k}:{v}" for k, v in strict_reasons.most_common(4)),
                "hybrid_top_reasons": "; ".join(f"{k}:{v}" for k, v in hybrid_reasons.most_common(4)),
                "align_count": counts.get("align", 0),
                "put_count": counts.get("put", 0),
                "sew_count": counts.get("sew", 0),
                "get_count": counts.get("get", 0),
                "idle_count": counts.get("idle", 0),
                "recommended_recovery_rule": "",
                "rationale": "",
            }
        )

    for row in rows:
        sid = row["station_id"]
        if sid == "3":
            row["recommended_recovery_rule"] = "Allow align->sew closure at next align when get is consistently missed"
            row["rationale"] = "Station 3 has almost no put detections and very few get detections, but repeated align/sew bursts dominate."
        elif sid == "4":
            row["recommended_recovery_rule"] = "Allow get-leading microcycle bridging and long sew-run split by nearby get bursts"
            row["rationale"] = "Station 4 starts with get before align and then stays in long sew spans, so cycle boundaries are badly shifted."
        elif sid == "5":
            row["recommended_recovery_rule"] = "Allow align->put->sew closure at next align when get is delayed or absent"
            row["rationale"] = "Station 5 shows some put evidence but most sequences never emit terminal get before the next align phase."

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "station_id",
                "strict_verified_cycles",
                "hybrid_verified_cycles",
                "strict_top_reasons",
                "hybrid_top_reasons",
                "align_count",
                "put_count",
                "sew_count",
                "get_count",
                "idle_count",
                "recommended_recovery_rule",
                "rationale",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
