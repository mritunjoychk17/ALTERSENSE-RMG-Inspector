#!/usr/bin/env python3
"""Build a transition-focused Stage 2 review queue from dense predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


QUEUE_FIELDS = [
    "video_id",
    "station_id",
    "frame_index",
    "timestamp_sec",
    "crop_path",
    "prev_crop_path",
    "next_crop_path",
    "presence_confidence",
    "pose_label",
    "pose_confidence",
    "pose_reason",
    "gemini_label",
    "gemini_confidence",
    "gemini_reason",
    "final_label",
    "review_status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="artifacts/stage2/eval/cam39_dense_activity_predictions_v2.csv")
    parser.add_argument("--manifest", default="datasets/interim/roi_crops_dense_cam39/manifest.csv")
    parser.add_argument("--output", default="datasets/processed/stage2/manifests/transition_review_queue.csv")
    parser.add_argument("--max-per-station", type=int, default=24)
    parser.add_argument("--min-time-gap-sec", type=float, default=5.0)
    parser.add_argument("--top-k-per-station", type=int, default=220)
    return parser.parse_args()


def resolve_ids(row: dict) -> tuple[str, str]:
    video_id = row.get("video_id", "")
    station_id = row.get("station_id", "")
    if video_id and station_id:
        return video_id, station_id
    source_path = Path(row.get("source", ""))
    parts = source_path.parts
    if len(parts) >= 3:
        station_part = parts[-2]
        video_part = parts[-3]
        if station_part.startswith("station_"):
            station_id = station_part.replace("station_", "", 1)
        video_id = video_part
    return video_id, station_id


def load_manifest(path: Path) -> tuple[dict[tuple[str, str, str], dict], dict[tuple[str, str], list[dict]]]:
    rows_by_key: dict[tuple[str, str, str], dict] = {}
    station_rows: dict[tuple[str, str], list[dict]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["video_id"], row["station_id"], row["frame_index"])
            rows_by_key[key] = row
            station_rows.setdefault((row["video_id"], row["station_id"]), []).append(row)
    for bucket in station_rows.values():
        bucket.sort(key=lambda r: int(r["frame_index"]))
    return rows_by_key, station_rows


def main() -> int:
    args = parse_args()
    manifest_rows, station_rows = load_manifest(Path(args.manifest))

    preds_by_station: dict[tuple[str, str], list[dict]] = {}
    with Path(args.predictions).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            video_id, station_id = resolve_ids(row)
            key = (video_id, station_id)
            row["video_id"] = video_id
            row["station_id"] = station_id
            preds_by_station.setdefault(key, []).append(row)

    output_rows: list[dict] = []
    for station_key, pred_rows in sorted(preds_by_station.items(), key=lambda kv: int(kv[0][1])):
        pred_rows.sort(key=lambda r: float(r.get("timestamp_sec") or 0.0))
        candidates: list[dict] = []
        for idx, row in enumerate(pred_rows):
            conf_fields = ["idle_confidence", "get_confidence", "put_confidence", "sew_confidence"]
            probs = []
            for field in conf_fields:
                try:
                    probs.append(float(row.get(field, 0.0) or 0.0))
                except ValueError:
                    probs.append(0.0)
            probs_sorted = sorted(probs, reverse=True)
            top1 = probs_sorted[0] if probs_sorted else 0.0
            top2 = probs_sorted[1] if len(probs_sorted) > 1 else 0.0
            margin = top1 - top2
            low_conf_score = 1.0 - top1

            drift = 0.0
            if idx > 0:
                prev_probs = [
                    float(pred_rows[idx - 1].get(field, 0.0) or 0.0)
                    for field in conf_fields
                ]
                drift += sum(abs(a - b) for a, b in zip(probs, prev_probs))
            if idx < len(pred_rows) - 1:
                next_probs = [
                    float(pred_rows[idx + 1].get(field, 0.0) or 0.0)
                    for field in conf_fields
                ]
                drift += sum(abs(a - b) for a, b in zip(probs, next_probs))

            score = low_conf_score + (1.0 - margin) + drift
            frame_key = (row["video_id"], row["station_id"], row["frame_index"])
            manifest_row = manifest_rows.get(frame_key)
            if not manifest_row:
                continue
            candidates.append(
                {
                    "video_id": row["video_id"],
                    "station_id": row["station_id"],
                    "frame_index": row["frame_index"],
                    "timestamp_sec": float(row.get("timestamp_sec") or 0.0),
                    "crop_path": manifest_row["crop_path"],
                    "score": score,
                    "margin": margin,
                    "top1": top1,
                    "drift": drift,
                    "predicted_label": row.get("predicted_label", ""),
                }
            )

        candidates.sort(key=lambda r: (-r["score"], r["timestamp_sec"]))
        shortlist = candidates[: max(args.top_k_per_station, args.max_per_station)]
        selected: list[dict] = []
        for item in shortlist:
            if len(selected) >= args.max_per_station:
                break
            if any(abs(item["timestamp_sec"] - prev["timestamp_sec"]) < args.min_time_gap_sec for prev in selected):
                continue
            selected.append(item)
        if len(selected) < args.max_per_station:
            stride = max(1, len(pred_rows) // max(args.max_per_station, 1))
            for idx in range(0, len(pred_rows), stride):
                row = pred_rows[idx]
                frame_key = (row["video_id"], row["station_id"], row["frame_index"])
                manifest_row = manifest_rows.get(frame_key)
                if not manifest_row:
                    continue
                item = {
                    "video_id": row["video_id"],
                    "station_id": row["station_id"],
                    "frame_index": row["frame_index"],
                    "timestamp_sec": float(row.get("timestamp_sec") or 0.0),
                    "crop_path": manifest_row["crop_path"],
                    "score": -1.0,
                    "margin": 0.0,
                    "top1": 0.0,
                    "drift": 0.0,
                    "predicted_label": row.get("predicted_label", ""),
                }
                if len(selected) >= args.max_per_station:
                    break
                if any(s["frame_index"] == item["frame_index"] for s in selected):
                    continue
                selected.append(item)

        rows_for_station = station_rows.get(station_key, [])
        frame_to_idx = {row["frame_index"]: idx for idx, row in enumerate(rows_for_station)}
        for item in sorted(selected, key=lambda r: r["timestamp_sec"]):
            idx = frame_to_idx.get(item["frame_index"], -1)
            prev_crop = rows_for_station[idx - 1]["crop_path"] if idx > 0 else ""
            next_crop = rows_for_station[idx + 1]["crop_path"] if 0 <= idx < len(rows_for_station) - 1 else ""
            output_rows.append(
                {
                    "video_id": item["video_id"],
                    "station_id": item["station_id"],
                    "frame_index": item["frame_index"],
                    "timestamp_sec": f"{item['timestamp_sec']:.3f}",
                    "crop_path": item["crop_path"],
                    "prev_crop_path": prev_crop,
                    "next_crop_path": next_crop,
                    "presence_confidence": "",
                    "pose_label": "",
                    "pose_confidence": "",
                    "pose_reason": "",
                    "gemini_label": "",
                    "gemini_confidence": "",
                    "gemini_reason": "",
                    "final_label": "",
                    "review_status": "pending",
                    "notes": (
                        f"transition_focus predicted={item['predicted_label']} "
                        f"top1={item['top1']:.3f} margin={item['margin']:.3f} drift={item['drift']:.3f}"
                    ),
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QUEUE_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
