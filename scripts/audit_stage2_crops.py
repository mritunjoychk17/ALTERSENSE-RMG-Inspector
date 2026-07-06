#!/usr/bin/env python3
"""Build Stage 2 crop-audit boards to judge ROI coverage for activity recognition."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="datasets/interim/roi_crops/manifest.csv")
    parser.add_argument("--output-dir", default="artifacts/stage2/visualizations/crop_audit")
    parser.add_argument("--video-id", help="Limit audit to one video.")
    parser.add_argument("--station-id", help="Limit audit to one station.")
    parser.add_argument("--samples-per-station", type=int, default=6, help="How many frames to include per station.")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: (row["video_id"], row["station_id"], int(row["frame_index"])))
    return rows


def choose_samples(rows: list[dict], count: int) -> list[dict]:
    if len(rows) <= count:
        return rows
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[i] for i in indices]


def make_contact_sheet(title: str, items: list[dict], out_path: Path) -> None:
    thumbs = []
    labels = []
    for item in items:
        image = cv2.imread(item["crop_path"])
        if image is None:
            continue
        thumb = cv2.resize(image, (250, 180))
        thumbs.append(thumb)
        labels.append(f"frame={item['frame_index']}  t={item['timestamp_sec']}s")

    if not thumbs:
        return

    cols = min(3, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    header_h = 70
    cell_w, cell_h = 250, 220
    canvas = np.full((header_h + rows * cell_h, cols * cell_w, 3), 247, dtype=np.uint8)
    cv2.putText(canvas, title, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (28, 54, 88), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Check worker visibility, hand coverage, and work-area coverage.", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (88, 103, 126), 1, cv2.LINE_AA)

    for idx, thumb in enumerate(thumbs):
        r = idx // cols
        c = idx % cols
        x = c * cell_w
        y = header_h + r * cell_h
        canvas[y:y + 180, x:x + 250] = thumb
        cv2.rectangle(canvas, (x, y), (x + 249, y + 179), (207, 220, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, labels[idx], (x + 10, y + 202), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (28, 54, 88), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), canvas):
        raise RuntimeError(f"Could not write {out_path}")


def main() -> int:
    args = parse_args()
    rows = load_manifest(Path(args.manifest))
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if args.video_id and row["video_id"] != args.video_id:
            continue
        if args.station_id and row["station_id"] != args.station_id:
            continue
        grouped[(row["video_id"], row["station_id"])].append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for (video_id, station_id), items in grouped.items():
        sample_items = choose_samples(items, args.samples_per_station)
        out_path = output_dir / video_id / f"station_{station_id}_audit.jpg"
        title = f"{video_id}  station {station_id}"
        make_contact_sheet(title, sample_items, out_path)
        summary_rows.append(
            [
                video_id,
                station_id,
                len(items),
                len(sample_items),
                str(out_path),
            ]
        )
        print(out_path)

    summary_path = output_dir / "audit_manifest.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "station_id", "total_crops", "sampled_crops", "audit_image"])
        writer.writerows(summary_rows)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
