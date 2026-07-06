#!/usr/bin/env python3
"""Reorder saved workstation annotations by their pixel positions.

Filled workstations are sorted in reading order by rows:
- top to bottom
- left to right within each row

This keeps masks attached to their station ROI and leaves empty station slots after
the filled ones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/roi_annotations.template.json")
    parser.add_argument("--video-id", required=True)
    parser.add_argument(
        "--row-threshold",
        type=float,
        default=220.0,
        help="Maximum centroid y-distance to consider annotations in the same row.",
    )
    return parser.parse_args()


def centroid(points: list[list[int]]) -> tuple[float, float]:
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def empty_station(station_id: str) -> dict:
    return {
        "station_id": station_id,
        "station_roi_polygon": [],
        "machine_mask_polygons": [],
        "notes": "",
    }


def group_rows(filled: list[dict], row_threshold: float) -> list[list[dict]]:
    rows: list[list[dict]] = []
    for ws in sorted(filled, key=lambda item: item["_centroid"][1]):
        placed = False
        for row in rows:
            row_y = sum(item["_centroid"][1] for item in row) / len(row)
            if abs(ws["_centroid"][1] - row_y) <= row_threshold:
                row.append(ws)
                placed = True
                break
        if not placed:
            rows.append([ws])
    for row in rows:
        row.sort(key=lambda item: item["_centroid"][0])
    return rows


def main() -> int:
    args = parse_args()
    path = Path(args.config)
    data = json.loads(path.read_text(encoding="utf-8"))
    video = next((v for v in data["videos"] if v["video_id"] == args.video_id), None)
    if video is None:
        raise ValueError(f"Unknown video_id: {args.video_id}")

    original_slots = len(video["workstations"])
    filled = []
    empty = []
    for ws in video["workstations"]:
        if ws["station_roi_polygon"]:
            item = dict(ws)
            item["_centroid"] = centroid(ws["station_roi_polygon"])
            item["_old_station_id"] = ws["station_id"]
            filled.append(item)
        else:
            empty.append(empty_station(ws["station_id"]))

    rows = group_rows(filled, args.row_threshold)
    ordered = [item for row in rows for item in row]

    new_workstations = []
    mapping = []
    for idx, ws in enumerate(ordered, start=1):
        new_ws = {
            "station_id": str(idx),
            "station_roi_polygon": ws["station_roi_polygon"],
            "machine_mask_polygons": ws["machine_mask_polygons"],
            "notes": ws.get("notes", ""),
        }
        new_workstations.append(new_ws)
        mapping.append((ws["_old_station_id"], str(idx), tuple(round(v, 1) for v in ws["_centroid"])))

    while len(new_workstations) < original_slots:
        new_workstations.append(empty_station(str(len(new_workstations) + 1)))

    video["workstations"] = new_workstations
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    print(f"Reorganized {len(ordered)} filled stations for {args.video_id}")
    for old_id, new_id, center in mapping:
        print(f"old {old_id} -> new {new_id} centroid={center}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
