#!/usr/bin/env python3
"""Interactive tool for annotating workstation ROIs and machine masks."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "RMG ROI Annotator"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/roi_annotations.template.json")
    parser.add_argument("--frames-dir", default="datasets/interim/annotation_frames")
    parser.add_argument("--video-id", help="Annotate one video_id only.")
    return parser.parse_args()


def make_palette() -> list[tuple[int, int, int]]:
    return [
        (255, 99, 71),
        (65, 105, 225),
        (50, 205, 50),
        (255, 215, 0),
        (218, 112, 214),
        (0, 206, 209),
        (255, 140, 0),
        (199, 21, 133),
        (30, 144, 255),
        (154, 205, 50),
    ]


class Annotator:
    def __init__(self, config_path: Path, frames_dir: Path, video_id: str | None) -> None:
        self.config_path = config_path
        self.frames_dir = frames_dir
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.videos = self.config["videos"]
        if video_id:
            self.videos = [video for video in self.videos if video["video_id"] == video_id]
            if not self.videos:
                raise ValueError(f"Unknown video_id: {video_id}")
        self.video_index = 0
        self.station_index = 0
        self.mode = "station"
        self.active_machine_index = -1
        self.current_polygon: list[list[int]] = []
        self.palette = make_palette()

    @property
    def video(self) -> dict:
        return self.videos[self.video_index]

    @property
    def stations(self) -> list[dict]:
        return self.video["workstations"]

    @property
    def station(self) -> dict:
        return self.stations[self.station_index]

    def frame_path(self) -> Path:
        return self.frames_dir / f"{self.video['video_id']}_frame_000000.jpg"

    def load_frame(self) -> np.ndarray:
        frame = cv2.imread(str(self.frame_path()))
        if frame is None:
            raise RuntimeError(f"Could not read {self.frame_path()}")
        return frame

    def save(self) -> None:
        self.config_path.write_text(json.dumps(self.config, indent=2) + "\n", encoding="utf-8")

    def next_station(self, delta: int) -> None:
        self.station_index = (self.station_index + delta) % len(self.stations)
        self.active_machine_index = -1
        self.current_polygon = []

    def next_video(self, delta: int) -> None:
        self.video_index = (self.video_index + delta) % len(self.videos)
        self.station_index = 0
        self.active_machine_index = -1
        self.current_polygon = []

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.current_polygon = []
        if mode == "station":
            self.active_machine_index = -1

    def add_point(self, x: int, y: int) -> None:
        self.current_polygon.append([x, y])

    def undo_point(self) -> None:
        if self.current_polygon:
            self.current_polygon.pop()

    def clear_current(self) -> None:
        self.current_polygon = []

    def commit_polygon(self) -> None:
        if len(self.current_polygon) < 3:
            return
        if self.mode == "station":
            self.station["station_roi_polygon"] = copy.deepcopy(self.current_polygon)
        else:
            self.station["machine_mask_polygons"].append(copy.deepcopy(self.current_polygon))
            self.active_machine_index = len(self.station["machine_mask_polygons"]) - 1
        self.current_polygon = []
        self.save()

    def delete_last_machine_mask(self) -> None:
        if self.station["machine_mask_polygons"]:
            self.station["machine_mask_polygons"].pop()
            self.active_machine_index = len(self.station["machine_mask_polygons"]) - 1
            self.save()

    def draw_polygon(self, canvas: np.ndarray, points: list[list[int]], color: tuple[int, int, int], closed: bool, fill_alpha: float = 0.18) -> None:
        if len(points) < 2:
            return
        pts = np.array(points, dtype=np.int32)
        overlay = canvas.copy()
        if closed and len(points) >= 3:
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, fill_alpha, canvas, 1 - fill_alpha, 0, canvas)
        cv2.polylines(canvas, [pts], closed, color, 2, cv2.LINE_AA)
        for x, y in points:
            cv2.circle(canvas, (x, y), 4, color, -1, cv2.LINE_AA)

    def render(self) -> np.ndarray:
        frame = self.load_frame()
        canvas = frame.copy()

        for idx, station in enumerate(self.stations):
            color = self.palette[idx % len(self.palette)]
            self.draw_polygon(canvas, station["station_roi_polygon"], color, closed=True, fill_alpha=0.10)
            for mask in station["machine_mask_polygons"]:
                self.draw_polygon(canvas, mask, (0, 0, 255), closed=True, fill_alpha=0.12)
            if station["station_roi_polygon"]:
                xs = [p[0] for p in station["station_roi_polygon"]]
                ys = [p[1] for p in station["station_roi_polygon"]]
                anchor = (min(xs), min(ys))
                cv2.putText(canvas, f"S{station['station_id']}", anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        preview_color = self.palette[self.station_index % len(self.palette)] if self.mode == "station" else (0, 0, 255)
        self.draw_polygon(canvas, self.current_polygon, preview_color, closed=False, fill_alpha=0.0)

        lines = [
            f"Video: {self.video['video_id']} ({self.video_index + 1}/{len(self.videos)})",
            f"Station: {self.station['station_id']} ({self.station_index + 1}/{len(self.stations)})",
            f"Mode: {self.mode}",
            "Keys: [ / ] station, - / = video, c station, m mask, enter save polygon, u undo point",
            "Keys: x clear current, d delete last mask, s save file, q quit",
        ]
        panel_h = 118
        panel = np.full((panel_h, canvas.shape[1], 3), 245, dtype=np.uint8)
        for i, text in enumerate(lines):
            cv2.putText(panel, text, (18, 26 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (35, 35, 35), 2, cv2.LINE_AA)
        return np.vstack([panel, canvas])


def main() -> int:
    args = parse_args()
    annotator = Annotator(Path(args.config), Path(args.frames_dir), args.video_id)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            y_adjusted = y - 118
            if y_adjusted >= 0:
                annotator.add_point(x, y_adjusted)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        canvas = annotator.render()
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("["):
            annotator.next_station(-1)
        elif key == ord("]"):
            annotator.next_station(1)
        elif key == ord("-"):
            annotator.next_video(-1)
        elif key == ord("="):
            annotator.next_video(1)
        elif key == ord("c"):
            annotator.set_mode("station")
        elif key == ord("m"):
            annotator.set_mode("mask")
        elif key in (13, 10):
            annotator.commit_polygon()
        elif key == ord("u"):
            annotator.undo_point()
        elif key == ord("x"):
            annotator.clear_current()
        elif key == ord("d"):
            annotator.delete_last_machine_mask()
        elif key == ord("s"):
            annotator.save()

    annotator.save()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
