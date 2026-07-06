#!/usr/bin/env python3
"""Generate Stage 2 weak labels from pose and short-term motion."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


LABELS = {"idle", "get", "put", "sew", "uncertain"}
ACCEPTED_STATUSES = {"done", "reviewed", "approved"}
QUEUE_COLUMNS = [
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


@dataclass
class Suggestion:
    label: str
    confidence: float
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-csv", default="datasets/processed/stage2/manifests/activity_review_queue.csv")
    parser.add_argument("--pose-model", default="yolo11n-pose.pt", help="Ultralytics pose model path or name.")
    parser.add_argument("--backend", choices=["auto", "ultralytics", "motion"], default="auto")
    parser.add_argument("--limit", type=int, default=0, help="Only update this many pending rows. 0 means all.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing pose suggestions.")
    return parser.parse_args()


def read_image(path_value: str) -> np.ndarray | None:
    if not path_value:
        return None
    image = cv2.imread(path_value)
    return image


def ensure_columns(rows: list[dict]) -> None:
    for row in rows:
        row.setdefault("pose_label", "")
        row.setdefault("pose_confidence", "")
        row.setdefault("pose_reason", "")


def choose_backend(requested: str) -> str:
    if requested in {"ultralytics", "motion"}:
        return requested
    try:
        import ultralytics  # noqa: F401

        return "ultralytics"
    except Exception:
        return "motion"


def largest_pose_keypoints(model, image: np.ndarray) -> np.ndarray | None:
    result = model.predict(image, verbose=False)[0]
    if not hasattr(result, "keypoints") or result.keypoints is None:
        return None
    data = result.keypoints.data
    if data is None or len(data) == 0:
        return None
    scores = []
    for person in data.cpu().numpy():
        confs = person[:, 2]
        scores.append(float(np.mean(confs[confs > 0])) if np.any(confs > 0) else 0.0)
    idx = int(np.argmax(scores))
    return data[idx].cpu().numpy()


def motion_energy(prev_img: np.ndarray | None, curr_img: np.ndarray | None, next_img: np.ndarray | None) -> tuple[float, float, float]:
    if curr_img is None:
        return 0.0, 0.0, 0.0
    gray_curr = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    diffs = []
    if prev_img is not None:
        gray_prev = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        diffs.append(cv2.absdiff(gray_curr, gray_prev))
    if next_img is not None:
        gray_next = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
        diffs.append(cv2.absdiff(gray_curr, gray_next))
    if not diffs:
        return 0.0, 0.0, 0.0
    diff = np.mean(diffs, axis=0).astype(np.uint8)
    h, w = diff.shape
    left = float(diff[:, : w // 2].mean())
    right = float(diff[:, w // 2 :].mean())
    center = float(diff[h // 4 : (3 * h) // 4, w // 4 : (3 * w) // 4].mean())
    return float(diff.mean()), left - right, center


def suggest_from_motion(prev_img: np.ndarray | None, curr_img: np.ndarray | None, next_img: np.ndarray | None) -> Suggestion:
    total_motion, horizontal_bias, center_motion = motion_energy(prev_img, curr_img, next_img)
    if total_motion < 6.0:
        return Suggestion("idle", 0.62, f"Low short-term motion ({total_motion:.1f}) across previous/current/next frames.")
    if center_motion > 18.0 and total_motion < 20.0:
        return Suggestion("sew", 0.58, f"Motion is concentrated near the center work area ({center_motion:.1f}).")
    if horizontal_bias > 4.0:
        return Suggestion("get", 0.46, f"More motion on the left side of the crop ({horizontal_bias:.1f}), which may indicate retrieving material.")
    if horizontal_bias < -4.0:
        return Suggestion("put", 0.46, f"More motion on the right side of the crop ({abs(horizontal_bias):.1f}), which may indicate placing material.")
    return Suggestion("uncertain", 0.35, f"Motion is present ({total_motion:.1f}) but direction and work-area intent are ambiguous.")


def suggest_from_pose_and_motion(model, prev_img: np.ndarray | None, curr_img: np.ndarray | None, next_img: np.ndarray | None) -> Suggestion:
    if curr_img is None:
        return Suggestion("uncertain", 0.0, "Current crop could not be loaded.")
    kp_curr = largest_pose_keypoints(model, curr_img)
    if kp_curr is None:
        return suggest_from_motion(prev_img, curr_img, next_img)

    wrist_indices = [9, 10]
    shoulder_indices = [5, 6]
    valid_wrists = [kp_curr[idx] for idx in wrist_indices if kp_curr[idx, 2] > 0.2]
    valid_shoulders = [kp_curr[idx] for idx in shoulder_indices if kp_curr[idx, 2] > 0.2]
    total_motion, horizontal_bias, center_motion = motion_energy(prev_img, curr_img, next_img)

    if total_motion < 6.0:
        return Suggestion("idle", 0.68, f"Pose detected and short-term motion is low ({total_motion:.1f}).")

    h, w = curr_img.shape[:2]
    reason_parts = [f"motion={total_motion:.1f}"]
    if valid_wrists:
        wrist_x = float(np.mean([pt[0] / w for pt in valid_wrists]))
        wrist_y = float(np.mean([pt[1] / h for pt in valid_wrists]))
        reason_parts.append(f"wrist_center=({wrist_x:.2f},{wrist_y:.2f})")
        if center_motion > 18.0:
            return Suggestion("sew", 0.66, "Pose wrists are active near the work area; " + ", ".join(reason_parts) + ".")
        if horizontal_bias > 4.0:
            return Suggestion("get", 0.54, "Pose and motion suggest material retrieval; " + ", ".join(reason_parts) + ".")
        if horizontal_bias < -4.0:
            return Suggestion("put", 0.54, "Pose and motion suggest material placement; " + ", ".join(reason_parts) + ".")

    if valid_shoulders and center_motion > 18.0:
        return Suggestion("sew", 0.58, "Upper-body pose is stable while center work-area motion is elevated.")

    return Suggestion("uncertain", 0.4, "Pose was detected, but the get/put/sew distinction remains ambiguous.")


def main() -> int:
    args = parse_args()
    queue_path = Path(args.queue_csv)
    with queue_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ensure_columns(rows)

    backend = choose_backend(args.backend)
    model = None
    if backend == "ultralytics":
        try:
            from ultralytics import YOLO

            model = YOLO(args.pose_model)
        except Exception as exc:
            print(f"Ultralytics backend unavailable ({exc}); falling back to motion-only suggestions.")
            backend = "motion"

    updated = 0
    for row in rows:
        if args.limit and updated >= args.limit:
            break
        if row.get("review_status", "") in ACCEPTED_STATUSES:
            continue
        if row.get("pose_label") and not args.overwrite:
            continue

        prev_img = read_image(row.get("prev_crop_path", ""))
        curr_img = read_image(row.get("crop_path", ""))
        next_img = read_image(row.get("next_crop_path", ""))
        suggestion = (
            suggest_from_pose_and_motion(model, prev_img, curr_img, next_img)
            if backend == "ultralytics"
            else suggest_from_motion(prev_img, curr_img, next_img)
        )
        if suggestion.label not in LABELS:
            suggestion = Suggestion("uncertain", 0.0, "Generated label was invalid.")
        row["pose_label"] = suggestion.label
        row["pose_confidence"] = f"{suggestion.confidence:.3f}"
        row["pose_reason"] = suggestion.reason
        updated += 1

    with queue_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QUEUE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {updated} rows in {queue_path} using backend={backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
