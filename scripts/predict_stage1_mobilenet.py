#!/usr/bin/env python3
"""Run Stage 1 MobileNetV3 presence predictions on images or ROI crops."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile
import torch
from torch import nn
from torchvision import models, transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/stage1/models/mobilenet_seed/best.pt")
    parser.add_argument("--image", help="Predict one image.")
    parser.add_argument("--image-dir", help="Predict all jpg/png files in a directory.")
    parser.add_argument("--manifest", help="Predict images listed in a CSV containing crop_path or image_path.")
    parser.add_argument("--video-id", help="Predict sampled frames from one video using saved ROIs.")
    parser.add_argument("--station-id", help="Required with --video-id.")
    parser.add_argument("--config", default="configs/roi_annotations.template.json")
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--present-threshold", type=float, default=0.5)
    parser.add_argument("--output", default="artifacts/stage1/eval/predictions.csv")
    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available in this environment. Falling back to CPU.")
        return "cpu"
    return requested_device


def build_model(checkpoint: dict, device: str) -> nn.Module:
    classes = checkpoint["classes"]
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess() -> transforms.Compose:
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


def predict_pil(
    model: nn.Module,
    image: Image.Image,
    tf: transforms.Compose,
    device: str,
    classes: list[str],
    present_threshold: float,
) -> tuple[str, float, float, float]:
    tensor = tf(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    present_idx = classes.index("present")
    absent_idx = classes.index("absent")
    present_conf = float(probs[present_idx])
    absent_conf = float(probs[absent_idx])
    label = "present" if present_conf >= present_threshold else "absent"
    predicted_conf = present_conf if label == "present" else absent_conf
    return label, predicted_conf, present_conf, absent_conf


def mask_from_station(frame_shape: tuple[int, int], station: dict) -> np.ndarray:
    roi = np.zeros(frame_shape, dtype=np.uint8)
    cv2.fillPoly(roi, [np.array(station["station_roi_polygon"], dtype=np.int32)], 255)
    machine = np.zeros(frame_shape, dtype=np.uint8)
    for poly in station["machine_mask_polygons"]:
        if poly:
            cv2.fillPoly(machine, [np.array(poly, dtype=np.int32)], 255)
    return cv2.bitwise_and(roi, cv2.bitwise_not(machine))


def bbox_from_polygon(points: list[list[int]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def predict_video_station(
    model: nn.Module,
    tf: transforms.Compose,
    device: str,
    classes: list[str],
    config_path: Path,
    video_id: str,
    station_id: str,
    sample_every: int,
    max_frames: int,
    present_threshold: float,
) -> list[dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    video = next(v for v in config["videos"] if v["video_id"] == video_id)
    station = next(s for s in video["workstations"] if s["station_id"] == station_id)
    if not station["station_roi_polygon"]:
        raise ValueError(f"No ROI saved for {video_id} station {station_id}")

    cap = cv2.VideoCapture(video["video_path"])
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read {video['video_path']}")
    mask = mask_from_station(frame0.shape[:2], station)
    x1, y1, x2, y2 = bbox_from_polygon(station["station_roi_polygon"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    rows = []
    frame_index = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % sample_every != 0:
            frame_index += 1
            continue
        if max_frames and saved >= max_frames:
            break
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        crop = masked[y1:y2, x1:x2]
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        label, conf, present_conf, absent_conf = predict_pil(
            model, image, tf, device, classes, present_threshold
        )
        rows.append(
            {
                "source": f"{video_id}:station_{station_id}",
                "frame_index": frame_index,
                "timestamp_sec": round(frame_index / fps, 3),
                "predicted_label": label,
                "confidence": round(conf, 6),
                "present_confidence": round(present_conf, 6),
                "absent_confidence": round(absent_conf, 6),
            }
        )
        frame_index += 1
        saved += 1
    cap.release()
    return rows


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint["classes"]
    model = build_model(checkpoint, device)
    tf = preprocess()

    rows: list[dict] = []

    if args.image:
        path = Path(args.image)
        label, conf, present_conf, absent_conf = predict_pil(
            model, Image.open(path), tf, device, classes, args.present_threshold
        )
        rows.append(
            {
                "source": str(path),
                "frame_index": "",
                "timestamp_sec": "",
                "predicted_label": label,
                "confidence": round(conf, 6),
                "present_confidence": round(present_conf, 6),
                "absent_confidence": round(absent_conf, 6),
            }
        )

    if args.image_dir:
        for path in sorted(Path(args.image_dir).glob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label, conf, present_conf, absent_conf = predict_pil(
                model, Image.open(path), tf, device, classes, args.present_threshold
            )
            rows.append(
                {
                    "source": str(path),
                    "frame_index": "",
                    "timestamp_sec": "",
                    "predicted_label": label,
                    "confidence": round(conf, 6),
                    "present_confidence": round(present_conf, 6),
                    "absent_confidence": round(absent_conf, 6),
                }
            )

    if args.manifest:
        with open(args.manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                path_value = row.get("crop_path") or row.get("image_path")
                if not path_value:
                    continue
                path = Path(path_value)
                label, conf, present_conf, absent_conf = predict_pil(
                    model, Image.open(path), tf, device, classes, args.present_threshold
                )
                rows.append(
                    {
                        "source": str(path),
                        "frame_index": row.get("frame_index", ""),
                        "timestamp_sec": row.get("timestamp_sec", ""),
                        "predicted_label": label,
                        "confidence": round(conf, 6),
                        "present_confidence": round(present_conf, 6),
                        "absent_confidence": round(absent_conf, 6),
                    }
                )

    if args.video_id:
        if not args.station_id:
            raise ValueError("--station-id is required with --video-id")
        rows.extend(
            predict_video_station(
                model,
                tf,
                device,
                classes,
                Path(args.config),
                args.video_id,
                args.station_id,
                args.sample_every,
                args.max_frames,
                args.present_threshold,
            )
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "frame_index",
                "timestamp_sec",
                "predicted_label",
                "confidence",
                "present_confidence",
                "absent_confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(output_path)
    if rows:
        print(rows[:5])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
