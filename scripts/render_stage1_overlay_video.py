#!/usr/bin/env python3
"""Render Stage 1 predictions as overlays on a source video."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--config", default="configs/roi_annotations.template.json")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--present-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="artifacts/stage1/visualizations/stage1_overlay.mp4")
    parser.add_argument("--max-frames", type=int, default=0)
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


def predict_crop(
    model: nn.Module,
    crop_bgr: np.ndarray,
    tf: transforms.Compose,
    device: str,
    classes: list[str],
    present_threshold: float,
) -> tuple[str, float, float]:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    tensor = tf(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    present_idx = classes.index("present")
    absent_idx = classes.index("absent")
    present_conf = float(probs[present_idx])
    absent_conf = float(probs[absent_idx])
    label = "present" if present_conf >= present_threshold else "absent"
    confidence = present_conf if label == "present" else absent_conf
    return label, confidence, present_conf


def station_color(label: str, present_conf: float, threshold: float) -> tuple[int, int, int]:
    if abs(present_conf - threshold) < 0.08:
        return (0, 215, 255)
    if label == "present":
        return (0, 200, 0)
    return (0, 0, 220)


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint["classes"]
    model = build_model(checkpoint, device)
    tf = preprocess()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    video = next(v for v in config["videos"] if v["video_id"] == args.video_id)
    stations = [ws for ws in video["workstations"] if ws["station_roi_polygon"]]
    if not stations:
        raise ValueError(f"No annotated stations found for {args.video_id}")

    cap = cv2.VideoCapture(video["video_path"])
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video['video_path']}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    station_data = []
    for ws in stations:
        mask = mask_from_station(frame0.shape[:2], ws)
        bbox = bbox_from_polygon(ws["station_roi_polygon"])
        station_data.append((ws, mask, bbox))

    frame_index = 0
    cached_predictions: dict[str, tuple[str, float, float]] = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and frame_index >= args.max_frames:
            break

        draw = frame.copy()
        if frame_index % args.sample_every == 0:
            cached_predictions = {}
            for ws, mask, bbox in station_data:
                x1, y1, x2, y2 = bbox
                crop = cv2.bitwise_and(frame, frame, mask=mask)[y1:y2, x1:x2]
                cached_predictions[ws["station_id"]] = predict_crop(
                    model, crop, tf, device, classes, args.present_threshold
                )

        for ws, _mask, bbox in station_data:
            x1, y1, x2, y2 = bbox
            label, confidence, present_conf = cached_predictions.get(
                ws["station_id"], ("unknown", 0.0, 0.0)
            )
            color = station_color(label, present_conf, args.present_threshold)
            pts = np.array(ws["station_roi_polygon"], dtype=np.int32)
            overlay = draw.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.12, draw, 0.88, 0, draw)
            cv2.polylines(draw, [pts], True, color, 2, cv2.LINE_AA)
            text = f"S{ws['station_id']} {label} p={present_conf:.2f}"
            cv2.putText(draw, text, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        banner = f"{args.video_id} frame={frame_index} sample_every={args.sample_every} threshold={args.present_threshold:.2f}"
        cv2.putText(draw, banner, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(draw)
        frame_index += 1

    cap.release()
    writer.release()
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
