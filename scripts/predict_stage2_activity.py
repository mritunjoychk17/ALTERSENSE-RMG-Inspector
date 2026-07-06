#!/usr/bin/env python3
"""Run Stage 2 activity predictions on images or manifests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageFile
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/stage2/models/mobilenet_seed/best.pt")
    parser.add_argument("--image", help="Predict one image.")
    parser.add_argument("--image-dir", help="Predict all jpg/png files in a directory.")
    parser.add_argument("--manifest", help="Predict images listed in a CSV containing image_path or crop_path.")
    parser.add_argument("--smoothing-window", type=int, default=1, help="Odd window size for majority smoothing across timestamp order.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="artifacts/stage2/eval/predictions.csv")
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


class PathDataset(Dataset):
    def __init__(self, items: list[dict], tf: transforms.Compose) -> None:
        self.items = items
        self.tf = tf

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        image = Image.open(item["path"]).convert("RGB")
        return self.tf(image), index


def predict_batch(model: nn.Module, tensors: torch.Tensor, classes: list[str]) -> tuple[list[str], list[float], list[dict[str, float]]]:
    with torch.no_grad():
        probs = torch.softmax(model(tensors), dim=1).cpu().numpy()
    labels = []
    confidences = []
    prob_rows = []
    for row in probs:
        best_idx = int(row.argmax())
        labels.append(classes[best_idx])
        confidences.append(float(row[best_idx]))
        prob_rows.append({classes[i]: float(row[i]) for i in range(len(classes))})
    return labels, confidences, prob_rows


def apply_majority_smoothing(rows: list[dict], window: int) -> list[dict]:
    if window <= 1 or not rows:
        return rows
    half = window // 2
    ordered = rows[:]
    for i, row in enumerate(ordered):
        left = max(0, i - half)
        right = min(len(ordered), i + half + 1)
        labels = [ordered[j]["predicted_label"] for j in range(left, right)]
        best = max(set(labels), key=labels.count)
        row["smoothed_label"] = best
    return ordered


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint["classes"]
    model = build_model(checkpoint, device)
    tf = preprocess()

    items: list[dict] = []
    if args.image:
        items.append({"path": Path(args.image), "extra": {}})
    if args.image_dir:
        for path in sorted(Path(args.image_dir).glob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.append({"path": path, "extra": {}})
    if args.manifest:
        with open(args.manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                path_value = row.get("image_path") or row.get("crop_path")
                if not path_value:
                    continue
                items.append(
                    {
                        "path": Path(path_value),
                        "extra": {
                            "frame_index": row.get("frame_index", ""),
                            "timestamp_sec": row.get("timestamp_sec", ""),
                            "video_id": row.get("video_id", ""),
                            "station_id": row.get("station_id", ""),
                        },
                    }
                )

    rows: list[dict] = [None] * len(items)
    if items:
        dataset = PathDataset(items, tf)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device == "cuda"),
        )
        for images, indices in loader:
            images = images.to(device, non_blocking=True)
            labels, confidences, prob_rows = predict_batch(model, images, classes)
            for idx, label, conf, probs in zip(indices.tolist(), labels, confidences, prob_rows):
                row = {"source": str(items[idx]["path"]), "predicted_label": label, "confidence": round(conf, 6)}
                for class_name, value in probs.items():
                    row[f"{class_name}_confidence"] = round(value, 6)
                row.update(items[idx]["extra"])
                rows[idx] = row
        rows = [row for row in rows if row is not None]

    if any(row.get("timestamp_sec", "") not in {"", None} for row in rows):
        rows = sorted(rows, key=lambda r: float(r.get("timestamp_sec") or 0.0))
        rows = apply_majority_smoothing(rows, args.smoothing_window)
    else:
        for row in rows:
            row["smoothed_label"] = row["predicted_label"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    class_fields = [f"{name}_confidence" for name in classes]
    fieldnames = ["source", "video_id", "station_id", "frame_index", "timestamp_sec", "predicted_label", "smoothed_label", "confidence"] + class_fields
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(output_path)
    if rows:
        print(rows[:5])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
