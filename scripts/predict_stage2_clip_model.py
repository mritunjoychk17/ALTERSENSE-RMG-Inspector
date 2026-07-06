#!/usr/bin/env python3
"""Run temporal Stage 2 predictions for GRU, TCN, hybrid clip+pose, or 3D CNN checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

from PIL import Image, ImageFile
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage2_temporal_models import POSE_LABELS, build_model, encode_pose_label, pose_feature_dim


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clip-csv", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--flush-every", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-label-column", default="predicted_label")
    parser.add_argument("--output-smoothed-column", default="smoothed_label")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available in this environment. Falling back to CPU.")
        return "cpu"
    return device


def build_tf(image_size: int) -> transforms.Compose:
    return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_clip_paths(paths: list[str], target_clip_len: int) -> list[str]:
    paths = [path for path in paths if path]
    if not paths or target_clip_len <= 0 or len(paths) == target_clip_len:
        return paths
    if len(paths) > target_clip_len:
        start = max(0, (len(paths) - target_clip_len) // 2)
        return paths[start : start + target_clip_len]
    while len(paths) < target_clip_len:
        paths.append(paths[-1])
    return paths


def clip_pose_features(row: dict, clip_len: int) -> torch.Tensor:
    label = (row.get("pose_label") or "").strip().lower()
    conf = parse_float(row.get("pose_confidence", ""), default=0.0)
    frame_vec = encode_pose_label(label) + [conf]
    return torch.tensor([frame_vec for _ in range(clip_len)], dtype=torch.float32)


class ClipPredictDataset(Dataset):
    def __init__(self, rows: list[dict], tf: transforms.Compose, target_clip_len: int) -> None:
        self.rows = rows
        self.tf = tf
        self.target_clip_len = target_clip_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        frames = []
        paths = normalize_clip_paths(row["clip_paths"].split("|"), self.target_clip_len)
        for path in paths:
            image = Image.open(path).convert("RGB")
            frames.append(self.tf(image))
        clip = torch.stack(frames, dim=0)
        pose = clip_pose_features(row, len(paths))
        return clip, pose, index


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def run_model(model, clips: torch.Tensor, pose: torch.Tensor, model_type: str):
    if model_type == "hybrid_pose":
        return model(clips, pose_features=pose)
    return model(clips)


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint["classes"]
    model_type = checkpoint.get("model_type", "gru")
    image_size = int(checkpoint.get("args", {}).get("image_size", 224))
    target_clip_len = int(checkpoint.get("target_clip_len") or checkpoint.get("args", {}).get("target_clip_len", 0) or 0)
    model = build_model(model_type, num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    rows = list(csv.DictReader(open(args.clip_csv, newline="", encoding="utf-8")))
    ds = ClipPredictDataset(rows, build_tf(image_size), target_clip_len=target_clip_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device == "cuda"))
    out = [None] * len(rows)
    completed = 0
    started_at = time.time()
    output_path = Path(args.output)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    fieldnames = list(rows[0].keys()) if rows else []

    with torch.no_grad():
        for batch_index, (clips, pose, indices) in enumerate(loader, start=1):
            clips = clips.to(device)
            pose = pose.to(device)
            probs = torch.softmax(run_model(model, clips, pose, model_type), dim=1).cpu().numpy()
            for idx, row_probs in zip(indices.tolist(), probs):
                best = int(row_probs.argmax())
                row = dict(rows[idx])
                row[args.output_label_column] = classes[best]
                row[args.output_smoothed_column] = classes[best]
                row["confidence"] = round(float(row_probs[best]), 6)
                for i, name in enumerate(classes):
                    row[f"{name}_confidence"] = round(float(row_probs[i]), 6)
                out[idx] = row
                completed += 1

            if any(item is not None for item in out) and (completed % args.flush_every == 0 or batch_index % args.progress_every == 0):
                partial_rows = [row for row in out if row is not None]
                if partial_rows:
                    write_rows(partial_path, partial_rows, list(partial_rows[0].keys()))

            if batch_index % args.progress_every == 0:
                elapsed = max(time.time() - started_at, 1e-6)
                rate = completed / elapsed
                print(f"model={model_type} batch={batch_index}/{len(loader)} rows={completed}/{len(rows)} rate={rate:.2f} rows_sec device={device}", flush=True)

    out_rows = [row for row in out if row is not None]
    fieldnames = list(out_rows[0].keys()) if out_rows else fieldnames
    write_rows(output_path, out_rows, fieldnames)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
