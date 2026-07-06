#!/usr/bin/env python3
"""Train temporal Stage 2 models: GRU, TCN, hybrid clip+pose, or 3D CNN."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
import sys

from PIL import Image, ImageFile
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage2_temporal_models import POSE_LABELS, build_model, encode_pose_label, pose_feature_dim


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-csv", required=True)
    parser.add_argument("--output-dir", default="artifacts/stage2/models/clip_temporal")
    parser.add_argument("--model-type", choices=["gru", "tcn", "hybrid_pose", "cnn3d"], default="gru")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-weighted-sampler", action="store_true")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available in this environment. Falling back to CPU.")
        return "cpu"
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tf(image_size: int) -> transforms.Compose:
    return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def target_clip_len_from_rows(rows: list[dict]) -> int:
    lengths = Counter()
    for row in rows:
        clip_paths = [item for item in (row.get("clip_paths") or "").split("|") if item.strip()]
        if clip_paths:
            lengths[len(clip_paths)] += 1
            continue
        clip_len = parse_int(row.get("clip_len", ""), 0)
        if clip_len > 0:
            lengths[clip_len] += 1
    if not lengths:
        return 0
    return lengths.most_common(1)[0][0]


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
    label_vec = encode_pose_label((row.get("pose_label") or "").strip().lower())
    conf = parse_float(row.get("pose_confidence", ""), default=0.0)
    frame_vec = label_vec + [conf]
    return torch.tensor([frame_vec for _ in range(clip_len)], dtype=torch.float32)


class ClipDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        class_to_idx: dict[str, int],
        tf: transforms.Compose,
        model_type: str,
        target_clip_len: int,
    ) -> None:
        self.rows = rows
        self.class_to_idx = class_to_idx
        self.tf = tf
        self.model_type = model_type
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
        return clip, pose, self.class_to_idx[row["label"]]


def split_rows(rows: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    items = rows[:]
    rng = random.Random(seed)
    rng.shuffle(items)
    val_len = max(1, int(len(items) * val_ratio))
    return items[val_len:], items[:val_len]


def run_model(model: nn.Module, clips: torch.Tensor, pose: torch.Tensor, model_type: str) -> torch.Tensor:
    if model_type == "hybrid_pose":
        return model(clips, pose_features=pose)
    return model(clips)


def eval_model(model: nn.Module, loader: DataLoader, device: str, criterion: nn.Module, model_type: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for clips, pose, labels in loader:
            clips = clips.to(device)
            pose = pose.to(device)
            labels = labels.to(device)
            logits = run_model(model, clips, pose, model_type)
            loss = criterion(logits, labels)
            total_loss += loss.item() * clips.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += clips.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)
    rows = list(csv.DictReader(open(args.clip_csv, newline="", encoding="utf-8")))
    rows = [row for row in rows if row.get("label")]
    if not rows:
        raise ValueError("No labeled clip rows found.")
    target_clip_len = target_clip_len_from_rows(rows)
    if target_clip_len <= 0:
        raise ValueError("Could not infer a valid target clip length from the manifest.")
    class_names = sorted({row["label"] for row in rows})
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    train_rows, val_rows = split_rows(rows, args.val_ratio, args.seed)
    tf = build_tf(args.image_size)
    train_ds = ClipDataset(train_rows, class_to_idx, tf, args.model_type, target_clip_len=target_clip_len)
    val_ds = ClipDataset(val_rows, class_to_idx, tf, args.model_type, target_clip_len=target_clip_len)

    sampler = None
    if args.use_weighted_sampler:
        counts = Counter(row["label"] for row in train_rows)
        weights = [1.0 / counts[row["label"]] for row in train_rows]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(args.model_type, num_classes=len(class_names)).to(device)
    train_counts = Counter(row["label"] for row in train_rows)
    total_train = sum(train_counts.values())
    class_weights = torch.tensor([total_train / max(train_counts[name], 1) for name in class_names], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_acc = -1.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for clips, pose, labels in train_loader:
            clips = clips.to(device)
            pose = pose.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = run_model(model, clips, pose, args.model_type)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * clips.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += clips.size(0)
        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = eval_model(model, val_loader, device, criterion, args.model_type)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
        }
        history.append(row)
        print(
            f"model={args.model_type} epoch={epoch} train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": class_names,
                    "history": history,
                    "args": vars(args),
                    "target_clip_len": target_clip_len,
                    "model_type": args.model_type,
                    "pose_labels": POSE_LABELS,
                    "pose_feature_dim": pose_feature_dim(),
                },
                output_dir / "best.pt",
            )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
