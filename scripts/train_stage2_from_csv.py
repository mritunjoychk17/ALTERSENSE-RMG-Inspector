#!/usr/bin/env python3
"""Train Stage 2 activity classifier from reviewed CSV labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import random

from PIL import Image, ImageFile
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, models, transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True


class CsvImageDataset(Dataset):
    def __init__(self, rows: list[dict], class_to_idx: dict[str, int], tf: transforms.Compose) -> None:
        self.rows = rows
        self.class_to_idx = class_to_idx
        self.tf = tf

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.tf(image), self.class_to_idx[row["label"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", required=True, help="Comma-separated reviewed CSV files.")
    parser.add_argument("--seed-dir", default="datasets/raw/activity")
    parser.add_argument("--output-dir", default="artifacts/stage2/models/mobilenet_reviewed")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include-seed-images", action="store_true")
    parser.add_argument(
        "--split-mode",
        default="video_station",
        choices=["random", "video", "station", "video_station"],
        help="How to split reviewed activity rows into train/val to reduce leakage.",
    )
    parser.add_argument(
        "--accepted-status",
        default="done",
        help="Comma-separated review_status values to accept.",
    )
    parser.add_argument("--use-weighted-sampler", action="store_true", help="Balance rare activity classes during training.")
    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available in this environment. Falling back to CPU.")
        return "cpu"
    return requested_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(6),
            transforms.ColorJitter(brightness=0.18, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
        ]
    )
    val_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    return train_tf, val_tf


def read_review_rows(path: Path, accepted_status: str) -> list[dict]:
    rows = []
    accepted = {item.strip() for item in accepted_status.split(",") if item.strip()}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row.get("final_label") or row.get("label")
            if not label:
                continue
            if accepted and row.get("review_status", "") not in accepted:
                continue
            image_path = row.get("crop_path") or row.get("image_path")
            if not image_path:
                continue
            rows.append(
                {
                    "image_path": image_path,
                    "label": label,
                    "video_id": row.get("video_id", ""),
                    "station_id": row.get("station_id", ""),
                    "frame_index": row.get("frame_index", ""),
                    "review_status": row.get("review_status", ""),
                }
            )
    return rows


def read_review_rows_multi(value: str, accepted_status: str) -> list[dict]:
    rows: list[dict] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        rows.extend(read_review_rows(Path(item), accepted_status))
    return rows


class LabeledPathDataset(Dataset):
    def __init__(self, rows: list[dict], class_to_idx: dict[str, int], tf: transforms.Compose) -> None:
        self.rows = rows
        self.class_to_idx = class_to_idx
        self.tf = tf

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.tf(image), self.class_to_idx[row["label"]]


def load_seed_rows(seed_dir: Path, class_names: list[str]) -> list[dict]:
    alias_map = {
        "get_put": ["get", "put"],
        "sew": ["sew"],
        "idle": ["idle"],
        "align": ["align"],
        "inspect": ["inspect"],
        "pass": ["pass"],
    }
    allowed = set(class_names)
    rows: list[dict] = []
    if not seed_dir.exists():
        return rows
    for cls_dir in sorted(path for path in seed_dir.iterdir() if path.is_dir()):
        mapped = [label for label in alias_map.get(cls_dir.name, [cls_dir.name]) if label in allowed]
        if not mapped:
            continue
        image_paths = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for image_path in image_paths:
            for label in mapped:
                rows.append(
                    {
                        "image_path": str(image_path),
                        "label": label,
                        "video_id": "seed_images",
                        "station_id": cls_dir.name,
                        "frame_index": "",
                        "review_status": "seed",
                    }
                )
    return rows


def split_key(row: dict, split_mode: str) -> str:
    if split_mode == "video":
        return row["video_id"]
    if split_mode == "station":
        return row["station_id"]
    if split_mode == "video_station":
        return f"{row['video_id']}::{row['station_id']}"
    return f"row::{row['image_path']}"


def split_review_rows(rows: list[dict], val_ratio: float, split_mode: str, seed: int) -> tuple[list[dict], list[dict]]:
    if split_mode == "random":
        shuffled = rows[:]
        rng = random.Random(seed)
        rng.shuffle(shuffled)
        val_len = max(1, int(len(shuffled) * val_ratio))
        return shuffled[val_len:], shuffled[:val_len]

    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(split_key(row, split_mode), []).append(row)
    keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    target_val = max(1, int(len(rows) * val_ratio))
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    current_val = 0
    for key in keys:
        bucket = groups[key]
        if current_val < target_val:
            val_rows.extend(bucket)
            current_val += len(bucket)
        else:
            train_rows.extend(bucket)
    if not train_rows or not val_rows:
        raise ValueError(
            f"Split mode '{split_mode}' produced an empty train or val set. "
            "Use more reviewed rows or a less strict split mode."
        )
    return train_rows, val_rows


def evaluate(model: nn.Module, loader: DataLoader, device: str, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    review_rows = read_review_rows_multi(args.review_csv, args.accepted_status)
    if not review_rows:
        raise ValueError(
            "No accepted activity review rows found. Fill final_label and mark review_status with one of the accepted statuses."
        )

    class_names = sorted({row["label"] for row in review_rows})
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    train_tf, val_tf = build_transforms()

    train_rows, val_rows = split_review_rows(review_rows, args.val_ratio, args.split_mode, args.seed)
    seed_rows: list[dict] = []
    if args.include_seed_images:
        seed_rows = load_seed_rows(Path(args.seed_dir), class_names)
        if seed_rows:
            train_rows = train_rows + seed_rows
    train_dataset = CsvImageDataset(train_rows, class_to_idx, train_tf)
    val_dataset = CsvImageDataset(val_rows, class_to_idx, val_tf)

    sampler = None
    if args.use_weighted_sampler:
        sample_weights = []
        train_counts = Counter(row["label"] for row in train_rows)
        for row in train_rows:
            sample_weights.append(1.0 / max(train_counts[row["label"]], 1))
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))

    seed_ckpt_path = Path("artifacts/stage2/models/mobilenet_seed/best.pt")
    if seed_ckpt_path.exists():
        ckpt = torch.load(seed_ckpt_path, map_location="cpu")
        if ckpt.get("classes") == class_names:
            model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    train_counts = Counter(row["label"] for row in train_rows)
    total_train = sum(train_counts.values())
    class_weights = torch.tensor(
        [total_train / max(train_counts[name], 1) for name in class_names],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_acc = -1.0
    best_path = output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
            }
        )
        print(f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": class_names,
                    "history": history,
                    "args": vars(args),
                },
                best_path,
            )

    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        f.write("\n")

    with (output_dir / "review_rows_used.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "label", "video_id", "station_id", "frame_index", "review_status"],
        )
        writer.writeheader()
        writer.writerows(review_rows)

    with (output_dir / "review_split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "image_path", "label", "video_id", "station_id", "frame_index", "review_status"],
        )
        writer.writeheader()
        for row in train_rows:
            writer.writerow({"split": "train", **row})
        for row in val_rows:
            writer.writerow({"split": "val", **row})

    print(best_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
