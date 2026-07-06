#!/usr/bin/env python3
"""Train a simple MobileNetV3-Small Stage 2 activity classifier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

from PIL import ImageFile
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets/raw/activity")
    parser.add_argument("--output-dir", default="artifacts/stage2/models/mobilenet_seed")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.15),
            transforms.ToTensor(),
        ]
    )
    val_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    return train_tf, val_tf


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

    train_tf, val_tf = build_transforms()
    base_dataset = datasets.ImageFolder(args.data_dir)
    class_names = base_dataset.classes
    if set(class_names) != {"get_put", "sew"}:
        raise ValueError(f"Expected classes ['get_put', 'sew'], found {class_names}")

    val_len = max(1, int(len(base_dataset) * args.val_ratio))
    train_len = len(base_dataset) - val_len
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(base_dataset, [train_len, val_len], generator=generator)

    train_dataset = datasets.ImageFolder(args.data_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(args.data_dir, transform=val_tf)
    train_dataset.samples = [train_dataset.samples[i] for i in train_subset.indices]
    train_dataset.targets = [train_dataset.targets[i] for i in train_subset.indices]
    train_dataset.imgs = train_dataset.samples
    val_dataset.samples = [val_dataset.samples[i] for i in val_subset.indices]
    val_dataset.targets = [val_dataset.targets[i] for i in val_subset.indices]
    val_dataset.imgs = val_dataset.samples

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    except Exception as exc:
        print(f"Falling back to random initialization because pretrained weights could not be loaded: {exc}")
        model = models.mobilenet_v3_small(weights=None)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)

    history = []
    best_acc = -1.0
    best_path = output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
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
            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        train_loss = running_loss / max(total, 1)
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

    with (output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label", "image_path"])
        for idx in train_subset.indices:
            path, target = base_dataset.samples[idx]
            writer.writerow(["train", class_names[target], path])
        for idx in val_subset.indices:
            path, target = base_dataset.samples[idx]
            writer.writerow(["val", class_names[target], path])

    print(best_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
