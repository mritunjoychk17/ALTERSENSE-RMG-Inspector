#!/usr/bin/env python3
"""Create small contact sheets per workstation from the seed person images."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw
from PIL import ImageFile


CELL_SIZE = (220, 220)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="datasets/processed/stage1/manifests/station_seed_images.csv")
    parser.add_argument("--output-dir", default="artifacts/stage1/visualizations/station_contact_sheets")
    return parser.parse_args()


def fit_image(path: Path) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return ImageOps.contain(im, CELL_SIZE)


def paste_center(canvas: Image.Image, tile: Image.Image, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    cell_w = x2 - x1
    cell_h = y2 - y1
    offset_x = x1 + (cell_w - tile.width) // 2
    offset_y = y1 + (cell_h - tile.height) // 2
    canvas.paste(tile, (offset_x, offset_y))


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    with open(args.manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            grouped[row["station_id"]][row["label"]].append(Path(row["image_path"]))

    for station_id, label_map in sorted(grouped.items(), key=lambda kv: int(kv[0])):
        labels = ["present", "absent"]
        cols = 5
        rows = len(labels)
        canvas = Image.new("RGB", (cols * CELL_SIZE[0], rows * CELL_SIZE[1] + 50), color=(248, 248, 248))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 12), f"Station {station_id}: present vs absent seeds", fill=(20, 20, 20))

        for row_idx, label in enumerate(labels):
            images = sorted(label_map.get(label, []))[:cols]
            for col_idx, image_path in enumerate(images):
                x1 = col_idx * CELL_SIZE[0]
                y1 = 50 + row_idx * CELL_SIZE[1]
                x2 = x1 + CELL_SIZE[0]
                y2 = y1 + CELL_SIZE[1]
                draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=(180, 180, 180), width=1)
                tile = fit_image(image_path)
                paste_center(canvas, tile, (x1, y1, x2, y2))
            draw.text((12, 50 + row_idx * CELL_SIZE[1] + 8), label, fill=(20, 20, 20))

        out_path = output_dir / f"station_{station_id}.jpg"
        canvas.save(out_path, quality=92)
        print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
