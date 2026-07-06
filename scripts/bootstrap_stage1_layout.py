#!/usr/bin/env python3
"""Create the folder scaffold for the Stage 1 presence-detection pipeline."""

from pathlib import Path


DIRS = [
    "datasets/interim/roi_crops",
    "datasets/interim/frame_exports",
    "datasets/processed/stage1",
    "datasets/processed/stage1/splits",
    "datasets/processed/stage1/manifests",
    "artifacts/stage1/models",
    "artifacts/stage1/eval",
    "artifacts/stage1/visualizations",
]


def main() -> int:
    for rel in DIRS:
        Path(rel).mkdir(parents=True, exist_ok=True)
    print("Stage 1 layout created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
