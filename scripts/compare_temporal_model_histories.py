#!/usr/bin/env python3
"""Summarize temporal model training histories side by side."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dirs", nargs="+", help="Directories that contain history.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for model_dir in args.model_dirs:
        history_path = Path(model_dir) / "history.json"
        if not history_path.exists():
            rows.append({"model_dir": model_dir, "status": "missing history.json"})
            continue
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not history:
            rows.append({"model_dir": model_dir, "status": "empty history"})
            continue
        best = max(history, key=lambda row: row.get("val_acc", -1.0))
        final = history[-1]
        rows.append(
            {
                "model_dir": model_dir,
                "best_epoch": best.get("epoch"),
                "best_val_acc": best.get("val_acc"),
                "best_val_loss": best.get("val_loss"),
                "final_train_acc": final.get("train_acc"),
                "final_val_acc": final.get("val_acc"),
            }
        )

    headers = ["model_dir", "best_epoch", "best_val_acc", "best_val_loss", "final_train_acc", "final_val_acc", "status"]
    print("\t".join(headers))
    for row in rows:
        print("\t".join(str(row.get(key, "")) for key in headers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
