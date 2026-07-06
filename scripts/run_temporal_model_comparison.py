#!/usr/bin/env python3
"""Train Stage 2 temporal models with shared settings and summarize results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-csv", required=True)
    parser.add_argument("--output-root", default="artifacts/stage2/models")
    parser.add_argument("--run-name", default="temporal_compare")
    parser.add_argument("--model-types", nargs="+", default=["hybrid_pose", "cnn3d"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-weighted-sampler", action="store_true")
    return parser.parse_args()


def summarize_history(model_dir: Path) -> dict[str, object]:
    history_path = model_dir / "history.json"
    if not history_path.exists():
        return {"model_dir": str(model_dir), "status": "missing history.json"}
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not history:
        return {"model_dir": str(model_dir), "status": "empty history"}
    best = max(history, key=lambda row: row.get("val_acc", -1.0))
    final = history[-1]
    return {
        "model_dir": str(model_dir),
        "best_epoch": best.get("epoch"),
        "best_val_acc": best.get("val_acc"),
        "best_val_loss": best.get("val_loss"),
        "final_train_acc": final.get("train_acc"),
        "final_val_acc": final.get("val_acc"),
        "status": "ok",
    }


def train_one(args: argparse.Namespace, model_type: str, model_dir: Path) -> int:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_stage2_clip_model.py"),
        "--clip-csv",
        args.clip_csv,
        "--output-dir",
        str(model_dir),
        "--model-type",
        model_type,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--val-ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--image-size",
        str(args.image_size),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.use_weighted_sampler:
        cmd.append("--use-weighted-sampler")
    print(f"\n=== Training {model_type} ===", flush=True)
    print(" ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    return completed.returncode


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    headers = ["model_dir", "best_epoch", "best_val_acc", "best_val_loss", "final_train_acc", "final_val_acc", "status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(key, "")) for key in headers) + "\n")


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    summaries = []
    failed = []

    for model_type in args.model_types:
        model_dir = output_root / f"{args.run_name}_{model_type}_img{args.image_size}"
        code = train_one(args, model_type, model_dir)
        if code != 0:
            failed.append(model_type)
            summaries.append({"model_dir": str(model_dir), "status": f"train_failed_exit_{code}"})
            continue
        summaries.append(summarize_history(model_dir))

    summary_json = output_root / f"{args.run_name}_summary.json"
    summary_tsv = output_root / f"{args.run_name}_summary.tsv"
    summary_json.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    write_tsv(summary_tsv, summaries)

    print("\n=== Comparison Summary ===")
    print(summary_tsv)
    for row in summaries:
        print(row)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
