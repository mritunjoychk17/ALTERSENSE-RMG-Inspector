#!/usr/bin/env python3
"""Extract labeled Stage 2 activity images from Work 2 into a clean dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from zipfile import ZipFile


VALID_EXTS = {".jpg", ".jpeg", ".png"}
CLASS_MAP = {"get_put": "get_put", "sew": "sew"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", default="Work 2-20260625T082324Z-3-001.zip")
    parser.add_argument("--output-dir", default="datasets/raw/activity")
    parser.add_argument("--manifest", default="datasets/manifests/activity_extraction_manifest.csv")
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def infer_label(member_name: str) -> str | None:
    lowered = member_name.lower()
    for key, value in CLASS_MAP.items():
        if f"/{key}/" in lowered:
            return value
    return None


def main() -> int:
    args = parse_args()
    zip_path = Path(args.zip_path)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[list[str]] = []
    seen_hashes: set[str] = set()

    with ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            suffix = Path(member.filename).suffix.lower()
            if suffix not in VALID_EXTS:
                continue
            label = infer_label(member.filename)
            if not label:
                continue

            data = zf.read(member)
            digest = sha256_bytes(data)
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)

            target_dir = output_dir / label
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / f"{digest}{suffix}"
            if not target_path.exists():
                target_path.write_bytes(data)

            rows.append(
                [
                    label,
                    member.filename,
                    digest,
                    str(target_path),
                ]
            )

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "source_path", "sha256", "output_path"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} activity images to {output_dir}")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
