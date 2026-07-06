#!/usr/bin/env python3
"""Prepare a clean, deduplicated dataset from the provided zip archives.

Scope for now:
- Keep only person-model images for training.
- Extract videos as raw assets for future work.
- Ignore Activity model assets entirely.

Extraction is content-addressed by SHA-256, so identical files are stored once
even if they appear in multiple archives.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import zipfile


@dataclass(frozen=True)
class ExtractedItem:
    kind: str
    label: str
    archive: str
    source_path: str
    sha256: str
    size_bytes: int
    output_path: str


def sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def is_interesting_member(name: str) -> bool:
    if not name or name.endswith("/"):
        return False
    if "__MACOSX" in name or "/._" in name or name.endswith(".DS_Store"):
        return False
    return ("person model/" in name) or ("/Video/" in name)


def infer_kind_and_label(name: str) -> tuple[str, str] | None:
    if "person model/" in name:
        rel = name.split("person model/", 1)[1]
        parts = rel.split("/")
        if len(parts) < 2:
            return None
        label = parts[0].strip().lower()
        if label not in {"present", "absent"}:
            return None
        return "person", label
    if "/Video/" in name:
        return "video", "raw"
    return None


def output_extension(source_name: str) -> str:
    suffix = Path(source_name).suffix.lower()
    return suffix if suffix else ""


def extract_archive(zip_path: Path, dataset_root: Path, manifest_dir: Path) -> list[ExtractedItem]:
    items: list[ExtractedItem] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            if not is_interesting_member(member.filename):
                continue

            inferred = infer_kind_and_label(member.filename)
            if inferred is None:
                continue
            kind, label = inferred

            with zf.open(member) as src:
                sha = sha256_stream(src)

            ext = output_extension(member.filename)
            if kind == "person":
                output_dir = dataset_root / "raw" / "person" / label
            else:
                output_dir = dataset_root / "raw" / "videos"
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"{sha}{ext}"
            if not output_path.exists():
                with zf.open(member) as src, output_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)

            items.append(
                ExtractedItem(
                    kind=kind,
                    label=label,
                    archive=zip_path.name,
                    source_path=member.filename,
                    sha256=sha,
                    size_bytes=member.file_size,
                    output_path=str(output_path.relative_to(dataset_root)),
                )
            )

    return items


def write_manifest(dataset_root: Path, items: list[ExtractedItem]) -> None:
    manifest_dir = dataset_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    csv_path = manifest_dir / "extraction_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "kind",
                "label",
                "archive",
                "source_path",
                "sha256",
                "size_bytes",
                "output_path",
            ]
        )
        for item in items:
            writer.writerow(
                [
                    item.kind,
                    item.label,
                    item.archive,
                    item.source_path,
                    item.sha256,
                    item.size_bytes,
                    item.output_path,
                ]
            )

    summary = {
        "person_files": sum(1 for item in items if item.kind == "person"),
        "video_files": sum(1 for item in items if item.kind == "video"),
        "unique_person_files": len({item.sha256 for item in items if item.kind == "person"}),
        "unique_video_files": len({item.sha256 for item in items if item.kind == "video"}),
        "archives": sorted({item.archive for item in items}),
    }
    with (manifest_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="datasets",
        help="Root folder for extracted assets and manifests.",
    )
    parser.add_argument(
        "--archives",
        nargs="*",
        default=["work 1.zip", "Work 2-20260625T082324Z-3-001.zip", "work3-20260625T082143Z-3-001.zip"],
        help="Zip archives to process.",
    )
    args = parser.parse_args(argv)

    dataset_root = Path(args.dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    all_items: list[ExtractedItem] = []
    for archive_name in args.archives:
        archive_path = Path(archive_name)
        if not archive_path.exists():
            print(f"Skipping missing archive: {archive_path}", file=sys.stderr)
            continue
        all_items.extend(extract_archive(archive_path, dataset_root, dataset_root / "manifests"))

    write_manifest(dataset_root, all_items)

    person_items = [item for item in all_items if item.kind == "person"]
    video_items = [item for item in all_items if item.kind == "video"]
    print(f"Extracted {len(person_items)} person files ({len({i.sha256 for i in person_items})} unique).")
    print(f"Extracted {len(video_items)} video files ({len({i.sha256 for i in video_items})} unique).")
    print(f"Manifest written to {dataset_root / 'manifests' / 'extraction_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
