#!/usr/bin/env python3
"""Gemini-assisted labeling for Stage 2 activity review queue rows."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import re
import sys
from pathlib import Path
from urllib import request, error

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage2_taxonomy import (
    allowed_labels_for_row,
    allowed_cycle_phases_for_row,
    default_cycle_phase_for_label,
    GEMINI_EXTRA_FIELDS,
    GEMINI_PROTOCOL_VERSION,
    infer_station_role,
    normalize_label_for_row,
    normalize_cycle_phase_for_row,
    phase_matches_label,
    role_description_for_row,
)


API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def resolve_api_key(cli_value: str) -> str:
    if cli_value:
        return cli_value
    return (
        os.environ.get("GOOGLE_API_KEY", "")
        or os.environ.get("GEMINI_API_KEY", "")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-csv", default="datasets/processed/stage2/manifests/activity_review_queue.csv")
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--limit", type=int, default=20, help="Max pending rows to label in one run.")
    parser.add_argument("--station-id", default="", help="Only label rows for this station_id.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Gemini suggestions.")
    parser.add_argument("--include-reviewed", action="store_true", help="Allow Gemini labeling even when review_status is already accepted.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--api-key", default="")
    return parser.parse_args()


def build_prompt(row: dict) -> str:
    allowed = allowed_labels_for_row(row)
    allowed_phases = allowed_cycle_phases_for_row(row)
    station_role = infer_station_role(row)
    station_id = str(row.get("station_id", "")).strip()
    role_desc = role_description_for_row(row)
    label_text = ", ".join(allowed)
    phase_text = ", ".join(allowed_phases)
    has_clip = bool((row.get("clip_paths") or "").strip())
    parts = [
        "You are labeling a garments factory workstation segment from a top-down camera.",
        "You are given a short clip window as an ordered sequence of frames from the same station."
        if has_clip
        else "You are given previous, current, and next crops from the same station.",
        f"The station role is: {station_role}. {role_desc}",
        f"Classify the worker activity into exactly one label from this set only: {label_text}.",
        f"Also classify the cycle phase into exactly one phase from this set only: {phase_text}.",
        "Use the ordered clip frames together to understand action evolution over time and decide the label for the center frame of the clip."
        if has_clip
        else "Use all three frames together to understand motion direction and whether the current frame belongs to a handoff, alignment, sewing, inspection, or idle moment.",
        "Focus primarily on hand position, arm reach direction, machine interaction, and fabric interaction in the work area.",
        "Use these rules: idle means present but not doing productive hand work; get means retrieving fabric/material toward the worker; put means placing or setting fabric/material on the table or machine area; align means arranging or fixing fabric shape/position without clear sewing; pass means handing off or moving prepared fabric onward to the next worker/zone; inspect means checking, verifying, or briefly examining fabric/work quality; adjust_machine means changing machine settings or manipulating machine hardware; sew means active machine-side sewing work with sustained hand interaction near the needle/feed area.",
        "Do not output sew for a preparation/pass station unless the machine-side sewing action is clearly visible and sew is in the allowed list.",
        "Prefer align or pass over sew when the worker is mainly fixing, straightening, or forwarding fabric.",
        "Return strict JSON only with these exact keys: protocol_version, label, confidence, cycle_phase, motion_direction, machine_engaged, hands_on_material, transition_ok, reasoning_short.",
        f"Set protocol_version to {GEMINI_PROTOCOL_VERSION}.",
        "confidence must be a number from 0.0 to 1.0.",
        "motion_direction must be one of: toward_worker, away_from_worker, machine_side, stationary, unclear.",
        "machine_engaged, hands_on_material, and transition_ok must be booleans.",
        "reasoning_short must be one short sentence only.",
    ]

    if station_id == "6":
        parts.extend(
            [
                "Station 6 special guidance: this is a sewing-focused station where the worker often alternates between fabric alignment and active sewing near the machine.",
                "For station 6, treat align followed by clear machine-side hand work as a valid sewing progression even if a separate pickup or get step is not visible.",
                "For station 6, the most meaningful cycle pattern is usually align -> sew or align -> put -> sew.",
                "Do not force get or pickup when the top view does not clearly show retrieval; if the worker goes from arranging fabric into sustained machine-side operation, prefer sew_phase over pickup_phase.",
                "When fabric is already under the machine area and the hands remain near the needle/feed zone across the clip, prefer label=sew and cycle_phase=sew_phase.",
                "Only use pickup_phase for station 6 when the clip clearly shows material being brought back toward the worker away from the machine area.",
            ]
        )
    return " ".join(parts)


def inline_image_part(image_path: Path, title: str) -> dict:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return {
        "inline_data": {
            "mime_type": mime_type,
            "data": image_b64,
        }
    }


def parse_gemini_json(output_text: str) -> dict:
    text = output_text.strip()
    candidates = [text]

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(chunk.strip() for chunk in fenced if chunk.strip())

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1].strip())

    if start != -1:
        depth = 0
        in_string = False
        escaped = False
        balanced_end = None
        for idx, ch in enumerate(text[start:], start=start):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    balanced_end = idx
                    break
        if balanced_end is not None:
            candidates.append(text[start : balanced_end + 1].strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            normalized = re.sub(r",\s*([}\]])", r"\1", candidate)
            normalized = normalized.replace("\r\n", "\n")
            normalized = re.sub(r'^\s*"\."\s*$\n?', "", normalized, flags=re.MULTILINE)
            normalized = re.sub(r'^\s*\."\s*$\n?', "", normalized, flags=re.MULTILINE)
            normalized = re.sub(r'"\s*\.\s*"', '", "', normalized)
            normalized = re.sub(r'\n\s*"\.\s*"', "", normalized)
            normalized = re.sub(r'\.\s*"\s*([}\]])', r'"\1', normalized)
            try:
                return json.loads(normalized)
            except json.JSONDecodeError:
                # Salvage common Gemini drift where label/confidence/reason exist
                # but extra free text is injected into the JSON body.
                label_match = re.search(r'"label"\s*:\s*"([^"]+)"', normalized, flags=re.IGNORECASE)
                confidence_match = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', normalized, flags=re.IGNORECASE)
                reason_match = re.search(
                    r'"reason"\s*:\s*"(.+?)"\s*(?:,?\s*[}\n])',
                    normalized,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if label_match and confidence_match and reason_match:
                    reason = reason_match.group(1)
                    reason = reason.replace('\\"', '"').replace("\\n", " ")
                    reason = re.sub(r"\s+", " ", reason).strip()
                    return {
                        "label": label_match.group(1).strip(),
                        "confidence": float(confidence_match.group(1)),
                        "reasoning_short": reason,
                    }
                continue

    raise RuntimeError(f"Could not parse Gemini JSON response: {output_text[:400]}")


def write_rows(queue_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    for extra in GEMINI_EXTRA_FIELDS:
        if extra not in fieldnames:
            fieldnames.append(extra)
    with queue_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def normalize_gemini_result(result: dict, row: dict) -> dict:
    if isinstance(result, list):
        result = next((item for item in result if isinstance(item, dict)), {}) if result else {}
    if not isinstance(result, dict):
        result = {}
    label = normalize_label_for_row(str(result.get("label", "")).strip(), row)
    cycle_phase = normalize_cycle_phase_for_row(str(result.get("cycle_phase", "")).strip(), row, label=label)
    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    machine_engaged = parse_bool(result.get("machine_engaged"), default=(label in {"sew", "adjust_machine"}))
    hands_on_material = parse_bool(result.get("hands_on_material"), default=(label not in {"idle", "uncertain"}))
    motion_direction = str(result.get("motion_direction", "")).strip().lower().replace("-", "_").replace(" ", "_")
    if motion_direction not in {"toward_worker", "away_from_worker", "machine_side", "stationary", "unclear"}:
        motion_direction = "unclear"
    transition_ok = parse_bool(result.get("transition_ok"), default=True)
    reasoning_short = str(
        result.get("reasoning_short")
        or result.get("reason")
        or ""
    ).strip()
    schema_error = ""
    safe_label = label

    if not phase_matches_label(label, cycle_phase, row):
        schema_error = "phase_label_mismatch"
        safe_label = "uncertain"
    elif label in {"sew", "adjust_machine"} and not machine_engaged:
        schema_error = "machine_not_engaged_for_sew"
        safe_label = "align"
        cycle_phase = default_cycle_phase_for_label(safe_label, row)
    elif label in {"put", "pass", "get"} and not hands_on_material:
        schema_error = "material_contact_missing"
        safe_label = "uncertain"
    elif not transition_ok and confidence < 0.85:
        schema_error = "transition_not_confident"
        safe_label = "uncertain"

    return {
        "protocol_version": str(result.get("protocol_version") or GEMINI_PROTOCOL_VERSION),
        "label": label,
        "confidence": confidence,
        "cycle_phase": cycle_phase,
        "motion_direction": motion_direction,
        "machine_engaged": machine_engaged,
        "hands_on_material": hands_on_material,
        "transition_ok": transition_ok,
        "reasoning_short": reasoning_short,
        "safe_label": safe_label,
        "schema_error": schema_error,
        "raw_json": json.dumps(result, ensure_ascii=True),
    }


def call_gemini(api_key: str, model: str, row: dict, image_path: Path) -> dict:
    parts = [{"text": build_prompt(row)}]
    clip_paths = [part for part in (row.get("clip_paths") or "").split("|") if part]
    if clip_paths:
        center_idx = len(clip_paths) // 2
        for idx, clip_path in enumerate(clip_paths):
            rel = "Center frame" if idx == center_idx else f"Clip frame {idx + 1}"
            parts.append({"text": rel})
            parts.append(inline_image_part(Path(clip_path), rel))
    else:
        if row.get("prev_crop_path"):
            parts.append({"text": "Previous frame"})
            parts.append(inline_image_part(Path(row["prev_crop_path"]), "previous"))
        parts.append({"text": "Current frame"})
        parts.append(inline_image_part(image_path, "current"))
        if row.get("next_crop_path"):
            parts.append({"text": "Next frame"})
            parts.append(inline_image_part(Path(row["next_crop_path"]), "next"))
    payload = {
        "contents": [
            {
                "parts": parts
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }
    req = request.Request(
        API_URL_TEMPLATE.format(model=model, api_key=api_key),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini HTTP error {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc

    output_text = ""
    for candidate in data.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            if "text" in part:
                output_text += part["text"]
    if not output_text.strip():
        raise RuntimeError(f"Gemini returned no output_text for {image_path}")
    return normalize_gemini_result(parse_gemini_json(output_text), row)


def main() -> int:
    args = parse_args()
    load_env_file(Path(args.env_file))
    api_key = resolve_api_key(args.api_key)
    if not api_key:
        raise ValueError("No Gemini API key provided. Set GOOGLE_API_KEY or GEMINI_API_KEY, use an .env file, or pass --api-key.")

    queue_path = Path(args.queue_csv)
    rows = list(csv.DictReader(queue_path.open(newline="", encoding="utf-8")))
    updated = 0
    for row in rows:
        if updated >= args.limit:
            break
        if args.station_id and str(row.get("station_id", "")).strip() != str(args.station_id).strip():
            continue
        has_protocol = bool((row.get("gemini_protocol_version") or "").strip())
        if not args.overwrite and row.get("gemini_label") and has_protocol:
            continue
        if not args.include_reviewed and row.get("review_status", "") in {"done", "reviewed", "approved"}:
            continue
        image_path = Path(row["crop_path"])
        result = call_gemini(api_key, args.model, row, image_path)
        row["gemini_label"] = str(result.get("label", "")).strip()
        row["gemini_confidence"] = str(result.get("confidence", ""))
        row["gemini_reason"] = str(result.get("reasoning_short", "")).strip()
        row["gemini_protocol_version"] = str(result.get("protocol_version", GEMINI_PROTOCOL_VERSION))
        row["gemini_cycle_phase"] = str(result.get("cycle_phase", ""))
        row["gemini_motion_direction"] = str(result.get("motion_direction", ""))
        row["gemini_machine_engaged"] = str(result.get("machine_engaged", ""))
        row["gemini_hands_on_material"] = str(result.get("hands_on_material", ""))
        row["gemini_transition_ok"] = str(result.get("transition_ok", ""))
        row["gemini_safe_label"] = str(result.get("safe_label", ""))
        row["gemini_schema_error"] = str(result.get("schema_error", ""))
        row["gemini_json"] = str(result.get("raw_json", ""))
        updated += 1
        write_rows(queue_path, rows)
        print(
            f"Labeled {image_path.name} -> {row['gemini_label']} "
            f"safe={row['gemini_safe_label']} phase={row['gemini_cycle_phase']} "
            f"({row['gemini_confidence']})"
        )

    write_rows(queue_path, rows)

    print(f"Updated {updated} rows in {queue_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
