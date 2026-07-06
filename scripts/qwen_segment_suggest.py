#!/usr/bin/env python3
"""Generate segment suggestions for method-study review using Qwen2.5-VL."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path


SYSTEM_PROMPT = (
    "You are assisting a garments factory method-study analyst. "
    "You receive either three top-down workstation crop images from the same station: previous, current, next; or an ordered short clip window from the same station. "
    "Your job is to suggest whether the current center frame likely belongs to the same action as its neighbors or indicates a meaningful action transition. "
    "Use short, industrial method-study wording. "
    "Return strict JSON with keys: segment_decision, action_label, segment_text, confidence, reason. "
    "segment_decision must be one of: continue, transition, uncertain. "
    "action_label must come from this set only: idle, get, put, sew, uncertain. "
    "Use get for reaching, retrieving, or pulling work items into the station. "
    "Use put for placing, aligning, positioning, or dispatching work items on the station. "
    "Use sew only when the worker is clearly operating a sewing machine or feeding fabric through the needle area."
)

UI_LABELS = {"idle", "get", "put", "sew", "uncertain"}
ACTION_ALIASES = {
    "align_fabric": "put",
    "place_on_bed": "put",
    "feed_to_needle": "sew",
    "release_fabric": "put",
    "reach_get_zone": "get",
    "pull_fabric": "get",
    "move_to_dispatch": "put",
    "pick_accessory": "get",
    "attach_accessory": "put",
    "reposition_fabric": "put",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-csv", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--station-id", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-reviewed", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--reference-report-text", default="")
    parser.add_argument("--reference-image-glob", default="")
    parser.add_argument("--reference-image-limit", type=int, default=4)
    return parser.parse_args()


def load_reference_text(path_value: str) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return text[:4000]


def load_reference_images(glob_value: str, limit: int) -> list[str]:
    if not glob_value:
        return []
    paths = sorted(glob.glob(glob_value))
    return paths[:limit]


def build_user_prompt(row: dict, reference_text: str) -> str:
    station_id = str(row.get("station_id", "")).strip()
    has_clip = bool((row.get("clip_paths") or "").strip())
    prompt = (
        "Review these workstation frames and suggest a method-study segment interpretation.\n"
        f"Station: {row.get('station_id', '')}\n"
        f"Anchor timestamp: {row.get('timestamp_sec', '')} sec\n"
        f"Existing note: {row.get('notes', '')}\n"
        f"Input mode: {'ordered clip window' if has_clip else 'previous/current/next frames'}.\n"
        "Decide if the current center frame should continue the same segment, start a transition, or remain uncertain. "
        "Suggest one concise action label and one concise report-style sentence."
    )
    if reference_text:
        prompt += (
            "\n\nReference method-study report style example:\n"
            f"{reference_text}\n\n"
            "Match the style of the reference report, but only describe the current station frames."
        )
    if station_id == "6":
        prompt += (
            "\n\nStation 6 special guidance:\n"
            "This is a sewing-focused station. The most meaningful local action progression is usually align -> sew or align -> put -> sew.\n"
            "If the worker is arranging fabric and then immediately operating near the needle/feed area, prefer sew over get.\n"
            "Only use get when the frames clearly show material being brought back toward the worker from away from the machine.\n"
            "Use put when there is a short placement step before machine engagement, but do not force put if it is visually merged with alignment.\n"
            "If the center frame is part of sustained machine-side operation, label it sew."
        )
    return prompt


def load_qwen(model_name: str):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def call_qwen(model, processor, row: dict, max_new_tokens: int, reference_text: str, reference_images: list[str]) -> dict:
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    reference_content = []
    for image_path in reference_images:
        reference_content.append({"type": "image", "image": Image.open(image_path).convert("RGB")})
    if reference_text:
        reference_content.insert(0, {"type": "text", "text": "Reference method-study sample from a single-station expert-analyzed task."})

    clip_paths = [part for part in (row.get("clip_paths") or "").split("|") if part]
    if clip_paths:
        center_idx = len(clip_paths) // 2
        clip_content = [{"type": "text", "text": build_user_prompt(row, reference_text)}]
        for idx, clip_path in enumerate(clip_paths):
            label = "Center frame" if idx == center_idx else f"Clip frame {idx + 1}"
            clip_content.append({"type": "text", "text": label})
            clip_content.append({"type": "image", "image": Image.open(clip_path).convert("RGB")})
        user_message = {"role": "user", "content": clip_content}
    else:
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": build_user_prompt(row, reference_text)},
                {"type": "image", "image": Image.open(row["prev_crop_path"]).convert("RGB")} if row.get("prev_crop_path") else {"type": "text", "text": "No previous frame available."},
                {"type": "image", "image": Image.open(row["crop_path"]).convert("RGB")},
                {"type": "image", "image": Image.open(row["next_crop_path"]).convert("RGB")} if row.get("next_crop_path") else {"type": "text", "text": "No next frame available."},
            ],
        }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": reference_content} if reference_content else None,
        user_message,
    ]
    messages = [msg for msg in messages if msg is not None]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed_ids = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return parse_qwen_json(output_text)


def parse_qwen_json(output_text: str) -> dict:
    text = output_text.strip()
    if not text:
        return {
            "segment_decision": "uncertain",
            "action_label": "uncertain",
            "segment_text": "",
            "confidence": "",
            "reason": "Model returned empty output.",
        }

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "segment_decision": "uncertain",
        "action_label": "uncertain",
        "segment_text": text[:300],
        "confidence": "",
        "reason": "Model did not return strict JSON; raw text was captured instead.",
    }


def normalize_action_label(label_value: str) -> str:
    label = (label_value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if label in UI_LABELS:
        return label
    return ACTION_ALIASES.get(label, "uncertain")


def main() -> int:
    args = parse_args()
    queue_path = Path(args.queue_csv)
    with queue_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {queue_path}")

    for row in rows:
        row.setdefault("qwen_segment_decision", "")
        row.setdefault("qwen_action_label", "")
        row.setdefault("qwen_segment_text", "")
        row.setdefault("qwen_confidence", "")
        row.setdefault("qwen_reason", "")

    model, processor = load_qwen(args.model)
    reference_text = load_reference_text(args.reference_report_text)
    reference_images = load_reference_images(args.reference_image_glob, args.reference_image_limit)

    updated = 0
    for row in rows:
        if updated >= args.limit:
            break
        if args.station_id and str(row.get("station_id", "")).strip() != args.station_id:
            continue
        if not args.include_reviewed and row.get("review_status", "") in {"done", "reviewed", "approved"}:
            continue
        if row.get("qwen_segment_decision") and not args.overwrite:
            continue
        result = call_qwen(
            model,
            processor,
            row,
            args.max_new_tokens,
            reference_text,
            reference_images,
        )
        row["qwen_segment_decision"] = str(result.get("segment_decision", "")).strip()
        row["qwen_action_label"] = normalize_action_label(str(result.get("action_label", "")))
        row["qwen_segment_text"] = str(result.get("segment_text", "")).strip()
        row["qwen_confidence"] = str(result.get("confidence", "")).strip()
        row["qwen_reason"] = str(result.get("reason", "")).strip()
        updated += 1
        print(
            f"Suggested {row.get('station_id')} frame {row.get('frame_index')} -> "
            f"{row['qwen_segment_decision']} / {row['qwen_action_label']}"
        )

    with queue_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {updated} rows in {queue_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
