#!/usr/bin/env python3
"""Lightweight local labeling UI for Stage 2 activity review queues."""

from __future__ import annotations

import csv
import html
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import sys
from urllib.parse import parse_qs, quote, unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_QUEUE = REPO_ROOT / "datasets" / "processed" / "stage2" / "manifests" / "activity_review_queue.csv"
from scripts.stage2_taxonomy import (
    accepted_statuses,
    allowed_labels_for_row,
    infer_station_role,
    keyboard_hints_for_row,
    normalize_label_for_row,
    role_description_for_row,
)

ACCEPTED_STATUSES = accepted_statuses()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def clip_frame_triplet(row: dict) -> tuple[str, str, str]:
    crop = (row.get("crop_path") or "").strip()
    prev_crop = (row.get("prev_crop_path") or "").strip()
    next_crop = (row.get("next_crop_path") or "").strip()
    if crop:
        return crop, prev_crop, next_crop

    clip_paths = [item.strip() for item in (row.get("clip_paths") or "").split("|") if item.strip()]
    if not clip_paths:
        return "", "", ""

    center = len(clip_paths) // 2
    crop = clip_paths[center]
    prev_crop = clip_paths[max(0, center - 1)] if center > 0 else ""
    next_crop = clip_paths[min(len(clip_paths) - 1, center + 1)] if center + 1 < len(clip_paths) else ""
    return crop, prev_crop, next_crop


def html_page(title: str, body: str) -> bytes:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f4f8fc;
      --paper: #ffffff;
      --ink: #15314a;
      --muted: #60798f;
      --line: #d5e2ef;
      --accent: #1f69df;
      --soft: #eef5ff;
    }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: linear-gradient(180deg, #f9fbff, var(--bg));
      color: var(--ink);
    }}
    .wrap {{ max-width: 1120px; margin: 0 auto; padding: 24px 18px 40px; }}
    .hero, .panel {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 12px 28px rgba(21,49,74,0.08);
    }}
    .hero {{ padding: 24px; margin-bottom: 18px; }}
    .panel {{ padding: 22px; }}
    .grid {{ display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 18px; }}
    .meta {{ display: grid; gap: 10px; }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .muted {{ color: var(--muted); }}
    .pill {{ display: inline-block; padding: 6px 10px; border-radius: 999px; background: #e8f1ff; color: #12488f; font-weight: 700; }}
    .imagebox {{ background: #f7fbff; border: 1px solid var(--line); border-radius: 18px; padding: 10px; }}
    img {{ width: 100%; border-radius: 14px; display: block; }}
    .triptych {{ display: grid; grid-template-columns: 0.8fr 1fr 0.8fr; gap: 10px; align-items: start; }}
    .framecard {{ display: grid; gap: 8px; }}
    .framecard small {{ color: var(--muted); font-weight: 700; text-align: center; }}
    .dim img {{ opacity: 0.72; }}
    .buttons {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 18px; }}
    button, .linkbtn {{
      padding: 12px 10px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      font-weight: 700;
      cursor: pointer;
      text-decoration: none;
      text-align: center;
    }}
    button.primary {{ background: var(--accent); color: white; border: none; }}
    textarea, input {{
      width: 100%;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 12px;
      font: inherit;
      box-sizing: border-box;
    }}
    .nav {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 18px; }}
    .hint {{ background: var(--soft); border-radius: 14px; padding: 14px; margin-top: 14px; }}
    .suggestion {{
      border-radius: 18px;
      padding: 16px;
      margin: 14px 0 18px;
      border: 1px solid #bfd6ff;
      background: linear-gradient(180deg, #f5f9ff, #e8f1ff);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.9);
    }}
    .suggestion strong {{ color: #114991; }}
    .suggestion .label {{ font-size: 1.1rem; font-weight: 700; }}
    .vlm {{
      border-radius: 18px;
      padding: 16px;
      margin: 14px 0 18px;
      border: 1px solid #c7d8ff;
      background: linear-gradient(180deg, #f8faff, #eef3ff);
    }}
    .vlm strong {{ color: #203e8a; }}
    .vlm .label {{ font-size: 1.08rem; font-weight: 700; }}
    .notice {{
      border-radius: 16px;
      padding: 14px 16px;
      margin: 14px 0 0;
      border: 1px solid #b8e2c4;
      background: linear-gradient(180deg, #f4fff7, #e5f8eb);
      color: #155b2f;
      font-weight: 700;
    }}
    .kbd {{ font-family: monospace; background: #edf1f6; padding: 2px 6px; border-radius: 6px; }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .triptych {{ grid-template-columns: 1fr; }}
      .buttons {{ grid-template-columns: repeat(2, 1fr); }}
    }}
  </style>
</head>
<body>
  <div class="wrap">{body}</div>
  <script>
    document.addEventListener("keydown", (event) => {{
      const keyMap = JSON.parse(document.body.getAttribute("data-keymap") || "{{}}");
      if (keyMap[event.key]) {{
        const input = document.querySelector(`button[data-label="${{keyMap[event.key]}}"]`);
        if (input) input.click();
      }}
    }});
  </script>
</body>
</html>""".encode("utf-8")


class LabelUIHandler(BaseHTTPRequestHandler):
    queue_path = DEFAULT_QUEUE
    qwen_model_name = ""
    qwen_reference_report_text = ""
    qwen_reference_image_glob = ""
    qwen_reference_image_limit = 4
    qwen_max_new_tokens = 128
    _qwen_bundle = None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.handle_index(parse_qs(parsed.query))
            return
        if parsed.path.startswith("/image/"):
            self.handle_image(parsed.path[len("/image/"):])
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if self.path == "/label":
            self.handle_label_post()
            return
        if self.path == "/suggest_qwen":
            self.handle_qwen_post()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def handle_label_post(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        data = parse_qs(self.rfile.read(length).decode("utf-8", errors="replace"))
        row_index = int(data.get("row_index", ["0"])[0])
        label = data.get("label", [""])[0]
        notes = data.get("notes", [""])[0]
        rows = read_rows(self.queue_path)
        if not (0 <= row_index < len(rows)):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid row index")
            return
        rows[row_index]["final_label"] = normalize_label_for_row(label, rows[row_index])
        rows[row_index]["review_status"] = "done"
        rows[row_index]["notes"] = notes
        write_rows(self.queue_path, rows)
        next_index = min(row_index + 1, len(rows) - 1)
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", f"/?row={next_index}&saved=1&saved_label={quote(rows[row_index]['final_label'])}&saved_row={row_index + 1}")
        self.end_headers()

    def get_qwen_bundle(self):
        if self._qwen_bundle is not None:
            return self._qwen_bundle
        if not self.qwen_model_name:
            raise RuntimeError("Start the UI with --qwen-model to enable in-page VLM suggestions.")
        from scripts.qwen_segment_suggest import load_qwen, load_reference_images, load_reference_text

        model, processor = load_qwen(self.qwen_model_name)
        self.__class__._qwen_bundle = (
            model,
            processor,
            load_reference_text(self.qwen_reference_report_text),
            load_reference_images(self.qwen_reference_image_glob, self.qwen_reference_image_limit),
        )
        return self._qwen_bundle

    def handle_qwen_post(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        data = parse_qs(self.rfile.read(length).decode("utf-8", errors="replace"))
        row_index = int(data.get("row_index", ["0"])[0])
        rows = read_rows(self.queue_path)
        if not (0 <= row_index < len(rows)):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid row index")
            return
        try:
            from scripts.qwen_segment_suggest import call_qwen, normalize_action_label

            model, processor, reference_text, reference_images = self.get_qwen_bundle()
            result = call_qwen(
                model,
                processor,
                rows[row_index],
                self.qwen_max_new_tokens,
                reference_text,
                reference_images,
            )
            rows[row_index]["qwen_segment_decision"] = str(result.get("segment_decision", "")).strip()
            rows[row_index]["qwen_action_label"] = normalize_action_label(str(result.get("action_label", "")))
            rows[row_index]["qwen_segment_text"] = str(result.get("segment_text", "")).strip()
            rows[row_index]["qwen_confidence"] = str(result.get("confidence", "")).strip()
            rows[row_index]["qwen_reason"] = str(result.get("reason", "")).strip()
            write_rows(self.queue_path, rows)
            location = f"/?row={row_index}&qwen_saved=1"
        except Exception as exc:
            location = f"/?row={row_index}&qwen_error={quote(str(exc))}"
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.end_headers()

    def handle_index(self, params: dict) -> None:
        rows = read_rows(self.queue_path)
        if not rows:
            self.respond("Stage 2 Labeling UI", "<div class='hero'><h1>No rows found</h1></div>")
            return

        requested = int(params.get("row", ["0"])[0] or 0)
        mode = params.get("mode", ["pending"])[0]
        if mode == "pending":
            pending_indices = [i for i, row in enumerate(rows) if row.get("review_status", "") not in ACCEPTED_STATUSES]
            current_index = pending_indices[0] if pending_indices else 0
            if requested in pending_indices:
                current_index = requested
        else:
            current_index = max(0, min(requested, len(rows) - 1))

        row = rows[current_index]
        allowed_labels = allowed_labels_for_row(row)
        key_hints = keyboard_hints_for_row(row)
        station_role = infer_station_role(row)
        role_desc = role_description_for_row(row)
        pending_count = sum(1 for r in rows if r.get("review_status", "") not in ACCEPTED_STATUSES)
        saved = params.get("saved", ["0"])[0] == "1"
        saved_label = params.get("saved_label", [""])[0]
        saved_row = params.get("saved_row", [""])[0]
        qwen_saved = params.get("qwen_saved", ["0"])[0] == "1"
        qwen_error = params.get("qwen_error", [""])[0]
        crop_path, prev_crop_path, next_crop_path = clip_frame_triplet(row)
        image_rel = quote(crop_path) if crop_path else ""
        prev_rel = quote(prev_crop_path) if prev_crop_path else ""
        next_rel = quote(next_crop_path) if next_crop_path else ""
        buttons = "\n".join(
            f'<button type="submit" name="label" value="{label}" data-label="{label}" class="{"primary" if label in {"align", "pass", "sew"} else ""}">{html.escape(label.replace("_", " ").title())}</button>'
            for label in allowed_labels
        )
        keymap_json = html.escape(json.dumps({key: label for key, label in key_hints}))
        key_hint_html = " ".join(
            f'<span class="kbd">{html.escape(key)}</span> {html.escape(label.replace("_", " "))}'
            for key, label in key_hints
        )
        pose_block = ""
        if row.get("pose_label"):
            pose_block = (
                f"<div class='suggestion'><strong>YOLO Pose Suggestion</strong><br>"
                f"<span class='label'>{html.escape(row['pose_label']).title()}</span>"
                f" <span class='muted'>(confidence {html.escape(row.get('pose_confidence', ''))})</span><br>"
                f"<span class='muted'>{html.escape(row.get('pose_reason', ''))}</span></div>"
            )
        qwen_block = ""
        if row.get("qwen_segment_decision") or row.get("qwen_action_label") or row.get("qwen_segment_text"):
            qwen_block = (
                f"<div class='vlm'>"
                f"<strong>Qwen VLM Segment Suggestion</strong><br>"
                f"<span class='label'>Decision: {html.escape((row.get('qwen_segment_decision') or 'unknown').title())}</span><br>"
                f"<span class='muted'>Suggested UI label: {html.escape((row.get('qwen_action_label') or 'unknown').title())}"
                f"{' | Confidence: ' + html.escape(row.get('qwen_confidence', '')) if row.get('qwen_confidence') else ''}</span><br>"
                f"<span class='muted'>{html.escape(row.get('qwen_segment_text', ''))}</span><br>"
                f"{'<span class=\"muted\">Reason: ' + html.escape(row.get('qwen_reason', '')) + '</span>' if row.get('qwen_reason') else ''}"
                f"</div>"
            )
        gemini_block = ""
        if row.get("gemini_label"):
            gemini_block = (
                f"<div class='hint'><strong>Gemini suggestion:</strong> {html.escape(row['gemini_label'])}"
                f" ({html.escape(row.get('gemini_confidence', ''))})<br>"
                f"<span class='muted'>{html.escape(row.get('gemini_reason', ''))}</span></div>"
            )

        body = f"""
        <section class="hero">
          <h1>Stage 2 Segment Review</h1>
          <p class="muted">Review one crop triplet at a time, inspect the segment context, and label the worker action with VLM assistance when needed.</p>
          {"<div class='notice'>Saved label <strong>" + html.escape(saved_label.title()) + "</strong> for row " + html.escape(saved_row) + ".</div>" if saved and saved_label else ""}
          {"<div class='notice'>Qwen suggestion saved for this row.</div>" if qwen_saved else ""}
          {"<div class='hint'><strong>Qwen error:</strong> " + html.escape(qwen_error) + "</div>" if qwen_error else ""}
          <div class="hint"><strong>Review flow:</strong> use the Qwen VLM segment suggestion and YOLO pose suggestion as starting points, then confirm or correct them from the previous, current, and next frames.</div>
          <div class="nav">
            <span class="pill">Pending rows: {pending_count}</span>
            <span class="pill">Current row: {current_index + 1} / {len(rows)}</span>
            <span class="pill">Station role: {html.escape(station_role.replace('_', ' ').title())}</span>
            <a class="linkbtn" href="/?row={max(0, current_index - 1)}&mode=all">Previous</a>
            <a class="linkbtn" href="/?row={min(len(rows) - 1, current_index + 1)}&mode=all">Next</a>
            <a class="linkbtn" href="/?mode=pending">Jump To Next Pending</a>
          </div>
        </section>
        <section class="panel">
          <div class="grid">
            <div class="imagebox">
              <div class="triptych">
                <div class="framecard dim">
                  <small>Previous frame</small>
                  {"<img src='/image/" + prev_rel + "' alt='previous frame'>" if prev_rel else "<div class='hint'>No earlier sample</div>"}
                </div>
                <div class="framecard">
                  <small>Current frame</small>
                  {f'<img src="/image/{image_rel}" alt="activity crop">' if image_rel else "<div class='hint'>No current frame found</div>"}
                </div>
                <div class="framecard dim">
                  <small>Next frame</small>
                  {"<img src='/image/" + next_rel + "' alt='next frame'>" if next_rel else "<div class='hint'>No later sample</div>"}
                </div>
              </div>
            </div>
            <div class="meta">
              <div class="row">
                <div><strong>Video</strong><br><span class="muted">{html.escape(row.get("video_id", ""))}</span></div>
                <div><strong>Station</strong><br><span class="muted">{html.escape(row.get("station_id", ""))}</span></div>
              </div>
              <div class="row">
                <div><strong>Frame</strong><br><span class="muted">{html.escape(row.get("frame_index", ""))}</span></div>
                <div><strong>Time</strong><br><span class="muted">{html.escape(row.get("timestamp_sec", ""))} sec</span></div>
              </div>
              <div><strong>Role guidance</strong><br><span class="muted">{html.escape(role_desc)}</span></div>
              <div><strong>Presence confidence</strong><br><span class="muted">{html.escape(row.get("presence_confidence", ""))}</span></div>
              <div><strong>Current label</strong><br><span class="muted">{html.escape(row.get("final_label", "") or 'not labeled')}</span></div>
              {qwen_block}
              <form method="post" action="/suggest_qwen">
                <input type="hidden" name="row_index" value="{current_index}">
                <button type="submit" class="linkbtn">Suggest With Qwen</button>
              </form>
              {pose_block}
              {gemini_block}
              <div class="hint">
                <strong>Labeling rule</strong><br>
                Focus on hands, arms, and fabric interaction near the work area. Use the previous and next frames to judge motion direction.<br>
                {key_hint_html}
              </div>
              <form method="post" action="/label">
                <input type="hidden" name="row_index" value="{current_index}">
                <textarea name="notes" rows="4" placeholder="Optional note about hand motion, fabric handling, or ambiguity...">{html.escape(row.get("notes", ""))}</textarea>
                <div class="buttons">{buttons}</div>
              </form>
            </div>
          </div>
        </section>
        """
        self.respond("Stage 2 Labeling UI", body, keymap_json)

    def handle_image(self, rel_path: str) -> None:
        path = Path(unquote(rel_path)).resolve()
        if not str(path).startswith(str(REPO_ROOT.resolve())) or not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
            return
        mime, _ = mimetypes.guess_type(str(path))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        with path.open("rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def respond(self, title: str, body: str, keymap_json: str = "{}") -> None:
        body = f'<div data-keymap="{keymap_json}">{body}</div>'
        payload = html_page(title, body)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-csv", default=str(DEFAULT_QUEUE))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--qwen-model", default="")
    parser.add_argument("--qwen-reference-report-text", default="")
    parser.add_argument("--qwen-reference-image-glob", default="")
    parser.add_argument("--qwen-reference-image-limit", type=int, default=4)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    LabelUIHandler.queue_path = Path(args.queue_csv).resolve()
    LabelUIHandler.qwen_model_name = args.qwen_model
    LabelUIHandler.qwen_reference_report_text = args.qwen_reference_report_text
    LabelUIHandler.qwen_reference_image_glob = args.qwen_reference_image_glob
    LabelUIHandler.qwen_reference_image_limit = args.qwen_reference_image_limit
    LabelUIHandler.qwen_max_new_tokens = args.qwen_max_new_tokens
    server = ThreadingHTTPServer((args.host, args.port), LabelUIHandler)
    print(f"Stage 2 labeling UI running at http://{args.host}:{args.port}")
    print("Use Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
