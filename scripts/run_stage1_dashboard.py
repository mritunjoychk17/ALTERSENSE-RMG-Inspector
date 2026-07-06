#!/usr/bin/env python3
"""Local-only web dashboard for Stage 1 video upload and inference."""

from __future__ import annotations

import csv
from email.parser import BytesParser
from email.policy import default
import html
import json
import mimetypes
import os
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from urllib.parse import parse_qs, quote, unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parent.parent
APP_ROOT = REPO_ROOT / "artifacts" / "webapp"
UPLOAD_ROOT = APP_ROOT / "uploads"
JOB_ROOT = APP_ROOT / "jobs"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "roi_annotations.template.json"
DEFAULT_CHECKPOINT = REPO_ROOT / "artifacts" / "stage1" / "models" / "mobilenet_station_domain_clean" / "best.pt"


def load_profiles(config_path: Path) -> list[dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    profiles = []
    for video in config["videos"]:
        annotated = [ws for ws in video["workstations"] if ws.get("station_roi_polygon")]
        if not annotated:
            continue
        profiles.append(
            {
                "video_id": video["video_id"],
                "source_name": video.get("source_name", video["video_id"]),
                "station_count": len(annotated),
                "video": video,
            }
        )
    return profiles


def html_page(title: str, body: str) -> bytes:
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #edf5ff;
      --paper: #ffffff;
      --ink: #12304a;
      --muted: #5f7891;
      --accent: #1e63d6;
      --accent-2: #0d3f8a;
      --line: #cfe0f5;
      --soft: #f5faff;
      --good: #dff5ea;
      --good-ink: #17724b;
      --warn: #fff4dd;
      --shadow: 0 14px 34px rgba(17, 55, 102, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(30,99,214,0.12), transparent 30%),
        radial-gradient(circle at top right, rgba(81,160,255,0.16), transparent 28%),
        linear-gradient(180deg, #f8fbff, var(--bg));
    }}
    .wrap {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px 20px 48px;
    }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 12px 2px 20px;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}
    .brandmark {{
      width: 48px;
      height: 48px;
      border-radius: 14px;
      background: linear-gradient(135deg, #0d4ca3, #4f97ff);
      color: white;
      display: grid;
      place-items: center;
      font-weight: 700;
      letter-spacing: 0.06em;
      box-shadow: var(--shadow);
    }}
    .brandtext strong {{
      display: block;
      font-size: 1.1rem;
      letter-spacing: 0.01em;
    }}
    .brandtext span {{
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .topmeta {{
      color: var(--muted);
      font-size: 0.92rem;
      text-align: right;
    }}
    .hero {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 0;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .herohead {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 0;
      min-height: 320px;
    }}
    .herotext {{
      padding: 34px 30px;
      background:
        linear-gradient(140deg, rgba(18,97,214,0.06), rgba(255,255,255,0.78)),
        var(--paper);
    }}
    .heroart {{
      position: relative;
      background:
        radial-gradient(circle at 30% 25%, rgba(102,170,255,0.50), transparent 24%),
        radial-gradient(circle at 70% 30%, rgba(30,99,214,0.20), transparent 20%),
        linear-gradient(145deg, #0d3f8a, #1f67da 62%, #8cc3ff);
      overflow: hidden;
    }}
    .heroart::before,
    .heroart::after {{
      content: "";
      position: absolute;
      border-radius: 28px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(255,255,255,0.08);
      backdrop-filter: blur(2px);
    }}
    .heroart::before {{
      width: 210px;
      height: 120px;
      right: 28px;
      top: 36px;
      transform: rotate(-8deg);
    }}
    .heroart::after {{
      width: 170px;
      height: 170px;
      left: 34px;
      bottom: 34px;
      transform: rotate(12deg);
    }}
    .herochips {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 18px 0 22px;
    }}
    .herochip {{
      padding: 8px 12px;
      border-radius: 999px;
      background: #e9f3ff;
      color: var(--accent-2);
      font-size: 0.92rem;
      font-weight: 700;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 16px;
    }}
    .stat {{
      background: white;
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
    }}
    .stat strong {{
      display: block;
      font-size: 1.55rem;
      color: var(--accent-2);
      margin-bottom: 6px;
    }}
    h1, h2, h3 {{ margin-top: 0; }}
    h1 {{ font-size: 2.2rem; letter-spacing: -0.03em; }}
    p, label, th, td, input, select, button {{ font-size: 1rem; }}
    .muted {{ color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-top: 22px;
    }}
    .card {{
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
    }}
    .stack {{
      display: grid;
      gap: 18px;
    }}
    form {{
      display: grid;
      gap: 16px;
      margin-top: 0;
    }}
    .panel {{
      padding: 28px 30px 30px;
      border-top: 1px solid var(--line);
      background: #fbfdff;
    }}
    .panelhead {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: end;
      margin-bottom: 18px;
    }}
    label {{
      display: grid;
      gap: 8px;
      font-weight: 700;
    }}
    .helper {{
      font-size: 0.92rem;
      color: var(--muted);
      font-weight: 400;
    }}
    input, select, button {{
      width: 100%;
      padding: 12px 14px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
    }}
    button {{
      background: var(--accent);
      color: white;
      border: none;
      cursor: pointer;
      font-weight: 700;
    }}
    button:hover {{ filter: brightness(1.03); }}
    .secondary {{
      background: white;
      color: var(--accent-2);
      border: 1px solid var(--line);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 18px;
      background: white;
      border-radius: 16px;
      overflow: hidden;
    }}
    th, td {{
      padding: 12px 10px;
      border-bottom: 1px solid #eee5d7;
      text-align: left;
    }}
    th {{
      background: #eff6ff;
    }}
    .actions {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 18px;
    }}
    .actions a {{
      text-decoration: none;
      flex: 1 1 220px;
    }}
    .pill {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: #e8f1ff;
      color: var(--accent-2);
      font-weight: 700;
      font-size: 0.92rem;
    }}
    .pill.good {{
      background: var(--good);
      color: var(--good-ink);
    }}
    .pill.warn {{
      background: var(--warn);
      color: #8a620d;
    }}
    video {{
      width: 100%;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: #000;
      margin-top: 18px;
    }}
    .error {{
      border-left: 4px solid #c23616;
      background: #fff4f1;
      padding: 14px 16px;
      border-radius: 12px;
      white-space: pre-wrap;
    }}
    .statusbox {{
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
    }}
    .progress {{
      width: 100%;
      height: 14px;
      background: #dce9fa;
      border-radius: 999px;
      overflow: hidden;
      margin: 14px 0 10px;
    }}
    .bar {{
      height: 100%;
      background: linear-gradient(90deg, #2d7cf6, #6aaeff);
      border-radius: 999px;
      transition: width 0.4s ease;
    }}
    .mono {{
      font-family: "Courier New", monospace;
      white-space: pre-wrap;
      font-size: 0.92rem;
    }}
    .sectiontitle {{
      margin-top: 28px;
      margin-bottom: 8px;
    }}
    @media (max-width: 860px) {{
      .herohead {{
        grid-template-columns: 1fr;
      }}
      .topbar {{
        flex-direction: column;
        align-items: flex-start;
      }}
      .topmeta {{
        text-align: left;
      }}
      .stats {{
        grid-template-columns: 1fr;
      }}
      .panelhead {{
        flex-direction: column;
        align-items: flex-start;
      }}
    }}
    code {{ background: #f3eadc; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <div class="wrap">{body}</div>
</body>
</html>"""
    return doc.encode("utf-8")


def parse_multipart_form(handler: BaseHTTPRequestHandler) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    content_type = handler.headers.get("Content-Type", "")
    content_length = int(handler.headers.get("Content-Length", "0"))
    if "multipart/form-data" not in content_type:
        raise ValueError("Expected multipart/form-data upload.")
    boundary_token = "boundary="
    if boundary_token not in content_type:
        raise ValueError("Missing multipart boundary.")
    boundary = content_type.split(boundary_token, 1)[1].strip().strip('"')
    body = handler.rfile.read(content_length)
    mime = (
        f"Content-Type: multipart/form-data; boundary={boundary}\r\n"
        f"MIME-Version: 1.0\r\n\r\n"
    ).encode("utf-8") + body
    message = BytesParser(policy=default).parsebytes(mime)

    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            files[name] = (filename, payload)
        else:
            fields[name] = payload.decode("utf-8", errors="replace")
    return fields, files


def summarize_station_csv(csv_path: Path) -> dict:
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    if not rows:
        return {"frames": 0, "present_frames": 0, "present_ratio": 0.0, "avg_present_conf": 0.0, "label": "unknown"}
    present_scores = [float(r.get("present_confidence") or 0.0) for r in rows]
    present_frames = sum(1 for r in rows if r.get("predicted_label") == "present")
    label = "present" if present_frames >= (len(rows) / 2) else "absent"
    return {
        "frames": len(rows),
        "present_frames": present_frames,
        "present_ratio": round(present_frames / len(rows), 3),
        "avg_present_conf": round(sum(present_scores) / len(present_scores), 3),
        "max_present_conf": round(max(present_scores), 3),
        "label": label,
    }


def write_job_status(job_dir: Path, status: str, message: str, progress: int, extra: dict | None = None) -> None:
    payload = {
        "status": status,
        "message": message,
        "progress": progress,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    (job_dir / "status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_job(
    job_id: str,
    job_dir: Path,
    upload_path: Path,
    profile_video_id: str,
    sample_every: int,
    present_threshold: float,
    device: str,
    checkpoint_path: Path,
) -> None:
    profiles = load_profiles(DEFAULT_CONFIG)
    profile = next((item for item in profiles if item["video_id"] == profile_video_id), None)
    if profile is None:
        raise ValueError(f"Unknown ROI profile: {profile_video_id}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    runtime_video_id = f"upload_{job_id}"
    runtime_video = dict(profile["video"])
    runtime_video["video_id"] = runtime_video_id
    runtime_video["video_path"] = str(upload_path)
    runtime_video["source_name"] = upload_path.name
    runtime_config = {"version": 2, "description": "Temporary upload config", "videos": [runtime_video]}
    config_path = job_dir / "runtime_config.json"
    config_path.write_text(json.dumps(runtime_config, indent=2) + "\n", encoding="utf-8")
    write_job_status(job_dir, "running", "Preparing temporary runtime configuration...", 10)

    overlay_path = job_dir / "overlay.mp4"
    overlay_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "render_stage1_overlay_video.py"),
        "--config",
        str(config_path),
        "--video-id",
        runtime_video_id,
        "--checkpoint",
        str(checkpoint_path),
        "--sample-every",
        str(sample_every),
        "--present-threshold",
        str(present_threshold),
        "--device",
        device,
        "--output",
        str(overlay_path),
    ]
    write_job_status(job_dir, "running", "Rendering the Stage 1 overlay video...", 35)
    subprocess.run(overlay_cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)

    station_summaries = []
    annotated_workstations = [ws for ws in runtime_video["workstations"] if ws.get("station_roi_polygon")]
    total_workstations = max(len(annotated_workstations), 1)
    for idx, workstation in enumerate(runtime_video["workstations"], start=1):
        if not workstation.get("station_roi_polygon"):
            continue
        station_id = workstation["station_id"]
        csv_path = job_dir / f"station_{station_id}.csv"
        progress = 40 + int((idx / total_workstations) * 50)
        write_job_status(job_dir, "running", f"Summarizing station {station_id}...", progress)
        predict_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "predict_stage1_mobilenet.py"),
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--video-id",
            runtime_video_id,
            "--station-id",
            station_id,
            "--sample-every",
            str(sample_every),
            "--present-threshold",
            str(present_threshold),
            "--device",
            device,
            "--output",
            str(csv_path),
        ]
        subprocess.run(predict_cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
        summary = summarize_station_csv(csv_path)
        summary["station_id"] = station_id
        station_summaries.append(summary)

    result = {
        "job_id": job_id,
        "upload_path": str(upload_path),
        "overlay_path": str(overlay_path),
        "profile_video_id": profile_video_id,
        "sample_every": sample_every,
        "present_threshold": present_threshold,
        "device": device,
        "checkpoint_path": str(checkpoint_path),
        "station_summaries": station_summaries,
    }
    (job_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_job_status(job_dir, "done", "Processing complete. Your overlay and station summary are ready.", 100, {"result_ready": True})


def run_job_background(job_id: str, job_dir: Path, upload_path: Path, profile_video_id: str, sample_every: int, present_threshold: float, device: str, checkpoint_path: Path) -> None:
    try:
        run_job(job_id, job_dir, upload_path, profile_video_id, sample_every, present_threshold, device, checkpoint_path)
    except Exception as exc:
        write_job_status(job_dir, "error", f"Stage 1 run failed:\n{exc}", 100)


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "RMGStage1Dashboard/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.handle_index()
            return
        if parsed.path == "/result":
            params = parse_qs(parsed.query)
            self.handle_result(params.get("job", [""])[0])
            return
        if parsed.path == "/status":
            params = parse_qs(parsed.query)
            self.handle_status(params.get("job", [""])[0])
            return
        if parsed.path.startswith("/files/"):
            params = parse_qs(parsed.query)
            self.handle_file(parsed.path[len("/files/"):], download=params.get("download", ["0"])[0] == "1")
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if self.path == "/upload":
            self.handle_upload()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def handle_index(self, error: str = "") -> None:
        profiles = load_profiles(DEFAULT_CONFIG)
        options = "\n".join(
            f'<option value="{html.escape(item["video_id"])}">{html.escape(item["video_id"])} '
            f'({item["station_count"]} stations)</option>'
            for item in profiles
        )
        error_html = f'<div class="error">{html.escape(error)}</div>' if error else ""
        body = f"""
        <div class="topbar">
          <div class="brand">
            <div class="brandmark">RMG</div>
            <div class="brandtext">
              <strong>RMG Vision Operations</strong>
              <span>Factory Floor Monitoring Portal</span>
            </div>
          </div>
          <div class="topmeta">
            Local deployment
            <br>
            Secure internal access
          </div>
        </div>
        <section class="hero">
          <div class="herohead">
            <div class="herotext">
              <div class="pill">Stage 1 Presence Monitoring</div>
              <h1>Factory floor visibility for workstation presence</h1>
              <p class="muted">Use this portal to upload a production-floor video and receive a station-by-station presence overview with an annotated playback result.</p>
              <div class="herochips">
                <span class="herochip">Worker presence overview</span>
                <span class="herochip">Station-based analysis</span>
                <span class="herochip">Private internal deployment</span>
              </div>
              <div class="stats">
                <div class="stat">
                  <strong>{len(profiles)}</strong>
                  <span class="muted">Saved floor layouts ready to use</span>
                </div>
                <div class="stat">
                  <strong>1</strong>
                  <span class="muted">Focused task: presence monitoring only</span>
                </div>
                <div class="stat">
                  <strong>Local</strong>
                  <span class="muted">Runs fully on your own device</span>
                </div>
              </div>
            </div>
            <div class="heroart"></div>
          </div>
          <div class="panel">
          <div class="panelhead">
            <div>
              <h2>Start a new analysis</h2>
              <p class="muted">Choose a video and the matching saved layout, then begin processing.</p>
            </div>
          </div>
          {error_html}
          <form method="post" action="/upload" enctype="multipart/form-data">
            <label>Video file
              <input type="file" name="video" accept=".mp4,.avi,.mov,.mkv" required>
              <span class="helper">Choose the factory video you want to analyze.</span>
            </label>
            <label>ROI profile
              <select name="profile_video_id" required>{options}</select>
              <span class="helper">Pick the saved camera layout that matches your uploaded video.</span>
            </label>
            <label>Sample every N frames
              <input type="number" min="1" name="sample_every" value="20">
              <span class="helper">Smaller numbers are slower but more detailed.</span>
            </label>
            <label>Present threshold
              <input type="number" min="0" max="1" step="0.01" name="present_threshold" value="0.50">
              <span class="helper">Higher threshold is stricter. Lower threshold is more sensitive.</span>
            </label>
            <label>Device
              <select name="device">
                <option value="cuda">cuda</option>
                <option value="cpu">cpu</option>
              </select>
              <span class="helper">Use <code>cuda</code> if your GPU environment is ready.</span>
            </label>
            <button type="submit">Start Presence Analysis</button>
          </form>
          </div>
        </section>
        """
        self.respond_html("RMG Stage 1 Dashboard", body)

    def handle_upload(self) -> None:
        try:
            fields, files = parse_multipart_form(self)
        except Exception as exc:
            self.handle_index(f"Could not parse upload form:\n{exc}")
            return

        if "video" not in files or not files["video"][0]:
            self.handle_index("Please choose a video file.")
            return

        video_filename, video_bytes = files["video"]
        profile_video_id = fields.get("profile_video_id", "").strip()
        checkpoint_path = Path(fields.get("checkpoint_path", str(DEFAULT_CHECKPOINT)).strip())
        sample_every = int(fields.get("sample_every", "20"))
        present_threshold = float(fields.get("present_threshold", "0.5"))
        device = fields.get("device", "cuda").strip() or "cuda"

        UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        safe_name = os.path.basename(video_filename)
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stamped_name = f"{job_id}_{safe_name}"
        upload_path = UPLOAD_ROOT / stamped_name
        with upload_path.open("wb") as f:
            f.write(video_bytes)

        job_dir = JOB_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        write_job_status(job_dir, "queued", "Upload complete. Waiting to start processing...", 5)
        thread = threading.Thread(
            target=run_job_background,
            args=(job_id, job_dir, upload_path, profile_video_id, sample_every, present_threshold, device, checkpoint_path),
            daemon=True,
        )
        thread.start()

        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", f"/status?job={quote(job_id)}")
        self.end_headers()

    def handle_status(self, job_id: str) -> None:
        job_dir = JOB_ROOT / job_id
        status_path = job_dir / "status.json"
        if not status_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown job id")
            return
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status["status"] == "done":
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", f"/result?job={quote(job_id)}")
            self.end_headers()
            return
        refresh = "<meta http-equiv='refresh' content='3'>" if status["status"] in {"queued", "running"} else ""
        body = f"""
        <section class="hero">
          {refresh}
          <h1>Processing Your Video</h1>
          <p class="muted">Please keep this page open. The dashboard will refresh automatically while the Stage 1 pipeline is running.</p>
          <div class="statusbox">
            <h3>Current Step</h3>
            <div class="pill {'warn' if status['status'] != 'error' else ''}">{html.escape(status['status'].upper())}</div>
            <p>{html.escape(status['message'])}</p>
            <div class="progress"><div class="bar" style="width:{max(2, min(int(status['progress']), 100))}%"></div></div>
            <p class="muted">Progress: {int(status['progress'])}%</p>
            <p class="muted">Last update: {html.escape(status.get('updated_at', ''))}</p>
          </div>
          <div class="actions">
            <a href="/status?job={quote(job_id)}"><button type="button">Refresh Now</button></a>
            <a href="/"><button type="button" class="secondary">Back To Upload Page</button></a>
          </div>
        </section>
        """
        self.respond_html("Processing Video", body)

    def handle_result(self, job_id: str) -> None:
        job_dir = JOB_ROOT / job_id
        result_path = job_dir / "result.json"
        if not result_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown job id")
            return
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rows = []
        for item in result["station_summaries"]:
            pill_class = "good" if item["label"] == "present" else "warn"
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(item['station_id']))}</td>"
                f"<td><span class='pill {pill_class}'>{html.escape(item['label'])}</span></td>"
                f"<td>{item['frames']}</td>"
                f"<td>{item['present_frames']}</td>"
                f"<td>{item['present_ratio']:.3f}</td>"
                f"<td>{item['avg_present_conf']:.3f}</td>"
                f"<td>{item['max_present_conf']:.3f}</td>"
                "</tr>"
            )
        table_rows = "\n".join(rows)
        overlay_rel = str(Path("jobs") / job_id / "overlay.mp4").replace(os.sep, "/")
        body = f"""
        <section class="hero">
          <h1>Presence Analysis Result</h1>
          <p class="muted">ROI profile: <code>{html.escape(result['profile_video_id'])}</code> | sample_every=<code>{result['sample_every']}</code> | threshold=<code>{result['present_threshold']}</code></p>
          <div class="actions">
            <a href="/"><button type="button" class="secondary">Analyze Another Video</button></a>
            <a href="/files/{overlay_rel}?download=1"><button type="button">Download Overlay Video</button></a>
          </div>
          <video controls src="/files/{overlay_rel}"></video>
          <h2 class="sectiontitle">Per-Station Summary</h2>
          <table>
            <thead>
              <tr>
                <th>Station</th>
                <th>Majority Label</th>
                <th>Sampled Frames</th>
                <th>Present Frames</th>
                <th>Present Ratio</th>
                <th>Avg Present Confidence</th>
                <th>Max Present Confidence</th>
              </tr>
            </thead>
            <tbody>{table_rows}</tbody>
          </table>
        </section>
        """
        self.respond_html("Stage 1 Result", body)

    def handle_file(self, rel_path: str, download: bool = False) -> None:
        rel_path = unquote(rel_path)
        target = (APP_ROOT / rel_path).resolve()
        if not str(target).startswith(str(APP_ROOT.resolve())) or not target.exists() or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        mime, _ = mimetypes.guess_type(str(target))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(target.stat().st_size))
        self.send_header("Accept-Ranges", "bytes")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{target.name}"')
        else:
            self.send_header("Content-Disposition", f'inline; filename="{target.name}"')
        self.end_headers()
        with target.open("rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def respond_html(self, title: str, body: str) -> None:
        payload = html_page(title, body)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> int:
    APP_ROOT.mkdir(parents=True, exist_ok=True)
    host = os.environ.get("RMG_DASHBOARD_HOST", "0.0.0.0")
    port = int(os.environ.get("RMG_DASHBOARD_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Stage 1 dashboard running at http://{host}:{port}")
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
