from __future__ import annotations

import json
import mimetypes
import os
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "artifacts" / "webapp"
JOB_ROOT = APP_ROOT / "jobs"
UPLOAD_ROOT = APP_ROOT / "uploads"
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python3")
CAM33_PROFILE = os.environ.get("PROFILE_VIDEO_ID", "cam_33_28_oct_f1_24")
ALLOWED_ORIGIN = os.environ.get("DASHBOARD_ORIGIN", "*")
WORKER_TOKEN = os.environ.get("GPU_WORKER_TOKEN", "").strip()

app = FastAPI(title="ALTERSENSE Local GPU Worker", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ensure_dirs() -> None:
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


def job_dir(job_id: str) -> Path:
    return JOB_ROOT / job_id


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_status(job_id: str, payload: dict) -> None:
    write_json(job_dir(job_id) / "status.json", payload)


def safe_artifact_path(job_id: str, artifact_path: str) -> Path:
    base = job_dir(job_id).resolve()
    target = (base / artifact_path).resolve()
    if not str(target).startswith(str(base)):
        raise HTTPException(status_code=400, detail="Invalid artifact path.")
    return target


def local_job_urls(job_id: str) -> dict:
    return {
        "report": f"/jobs/{job_id}/report",
        "overlay": f"/jobs/{job_id}/artifacts/stage1_overlay.mp4",
        "log": f"/jobs/{job_id}/artifacts/pipeline.log",
    }


def create_job_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{stamp}_{secrets.token_hex(4)}"


def content_type_for(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def require_auth(authorization: str | None) -> None:
    if not WORKER_TOKEN:
        return
    expected = f"Bearer {WORKER_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized.")


@app.get("/health")
def health(authorization: str | None = Header(default=None)) -> dict:
    require_auth(authorization)
    return {"ok": True, "worker": "local-gpu"}


@app.post("/jobs")
async def create_job(
    video: UploadFile = File(...),
    sampleEvery: int = Form(20),
    presentThreshold: float = Form(0.5),
    device: str = Form("cuda"),
    poseBackend: str = Form("auto"),
    authorization: str | None = Header(default=None),
) -> dict:
    require_auth(authorization)
    ensure_dirs()
    job_id = create_job_id()
    job_path = job_dir(job_id)
    job_path.mkdir(parents=True, exist_ok=True)

    safe_name = Path(video.filename or "upload.mp4").name
    upload_name = f"{job_id}_{safe_name}"
    upload_path = UPLOAD_ROOT / upload_name
    with upload_path.open("wb") as f:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    write_status(
        job_id,
        {
            "status": "queued",
            "message": "Upload complete. Waiting to start the operator evaluation pipeline...",
            "progress": 3,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "result_ready": False,
        },
    )

    stdout_path = job_path / "worker.out.log"
    stderr_path = job_path / "worker.err.log"
    stdout = stdout_path.open("ab")
    stderr = stderr_path.open("ab")

    args = [
        PYTHON_BIN,
        str(REPO_ROOT / "scripts" / "run_dashboard_operator_job.py"),
        "--job-id",
        job_id,
        "--job-dir",
        str(job_path),
        "--video",
        str(upload_path),
        "--profile-video-id",
        CAM33_PROFILE,
        "--sample-every",
        str(sampleEvery),
        "--present-threshold",
        str(presentThreshold),
        "--device",
        device,
        "--pose-backend",
        poseBackend,
    ]
    subprocess.Popen(
        args,
        cwd=REPO_ROOT,
        stdout=stdout,
        stderr=stderr,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        start_new_session=True,
    )
    return {
        "jobId": job_id,
        "profileVideoId": CAM33_PROFILE,
        "message": "Upload accepted. The operator evaluation job has started.",
        "urls": local_job_urls(job_id),
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str, authorization: str | None = Header(default=None)) -> dict:
    require_auth(authorization)
    status_path = job_dir(job_id) / "status.json"
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="Unknown job id.")
    status = read_json(status_path)
    result_path = job_dir(job_id) / "result.json"
    result = read_json(result_path) if result_path.exists() else None
    return {
        "jobId": job_id,
        "status": status,
        "result": result,
        "urls": local_job_urls(job_id),
    }


@app.get("/jobs/{job_id}/report")
def get_job_report(job_id: str, authorization: str | None = Header(default=None)):
    require_auth(authorization)
    result_path = job_dir(job_id) / "result.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Job result is not ready yet.")
    result = read_json(result_path)
    report_path = Path(result.get("report_json", ""))
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report file is missing.")
    return JSONResponse(read_json(report_path))


@app.get("/jobs/{job_id}/artifacts/{artifact_path:path}")
def get_job_artifact(job_id: str, artifact_path: str, authorization: str | None = Header(default=None)):
    require_auth(authorization)
    target = safe_artifact_path(job_id, artifact_path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return FileResponse(target, media_type=content_type_for(target), filename=target.name)


@app.get("/operator-image")
def get_operator_image(
    videoId: str = Query(..., alias="videoId"),
    stationId: str = Query(..., alias="stationId"),
    authorization: str | None = Header(default=None),
):
    require_auth(authorization)
    candidates = [
        REPO_ROOT / "artifacts" / "stage1" / "visualizations" / "cam33_roi_previews" / videoId / f"station_{stationId}_masked_crop.jpg",
        REPO_ROOT / "artifacts" / "stage1" / "visualizations" / "roi_previews" / videoId / f"station_{stationId}_masked_crop.jpg",
        REPO_ROOT / "datasets" / "processed" / "stage1" / "domain_reference_crops" / "present" / f"{videoId}_station_{stationId}_frame_000000.jpg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return FileResponse(candidate, media_type="image/jpeg", filename=candidate.name)
    raise HTTPException(status_code=404, detail="Operator image not found.")
