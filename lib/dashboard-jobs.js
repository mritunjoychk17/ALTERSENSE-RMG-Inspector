import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { createReadStream, openSync } from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { spawn } from "node:child_process";
import { Readable } from "node:stream";

const REPO_ROOT = process.cwd();
const APP_ROOT = path.join(REPO_ROOT, "artifacts", "webapp");
const JOB_ROOT = path.join(APP_ROOT, "jobs");
const UPLOAD_ROOT = path.join(APP_ROOT, "uploads");
const PYTHON_BIN = process.env.PYTHON_BIN || "python3";
const CAM33_PROFILE = "cam_33_28_oct_f1_24";
const GPU_WORKER_BASE_URL = (process.env.GPU_WORKER_BASE_URL || "").trim().replace(/\/+$/, "");
const GPU_WORKER_TOKEN = (process.env.GPU_WORKER_TOKEN || "").trim();

export async function ensureJobDirs() {
  await mkdir(JOB_ROOT, { recursive: true });
  await mkdir(UPLOAD_ROOT, { recursive: true });
}

export function hasRemoteWorker() {
  return Boolean(GPU_WORKER_BASE_URL);
}

export function remoteWorkerBaseUrl() {
  return GPU_WORKER_BASE_URL;
}

function remoteAuthHeaders() {
  return GPU_WORKER_TOKEN ? { Authorization: `Bearer ${GPU_WORKER_TOKEN}` } : {};
}

export function cam33ProfileId() {
  return CAM33_PROFILE;
}

export function jobDir(jobId) {
  return path.join(JOB_ROOT, jobId);
}

export async function createUploadJob({ file, sampleEvery, presentThreshold, device, poseBackend }) {
  if (hasRemoteWorker()) {
    const formData = new FormData();
    formData.append("video", file, file.name || "upload.mp4");
    formData.append("sampleEvery", String(sampleEvery));
    formData.append("presentThreshold", String(presentThreshold));
    formData.append("device", String(device));
    formData.append("poseBackend", String(poseBackend));
    const response = await fetch(`${GPU_WORKER_BASE_URL}/jobs`, {
      method: "POST",
      body: formData,
      cache: "no-store",
      headers: remoteAuthHeaders(),
    });
    if (!response.ok) {
      const payload = await response.text();
      throw new Error(payload || "Remote worker rejected the job.");
    }
    return response.json();
  }

  await ensureJobDirs();
  const jobId = `${new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14)}_${crypto.randomUUID().slice(0, 8)}`;
  const safeName = path.basename(file.name || "upload.mp4");
  const uploadName = `${jobId}_${safeName}`;
  const uploadPath = path.join(UPLOAD_ROOT, uploadName);
  const jobPath = jobDir(jobId);
  await mkdir(jobPath, { recursive: true });
  const bytes = Buffer.from(await file.arrayBuffer());
  await writeFile(uploadPath, bytes);
  await writeStatus(jobId, {
    status: "queued",
    message: "Upload complete. Waiting to start the operator evaluation pipeline...",
    progress: 3,
    updated_at: new Date().toISOString(),
    result_ready: false
  });

  const stdoutPath = path.join(jobPath, "worker.out.log");
  const stderrPath = path.join(jobPath, "worker.err.log");
  const stdoutFd = openSync(stdoutPath, "a");
  const stderrFd = openSync(stderrPath, "a");

  const args = [
    path.join(REPO_ROOT, "scripts", "run_dashboard_operator_job.py"),
    "--job-id", jobId,
    "--job-dir", jobPath,
    "--video", uploadPath,
    "--profile-video-id", CAM33_PROFILE,
    "--sample-every", String(sampleEvery),
    "--present-threshold", String(presentThreshold),
    "--device", device,
    "--pose-backend", poseBackend,
  ];

  const child = spawn(PYTHON_BIN, args, {
    cwd: REPO_ROOT,
    detached: true,
    stdio: ["ignore", stdoutFd, stderrFd],
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });
  child.unref();
  return { jobId };
}

async function fetchRemoteJson(pathname) {
  const response = await fetch(`${GPU_WORKER_BASE_URL}${pathname}`, {
    cache: "no-store",
    headers: remoteAuthHeaders(),
  });
  if (!response.ok) {
    const payload = await response.text();
    throw new Error(payload || `Remote request failed for ${pathname}`);
  }
  return response.json();
}

export async function readWorkerHealth() {
  if (!hasRemoteWorker()) {
    return { ok: true, mode: "local", worker: "embedded-local" };
  }
  try {
    const response = await fetch(`${GPU_WORKER_BASE_URL}/health`, {
      cache: "no-store",
      headers: remoteAuthHeaders(),
    });
    if (!response.ok) {
      throw new Error(`Health check failed with ${response.status}`);
    }
    const payload = await response.json();
    return { ok: true, mode: "remote", ...payload };
  } catch (error) {
    return {
      ok: false,
      mode: "remote",
      error: error.message || "Worker unreachable",
      worker: "unreachable",
    };
  }
}

export async function readUnifiedJobStatus(jobId) {
  if (hasRemoteWorker()) {
    return fetchRemoteJson(`/jobs/${jobId}`);
  }
  const status = await readJobStatus(jobId);
  const resultPath = `${jobDir(jobId)}/result.json`;
  const hasResult = await fileExists(resultPath);
  const result = hasResult ? await readJobResult(jobId) : null;
  return {
    jobId,
    status,
    result,
    urls: {
      report: `/api/jobs/${jobId}/report`,
      overlay: `/api/jobs/${jobId}/artifacts/stage1_overlay.mp4`,
      log: `/api/jobs/${jobId}/artifacts/pipeline.log`,
    },
  };
}

export async function readUnifiedJobReport(jobId) {
  if (hasRemoteWorker()) {
    return fetchRemoteJson(`/jobs/${jobId}/report`);
  }
  const resultPath = `${jobDir(jobId)}/result.json`;
  if (!(await fileExists(resultPath))) {
    throw new Error("Job result is not ready yet.");
  }
  const result = await readJobResult(jobId);
  const reportPath = result.report_json;
  if (!(await fileExists(reportPath))) {
    throw new Error("Report file is missing.");
  }
  return JSON.parse(await readFile(reportPath, "utf-8"));
}

export async function writeStatus(jobId, payload) {
  const statusPath = path.join(jobDir(jobId), "status.json");
  await writeFile(statusPath, JSON.stringify(payload, null, 2) + "\n", "utf-8");
}

export async function readJobJson(jobId, name) {
  const target = path.join(jobDir(jobId), name);
  const raw = await readFile(target, "utf-8");
  return JSON.parse(raw);
}

export async function readJobStatus(jobId) {
  return readJobJson(jobId, "status.json");
}

export async function readJobResult(jobId) {
  return readJobJson(jobId, "result.json");
}

export async function fileExists(filePath) {
  try {
    await stat(filePath);
    return true;
  } catch {
    return false;
  }
}

export async function streamJobFile(jobId, filename) {
  if (hasRemoteWorker()) {
    const safeName = String(filename || "").replace(/^\/+/, "");
    const response = await fetch(`${GPU_WORKER_BASE_URL}/jobs/${jobId}/artifacts/${safeName}`, {
      cache: "no-store",
      headers: remoteAuthHeaders(),
    });
    if (!response.ok || !response.body) {
      throw new Error("Artifact not found.");
    }
    const length = Number(response.headers.get("content-length") || 0);
    const contentType = response.headers.get("content-type") || "application/octet-stream";
    return {
      size: length,
      stream: response.body,
      path: safeName,
      contentType,
    };
  }

  const target = path.resolve(jobDir(jobId), filename);
  const root = path.resolve(jobDir(jobId));
  if (!target.startsWith(root)) {
    throw new Error("Invalid file path.");
  }
  const info = await stat(target);
  return {
    size: info.size,
    stream: Readable.toWeb(createReadStream(target)),
    path: target,
    contentType: "",
  };
}
