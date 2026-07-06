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

export async function ensureJobDirs() {
  await mkdir(JOB_ROOT, { recursive: true });
  await mkdir(UPLOAD_ROOT, { recursive: true });
}

export function cam33ProfileId() {
  return CAM33_PROFILE;
}

export function jobDir(jobId) {
  return path.join(JOB_ROOT, jobId);
}

export async function createUploadJob({ file, sampleEvery, presentThreshold, device, poseBackend }) {
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
  };
}
