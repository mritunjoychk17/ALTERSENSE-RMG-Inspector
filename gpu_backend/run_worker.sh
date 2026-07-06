#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/gpu_backend/worker.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/gpu_backend/worker.env"
  set +a
fi

HOST="${GPU_WORKER_HOST:-0.0.0.0}"
PORT="${GPU_WORKER_PORT:-9000}"

exec "${PYTHON_BIN:-python3}" -m uvicorn gpu_backend.main:app --host "$HOST" --port "$PORT"
