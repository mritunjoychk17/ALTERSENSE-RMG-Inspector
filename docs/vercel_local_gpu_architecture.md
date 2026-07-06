# Vercel + Local GPU Architecture

## Goal

Host the dashboard UI on Vercel while keeping all heavy video inference on your local or on-prem GPU machine.

## Recommended architecture

1. **Vercel hosts the Next.js dashboard**
2. **Local GPU worker hosts a FastAPI service**
3. The dashboard sends upload and polling requests to its own `/api/jobs` routes
4. Those Next.js routes proxy to the FastAPI GPU worker when `GPU_WORKER_BASE_URL` is set
5. The GPU worker runs the existing Python pipeline locally and exposes:
   - `POST /jobs`
   - `GET /jobs/{job_id}`
   - `GET /jobs/{job_id}/report`
   - `GET /jobs/{job_id}/artifacts/{artifact_path}`
   - `GET /operator-image`

## Why this is the right split

- Vercel is good for:
  - UI hosting
  - lightweight API orchestration
  - report display
- Your local GPU machine is good for:
  - CUDA inference
  - long-running video jobs
  - local filesystem artifacts
  - overlay rendering

## Security model

- Do **not** expose raw local folders directly
- Expose only the FastAPI worker endpoints
- Restrict the worker with:
  - `DASHBOARD_ORIGIN`
  - a secure tunnel such as Cloudflare Tunnel
  - a shared bearer token via `GPU_WORKER_TOKEN`

## Required environment variables

### On Vercel

- `GPU_WORKER_BASE_URL`
- `GPU_WORKER_TOKEN`
- optionally `REPORT_JSON_PATH` for a static fallback report

### On local GPU worker

- `PYTHON_BIN`
- `GOOGLE_API_KEY`
- `DASHBOARD_ORIGIN`
- `PROFILE_VIDEO_ID`
- `GPU_WORKER_TOKEN`

## FastAPI worker startup

Install:

```bash
cd gpu_backend
pip install -r requirements.txt
```

Run:

```bash
cd "/media/milab-1/009a6625-83db-44d1-8d42-364400c9fc34/Mritunjoys' Workplace/RMG"
cp gpu_backend/worker.env.example gpu_backend/worker.env
bash gpu_backend/run_worker.sh
```

## Health check

The dashboard now checks `GET /health` before upload starts.

- `ok: true` means Vercel can reach the worker
- `ok: false` means upload jobs should be blocked until the worker or tunnel is back online

## Cloudflare Tunnel example

1. Install `cloudflared` on the GPU machine
2. Create a tunnel and map it to the FastAPI worker on port `9000`
3. Use `gpu_backend/cloudflared.example.yml` as the template
4. Start the tunnel so your hostname points at `http://localhost:9000`

Example quick test:

```bash
cloudflared tunnel --url http://localhost:9000
```

For a named tunnel, your Vercel env should point to something like:

`GPU_WORKER_BASE_URL=https://your-worker-domain.example.com`

and both sides must share the same:

`GPU_WORKER_TOKEN=replace_with_a_long_random_secret`

## GPU machine startup options

### Manual

```bash
bash gpu_backend/run_worker.sh
```

### systemd service

```bash
sudo cp gpu_backend/altersense-gpu-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now altersense-gpu-worker
sudo systemctl status altersense-gpu-worker
```

## Current code path

- Next.js proxy logic:
  - `lib/dashboard-jobs.js`
  - `app/api/jobs/*`
  - `app/api/worker-health/route.js`
  - `app/api/operator-image/route.js`
- Local GPU worker:
  - `gpu_backend/main.py`

## Safe deployment rule

Vercel should never be responsible for CUDA execution directly.
