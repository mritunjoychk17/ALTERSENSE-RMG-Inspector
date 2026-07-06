# Cloudflare Tunnel Setup For The GPU Worker

## Goal

Expose the local FastAPI GPU worker to the Vercel dashboard without opening raw local ports publicly.

## Files used

- `gpu_backend/run_worker.sh`
- `gpu_backend/worker.env`
- `gpu_backend/cloudflared.example.yml`

## 1. Prepare the worker

```bash
cd "/media/milab-1/009a6625-83db-44d1-8d42-364400c9fc34/Mritunjoys' Workplace/RMG"
cp gpu_backend/worker.env.example gpu_backend/worker.env
```

Set at minimum:

- `GPU_WORKER_TOKEN`
- `DASHBOARD_ORIGIN`
- `GOOGLE_API_KEY`

Start the worker:

```bash
bash gpu_backend/run_worker.sh
```

## 2. Start Cloudflare Tunnel

Quick temporary tunnel:

```bash
cloudflared tunnel --url http://localhost:9000
```

Named tunnel:

1. Create the tunnel in Cloudflare
2. Copy `gpu_backend/cloudflared.example.yml` to `gpu_backend/cloudflared.yml`
3. Replace the hostname and tunnel credentials path
4. Run:

```bash
cloudflared tunnel run
```

Optional systemd service:

```bash
sudo cp gpu_backend/cloudflared-altersense-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cloudflared-altersense-worker
sudo systemctl status cloudflared-altersense-worker
```

## 3. Configure Vercel

Set these environment variables in Vercel:

```env
GPU_WORKER_BASE_URL=https://gpu-worker.your-domain.com
GPU_WORKER_TOKEN=replace_with_a_long_random_secret
```

## 4. Verify

From any machine that can reach the tunnel:

```bash
curl -H "Authorization: Bearer replace_with_a_long_random_secret" \
  https://gpu-worker.your-domain.com/health
```

Expected response:

```json
{"ok": true, "worker": "local-gpu"}
```

## 5. Dashboard behavior

The dashboard now checks worker health before upload and shows:

- `Worker Online` when the tunnel and token are valid
- `Worker Offline` when the worker is unreachable or auth fails
