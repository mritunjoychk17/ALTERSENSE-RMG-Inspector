# Vercel Web App

## Purpose

This web app is the **hostable operator dashboard** for ALTERSENSE.

It is designed for:

- Vercel-hosted frontend delivery
- reading the latest verified operator report JSON
- showing station KPIs with reliability badges

It is **not** meant to run full Python video inference inside Vercel.

## Why this architecture

The repo's reliable path is:

1. GPU worker or on-prem Python service runs Stage 1 and Stage 2
2. worker writes the verified JSON report artifact
3. Vercel app reads and displays that report

This keeps the hosted UI fast and stable while leaving heavy video processing in the correct runtime.

## Environment

Vercel UI:

`REPORT_JSON_PATH`

Defaults to:

`artifacts/altersense/operator_report_cam33_phase_mixed_station136_plus5.json`

Local GPU worker only:

- `PYTHON_BIN`
- `GOOGLE_API_KEY`

These should stay on the local/on-prem processing machine, not in the Vercel project, unless you intentionally move Python inference into another hosted runtime.

## Local run

```bash
npm install
npm run dev
```

Then open:

`http://127.0.0.1:3000`

## Vercel deploy

1. import the repo into Vercel
2. use the default Next.js build settings
3. optionally set `REPORT_JSON_PATH`
4. deploy

## Recommended first GitHub push

Keep the first hosted push focused on the dashboard app and lightweight config:

- `app/`
- `lib/`
- `configs/`
- `docs/`
- `scripts/`
- `package.json`
- `package-lock.json`
- `next.config.mjs`
- `jsconfig.json`
- `vercel.json`
- `.env.example`

Do not push:

- `datasets/`
- `artifacts/`
- `rmg/`
- raw videos, zips, pdfs, and local model weights

## API routes

- `/api/report`
- `/api/pipeline`
