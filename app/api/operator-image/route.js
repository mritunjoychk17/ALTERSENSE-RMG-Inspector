import { readFile } from "node:fs/promises";
import path from "node:path";

import { remoteWorkerBaseUrl } from "@/lib/dashboard-jobs";
import { resolveStationImagePath } from "@/lib/report";

export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const videoId = String(searchParams.get("videoId") || "").trim();
  const stationId = String(searchParams.get("stationId") || "").trim();

  if (!videoId || !stationId) {
    return new Response("Missing videoId or stationId.", { status: 400 });
  }

  if (remoteWorkerBaseUrl()) {
    const params = new URLSearchParams({ videoId, stationId });
    const response = await fetch(`${remoteWorkerBaseUrl()}/operator-image?${params.toString()}`, {
      cache: "no-store",
    });
    if (!response.ok || !response.body) {
      return new Response("Operator image not found.", { status: 404 });
    }
    return new Response(response.body, {
      headers: {
        "Content-Type": response.headers.get("content-type") || "image/jpeg",
        "Cache-Control": "no-store",
      },
    });
  }

  const imagePath = await resolveStationImagePath(videoId, stationId);
  if (!imagePath) {
    return new Response("Operator image not found.", { status: 404 });
  }

  const repoRoot = path.resolve(process.cwd());
  const resolved = path.resolve(imagePath);
  if (!resolved.startsWith(repoRoot)) {
    return new Response("Invalid image path.", { status: 400 });
  }

  const buffer = await readFile(resolved);
  return new Response(buffer, {
    headers: {
      "Content-Type": "image/jpeg",
      "Cache-Control": "no-store",
    },
  });
}
