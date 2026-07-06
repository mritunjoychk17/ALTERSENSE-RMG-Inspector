import { cam33ProfileId, createUploadJob } from "@/lib/dashboard-jobs";

export async function POST(request) {
  const form = await request.formData();
  const video = form.get("video");
  if (!video || typeof video === "string") {
    return Response.json({ error: "Please attach a video file." }, { status: 400 });
  }

  const sampleEvery = Number(form.get("sampleEvery") || 20);
  const presentThreshold = Number(form.get("presentThreshold") || 0.5);
  const device = String(form.get("device") || "cuda");
  const poseBackend = String(form.get("poseBackend") || "auto");

  const job = await createUploadJob({
    file: video,
    sampleEvery,
    presentThreshold,
    device,
    poseBackend,
  });

  return Response.json({
    jobId: job.jobId,
    profileVideoId: cam33ProfileId(),
    message: "Upload accepted. The operator evaluation job has started.",
  });
}
