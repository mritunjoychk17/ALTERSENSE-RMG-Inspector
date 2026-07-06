import { fileExists, hasRemoteWorker, jobDir, readUnifiedJobStatus } from "@/lib/dashboard-jobs";

export async function GET(_request, { params }) {
  const { jobId } = params;
  if (hasRemoteWorker()) {
    try {
      return Response.json(await readUnifiedJobStatus(jobId));
    } catch (error) {
      return Response.json({ error: error.message || "Unknown job id." }, { status: 404 });
    }
  }
  const statusPath = `${jobDir(jobId)}/status.json`;
  if (!(await fileExists(statusPath))) {
    return Response.json({ error: "Unknown job id." }, { status: 404 });
  }
  return Response.json(await readUnifiedJobStatus(jobId));
}
