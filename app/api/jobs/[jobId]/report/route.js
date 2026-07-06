import { hasRemoteWorker, readUnifiedJobReport } from "@/lib/dashboard-jobs";

export async function GET(_request, { params }) {
  const { jobId } = params;
  try {
    return Response.json(await readUnifiedJobReport(jobId));
  } catch (error) {
    return Response.json(
      { error: error.message || (hasRemoteWorker() ? "Remote report is not ready yet." : "Job result is not ready yet.") },
      { status: 404 }
    );
  }
}
