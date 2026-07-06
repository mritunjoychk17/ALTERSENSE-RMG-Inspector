import { fileExists, jobDir, readJobResult, readJobStatus } from "@/lib/dashboard-jobs";

export async function GET(_request, { params }) {
  const { jobId } = params;
  const statusPath = `${jobDir(jobId)}/status.json`;
  if (!(await fileExists(statusPath))) {
    return Response.json({ error: "Unknown job id." }, { status: 404 });
  }

  const status = await readJobStatus(jobId);
  const resultPath = `${jobDir(jobId)}/result.json`;
  const hasResult = await fileExists(resultPath);
  const result = hasResult ? await readJobResult(jobId) : null;

  return Response.json({
    jobId,
    status,
    result,
    urls: {
      report: `/api/jobs/${jobId}/report`,
      overlay: `/api/jobs/${jobId}/artifacts/stage1_overlay.mp4`,
      log: `/api/jobs/${jobId}/artifacts/pipeline.log`,
    },
  });
}
