import { fileExists, jobDir, readJobResult } from "@/lib/dashboard-jobs";

export async function GET(_request, { params }) {
  const { jobId } = params;
  const resultPath = `${jobDir(jobId)}/result.json`;
  if (!(await fileExists(resultPath))) {
    return Response.json({ error: "Job result is not ready yet." }, { status: 404 });
  }

  const result = await readJobResult(jobId);
  const reportPath = result.report_json;
  if (!(await fileExists(reportPath))) {
    return Response.json({ error: "Report file is missing." }, { status: 404 });
  }

  const report = JSON.parse(await (await import("node:fs/promises")).readFile(reportPath, "utf-8"));
  return Response.json(report);
}
