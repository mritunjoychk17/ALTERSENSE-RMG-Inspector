import { loadReport, resolveReportPath, summarizeReliability } from "@/lib/report";

export async function GET() {
  const report = await loadReport();
  return Response.json({
    report,
    meta: {
      reportPath: resolveReportPath(),
      reliability: summarizeReliability(report.stations)
    }
  });
}
