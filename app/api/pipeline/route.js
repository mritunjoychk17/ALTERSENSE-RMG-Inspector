import { pipelineDefinition } from "@/lib/report";

export async function GET() {
  return Response.json(pipelineDefinition());
}
