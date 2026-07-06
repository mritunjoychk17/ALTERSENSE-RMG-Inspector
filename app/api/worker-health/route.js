import { readWorkerHealth } from "@/lib/dashboard-jobs";

export async function GET() {
  return Response.json(await readWorkerHealth());
}
