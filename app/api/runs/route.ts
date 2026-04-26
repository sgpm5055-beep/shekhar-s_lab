import { NextRequest, NextResponse } from "next/server";
import { RunRecord } from "@/lib/agents/types";

// In-memory store for MVP (Phase 2: use Upstash Redis)
const runs: RunRecord[] = [];

export async function GET() {
  return NextResponse.json(runs.slice(-50).reverse());
}

export async function POST(req: NextRequest) {
  const body = await req.json();
  const run: RunRecord = {
    id: crypto.randomUUID(),
    timestamp: Date.now(),
    ...body,
  };
  runs.push(run);
  return NextResponse.json(run);
}
