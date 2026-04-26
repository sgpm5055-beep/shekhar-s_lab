import { NextRequest } from "next/server";
import { runBattle } from "@/lib/battle/engine";
import { runJudge } from "@/lib/battle/judge";
import { BattleConfig, TraceEvent, OutputBubble } from "@/lib/agents/types";

export const maxDuration = 300;

function sseEvent(data: Record<string, unknown>): string {
  return `data: ${JSON.stringify(data)}\n\n`;
}

export async function POST(req: NextRequest) {
  const claudeApiKey = process.env.ANTHROPIC_API_KEY;
  const gptApiKey = process.env.OPENAI_API_KEY;

  if (!claudeApiKey || !gptApiKey) {
    return new Response(JSON.stringify({ error: "API keys not configured on server" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }

  const body = await req.json();
  const config: BattleConfig = {
    topic: body.topic,
    rounds: body.rounds ?? 2,
    judgeModel: body.judgeModel ?? "claude",
    claudeApiKey,
    gptApiKey,
  };

  const encoder = new TextEncoder();
  let controller!: ReadableStreamDefaultController;

  const stream = new ReadableStream({
    start(c) {
      (controller as ReadableStreamDefaultController) = c;
    },
  });

  const send = (data: Record<string, unknown>) => {
    try {
      controller.enqueue(encoder.encode(sseEvent(data)));
    } catch {
      // stream closed
    }
  };

  (async () => {
    try {
      const { transcript } = await runBattle(config, {
        onEvent: (e: TraceEvent) => send({ type: "event", payload: e }),
        onBubble: (b: OutputBubble) => send({ type: "bubble", payload: b }),
        onAgentStatus: (agent: string, status: string) =>
          send({ type: "agent_status", agent, status }),
        onToolCount: (agent: string, count: number) =>
          send({ type: "tool_count", agent, count }),
      });

      send({
        type: "event",
        payload: {
          id: "judging",
          timestamp: Date.now(),
          agent: "system",
          type: "system",
          content: "Debate concluded. Calling judge...",
        },
      });
      send({ type: "agent_status", agent: "judge", status: "thinking" });

      const scores = await runJudge(
        config.topic,
        transcript,
        config.judgeModel,
        (e: TraceEvent) => send({ type: "event", payload: e }),
        { claude: claudeApiKey, gpt: gptApiKey }
      );

      send({ type: "scores", payload: scores });
      send({ type: "done" });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unknown error";
      send({ type: "error", message });
    } finally {
      try {
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
      } catch {
        // already closed
      }
    }
  })();

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
