import { runAgent } from "../agents/dispatcher";
import { getBattleSystemPrompt } from "../agents/prompts";
import { BattleConfig, TraceEvent, OutputBubble } from "../agents/types";

export interface BattleCallbacks {
  onEvent: (event: TraceEvent) => void;
  onBubble: (bubble: OutputBubble) => void;
  onAgentStatus: (agent: string, status: string) => void;
  onToolCount: (agent: string, count: number) => void;
}

export async function runBattle(
  config: BattleConfig,
  callbacks: BattleCallbacks
): Promise<{ transcript: string; claudeArgs: string[]; gptArgs: string[] }> {
  const { topic, rounds, claudeApiKey, gptApiKey } = config;
  const { onEvent, onBubble, onAgentStatus, onToolCount } = callbacks;

  const apiKeys = { claude: claudeApiKey, gpt: gptApiKey };
  const claudeArgs: string[] = [];
  const gptArgs: string[] = [];
  const transcriptLines: string[] = [];

  let claudeToolCount = 0;
  let gptToolCount = 0;

  const emit = (event: TraceEvent) => {
    if (event.type === "tool_call") {
      if (event.provider === "claude") {
        claudeToolCount++;
        onToolCount("claude", claudeToolCount);
      } else {
        gptToolCount++;
        onToolCount("gpt", gptToolCount);
      }
    }
    onEvent(event);
  };

  // Intro system event
  onEvent({
    id: "sys-start",
    timestamp: Date.now(),
    agent: "system",
    type: "system",
    content: `Battle started: "${topic}" — ${rounds} rounds`,
  });

  let previousArgument = "";

  for (let round = 1; round <= rounds; round++) {
    // Claude argues PRO
    onAgentStatus("claude", "thinking");
    const claudeMsg = round === 1
      ? `Make your opening argument FOR: "${topic}"`
      : `Round ${round}. Your opponent (GPT-4o) argued: "${previousArgument.slice(0, 500)}..."\n\nNow rebut and advance your case FOR: "${topic}"`;

    const claudeResult = await runAgent(
      "claude",
      getBattleSystemPrompt("pro", topic),
      claudeMsg,
      emit,
      "claude",
      apiKeys
    );

    onAgentStatus("claude", "done");
    claudeArgs.push(claudeResult.text);
    transcriptLines.push(`[CLAUDE - Round ${round}]: ${claudeResult.text}`);

    onBubble({
      id: `claude-${round}`,
      agent: "Claude",
      role: "claude",
      content: claudeResult.text,
      provider: "claude",
      round,
      timestamp: Date.now(),
    });

    onEvent({
      id: `handoff-to-gpt-${round}`,
      timestamp: Date.now(),
      agent: "system",
      type: "handoff",
      content: `Claude finished round ${round} → GPT-4o to respond`,
    });

    previousArgument = claudeResult.text;

    // GPT argues CON
    onAgentStatus("gpt", "thinking");
    const gptMsg = `Round ${round}. Your opponent (Claude) just argued: "${previousArgument.slice(0, 500)}..."\n\nNow rebut and advance your case AGAINST: "${topic}"`;

    const gptResult = await runAgent(
      "gpt",
      getBattleSystemPrompt("con", topic),
      gptMsg,
      emit,
      "gpt",
      apiKeys
    );

    onAgentStatus("gpt", "done");
    gptArgs.push(gptResult.text);
    transcriptLines.push(`[GPT-4o - Round ${round}]: ${gptResult.text}`);

    onBubble({
      id: `gpt-${round}`,
      agent: "GPT-4o",
      role: "gpt",
      content: gptResult.text,
      provider: "gpt",
      round,
      timestamp: Date.now(),
    });

    previousArgument = gptResult.text;

    if (round < rounds) {
      onEvent({
        id: `handoff-to-claude-${round}`,
        timestamp: Date.now(),
        agent: "system",
        type: "handoff",
        content: `GPT-4o finished round ${round} → Claude to respond`,
      });
    }
  }

  return {
    transcript: transcriptLines.join("\n\n"),
    claudeArgs,
    gptArgs,
  };
}
