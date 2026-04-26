import { runClaudeAgent } from "./anthropic-runtime";
import { runGPTAgent } from "./openai-runtime";
import { AgentResult, Provider, TraceEvent } from "./types";

export async function runAgent(
  provider: Provider,
  systemPrompt: string,
  userMessage: string,
  onEvent: (event: TraceEvent) => void,
  label: string,
  apiKeys: { claude?: string; gpt?: string }
): Promise<AgentResult> {
  if (provider === "claude") {
    if (!apiKeys.claude) throw new Error("Claude API key required");
    return runClaudeAgent(systemPrompt, userMessage, onEvent, label, apiKeys.claude);
  } else {
    if (!apiKeys.gpt) throw new Error("OpenAI API key required");
    return runGPTAgent(systemPrompt, userMessage, onEvent, label, apiKeys.gpt);
  }
}

export function oppositeProvider(p: Provider): Provider {
  return p === "claude" ? "gpt" : "claude";
}
