import { runAgent, oppositeProvider } from "../agents/dispatcher";
import {
  getSupervisorPrompt,
  getResearcherPrompt,
  getWriterPrompt,
  getCriticPrompt,
  getRevisorPrompt,
} from "../agents/prompts";
import { OrchestrationConfig, TraceEvent, OutputBubble, Provider } from "../agents/types";
import { PipelineContext } from "./context";

export interface OrchCallbacks {
  onEvent: (event: TraceEvent) => void;
  onBubble: (bubble: OutputBubble) => void;
  onAgentStatus: (agent: string, status: string) => void;
  onToolCount: (agent: string, count: number) => void;
}

function makeEmitter(
  label: string,
  provider: Provider,
  callbacks: OrchCallbacks,
  localToolCounts: Record<string, number>
) {
  return (event: TraceEvent) => {
    if (event.type === "tool_call") {
      localToolCounts[label] = (localToolCounts[label] ?? 0) + 1;
      callbacks.onToolCount(label, localToolCounts[label]);
    }
    callbacks.onEvent(event);
  };
}

export async function runOrchestration(
  config: OrchestrationConfig,
  callbacks: OrchCallbacks
): Promise<string> {
  const { task, pattern, provider, claudeApiKey, gptApiKey } = config;
  const { onEvent, onBubble, onAgentStatus } = callbacks;
  const apiKeys = { claude: claudeApiKey, gpt: gptApiKey };
  const ctx = new PipelineContext();
  const localToolCounts: Record<string, number> = {};

  onEvent({
    id: "sys-start",
    timestamp: Date.now(),
    agent: "system",
    type: "system",
    content: `Orchestration started: "${task}" | Pattern: ${pattern}`,
  });

  const bubble = (
    agent: string,
    role: OutputBubble["role"],
    content: string,
    p: Provider
  ) => {
    onBubble({
      id: `${agent}-${Date.now()}`,
      agent,
      role,
      content,
      provider: p,
      timestamp: Date.now(),
    });
  };

  const hand = (from: string, to: string) => {
    onEvent({
      id: `handoff-${from}-${to}-${Date.now()}`,
      timestamp: Date.now(),
      agent: "system",
      type: "handoff",
      content: `${from} → ${to}`,
    });
  };

  if (pattern === "research-report") {
    onAgentStatus("supervisor", "thinking");
    const supResult = await runAgent(
      provider,
      getSupervisorPrompt(task),
      `Plan research for: ${task}`,
      makeEmitter("supervisor", provider, callbacks, localToolCounts),
      "supervisor",
      apiKeys
    );
    onAgentStatus("supervisor", "done");
    ctx.set("plan", supResult.text);
    bubble("Supervisor", "supervisor", supResult.text, provider);
    hand("Supervisor", "Researcher");

    onAgentStatus("researcher", "thinking");
    const resResult = await runAgent(
      provider,
      getResearcherPrompt(task, 0),
      `Research task: ${task}\n\nPlan: ${supResult.text}`,
      makeEmitter("researcher", provider, callbacks, localToolCounts),
      "researcher",
      apiKeys
    );
    onAgentStatus("researcher", "done");
    ctx.set("research", resResult.text);
    bubble("Researcher", "researcher", resResult.text, provider);
    hand("Researcher", "Writer");

    onAgentStatus("writer", "thinking");
    const writeResult = await runAgent(
      provider,
      getWriterPrompt(task, ctx.summary()),
      `Write the report for: ${task}`,
      makeEmitter("writer", provider, callbacks, localToolCounts),
      "writer",
      apiKeys
    );
    onAgentStatus("writer", "done");
    ctx.set("draft", writeResult.text);
    bubble("Writer", "writer", writeResult.text, provider);
    hand("Writer", "Critic");

    onAgentStatus("critic", "thinking");
    const critiqueResult = await runAgent(
      provider,
      getCriticPrompt(writeResult.text),
      `Critique this report`,
      makeEmitter("critic", provider, callbacks, localToolCounts),
      "critic",
      apiKeys
    );
    onAgentStatus("critic", "done");
    bubble("Critic", "critic", critiqueResult.text, provider);

    return writeResult.text;
  }

  if (pattern === "parallel-research") {
    const altProvider = oppositeProvider(provider);

    onAgentStatus("researcher-a", "thinking");
    onAgentStatus("researcher-b", "thinking");

    const [resA, resB] = await Promise.all([
      runAgent(
        provider,
        getResearcherPrompt(task, 0),
        `Research task: ${task}`,
        makeEmitter("researcher-a", provider, callbacks, localToolCounts),
        "researcher-a",
        apiKeys
      ),
      runAgent(
        altProvider,
        getResearcherPrompt(task, 1),
        `Research alternative angles for: ${task}`,
        makeEmitter("researcher-b", altProvider, callbacks, localToolCounts),
        "researcher-b",
        apiKeys
      ),
    ]);

    onAgentStatus("researcher-a", "done");
    onAgentStatus("researcher-b", "done");
    ctx.set("research-a", resA.text);
    ctx.set("research-b", resB.text);
    bubble(`Researcher A (${provider === "claude" ? "Claude" : "GPT"})`, "researcher", resA.text, provider);
    bubble(`Researcher B (${altProvider === "claude" ? "Claude" : "GPT"})`, "researcher", resB.text, altProvider);
    hand("Researchers", "Writer");

    onAgentStatus("writer", "thinking");
    const writeResult = await runAgent(
      provider,
      getWriterPrompt(task, ctx.summary()),
      `Synthesize both research streams into a unified report for: ${task}`,
      makeEmitter("writer", provider, callbacks, localToolCounts),
      "writer",
      apiKeys
    );
    onAgentStatus("writer", "done");
    ctx.set("draft", writeResult.text);
    bubble("Writer", "writer", writeResult.text, provider);
    hand("Writer", "Critic");

    onAgentStatus("critic", "thinking");
    const critiqueResult = await runAgent(
      altProvider,
      getCriticPrompt(writeResult.text),
      `Critically evaluate this synthesized report`,
      makeEmitter("critic", altProvider, callbacks, localToolCounts),
      "critic",
      apiKeys
    );
    onAgentStatus("critic", "done");
    bubble("Critic", "critic", critiqueResult.text, altProvider);

    return writeResult.text;
  }

  if (pattern === "adversarial") {
    const altProvider = oppositeProvider(provider);

    onAgentStatus("writer", "thinking");
    const draft = await runAgent(
      provider,
      getWriterPrompt(task, "No prior research — draw from your knowledge."),
      `Write an initial draft on: ${task}`,
      makeEmitter("writer", provider, callbacks, localToolCounts),
      "writer",
      apiKeys
    );
    onAgentStatus("writer", "done");
    ctx.set("draft", draft.text);
    bubble("Writer", "writer", draft.text, provider);
    hand("Writer", `Critic (${altProvider})`);

    onAgentStatus("critic", "thinking");
    const critique = await runAgent(
      altProvider,
      getCriticPrompt(draft.text),
      `Adversarially critique this draft. Be thorough and demanding.`,
      makeEmitter("critic", altProvider, callbacks, localToolCounts),
      "critic",
      apiKeys
    );
    onAgentStatus("critic", "done");
    bubble("Critic", "critic", critique.text, altProvider);
    hand("Critic", "Revisor");

    onAgentStatus("revisor", "thinking");
    const revised = await runAgent(
      provider,
      getRevisorPrompt(draft.text, critique.text),
      `Revise the draft incorporating the critique`,
      makeEmitter("revisor", provider, callbacks, localToolCounts),
      "revisor",
      apiKeys
    );
    onAgentStatus("revisor", "done");
    bubble("Revisor", "revisor", revised.text, provider);

    return revised.text;
  }

  throw new Error(`Unknown pattern: ${pattern}`);
}
