import Anthropic from "@anthropic-ai/sdk";
import { toAnthropicTools, executeTool } from "./tools";
import { AgentResult, TraceEvent, ToolCall } from "./types";

const MAX_ITERATIONS = 6;

export async function runClaudeAgent(
  systemPrompt: string,
  userMessage: string,
  onEvent: (event: TraceEvent) => void,
  label: string,
  apiKey: string
): Promise<AgentResult> {
  const client = new Anthropic({ apiKey });
  const tools = toAnthropicTools();
  const allEvents: TraceEvent[] = [];
  const allToolCalls: ToolCall[] = [];

  const emit = (event: TraceEvent) => {
    allEvents.push(event);
    onEvent(event);
  };

  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: userMessage },
  ];

  let finalText = "";

  for (let iter = 0; iter < MAX_ITERATIONS; iter++) {
    const response = await client.messages.create({
      model: "claude-sonnet-4-5",
      max_tokens: 2048,
      system: systemPrompt,
      tools,
      messages,
    });

    const textBlocks = response.content.filter((b) => b.type === "text");
    const toolUseBlocks = response.content.filter((b) => b.type === "tool_use");

    if (textBlocks.length > 0) {
      const thought = textBlocks.map((b) => (b as Anthropic.TextBlock).text).join("\n");
      finalText = thought;
      emit({
        id: `${label}-thought-${iter}`,
        timestamp: Date.now(),
        agent: label,
        type: "thought",
        content: thought,
        provider: "claude",
      });
    }

    if (response.stop_reason !== "tool_use" || toolUseBlocks.length === 0) {
      break;
    }

    const toolResults: Anthropic.ToolResultBlockParam[] = [];

    for (const block of toolUseBlocks) {
      const tb = block as Anthropic.ToolUseBlock;
      const call: ToolCall = {
        id: tb.id,
        name: tb.name,
        input: tb.input as Record<string, unknown>,
      };
      allToolCalls.push(call);

      emit({
        id: `${label}-tool-${tb.id}`,
        timestamp: Date.now(),
        agent: label,
        type: "tool_call",
        content: `${tb.name}(${JSON.stringify(tb.input)})`,
        tool: tb.name,
        provider: "claude",
      });

      const result = await executeTool(call);

      emit({
        id: `${label}-result-${tb.id}`,
        timestamp: Date.now(),
        agent: label,
        type: "tool_result",
        content: result.output,
        tool: tb.name,
        provider: "claude",
      });

      toolResults.push({
        type: "tool_result",
        tool_use_id: tb.id,
        content: result.output,
      });
    }

    messages.push({ role: "assistant", content: response.content });
    messages.push({ role: "user", content: toolResults });
  }

  // If all iterations were consumed by tool calls, force one final text response
  if (!finalText && allToolCalls.length > 0) {
    const finalResponse = await client.messages.create({
      model: "claude-sonnet-4-5",
      max_tokens: 2048,
      system: systemPrompt,
      messages,
    });
    const textBlocks = finalResponse.content.filter((b) => b.type === "text");
    finalText = textBlocks.map((b) => (b as Anthropic.TextBlock).text).join("\n");
    if (finalText) {
      emit({
        id: `${label}-final`,
        timestamp: Date.now(),
        agent: label,
        type: "thought",
        content: finalText,
        provider: "claude",
      });
    }
  }

  return { text: finalText, toolCalls: allToolCalls, events: allEvents };
}
