import OpenAI from "openai";
import { toOpenAITools, executeTool } from "./tools";
import { AgentResult, TraceEvent, ToolCall } from "./types";

const MAX_ITERATIONS = 6;

export async function runGPTAgent(
  systemPrompt: string,
  userMessage: string,
  onEvent: (event: TraceEvent) => void,
  label: string,
  apiKey: string
): Promise<AgentResult> {
  const client = new OpenAI({ apiKey });
  const tools = toOpenAITools();
  const allEvents: TraceEvent[] = [];
  const allToolCalls: ToolCall[] = [];

  const emit = (event: TraceEvent) => {
    allEvents.push(event);
    onEvent(event);
  };

  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    { role: "system", content: systemPrompt },
    { role: "user", content: userMessage },
  ];

  let finalText = "";

  for (let iter = 0; iter < MAX_ITERATIONS; iter++) {
    const response = await client.chat.completions.create({
      model: "gpt-4o",
      max_tokens: 2048,
      tools,
      messages,
    });

    const choice = response.choices[0];
    const message = choice.message;

    if (message.content) {
      finalText = message.content;
      emit({
        id: `${label}-thought-${iter}`,
        timestamp: Date.now(),
        agent: label,
        type: "thought",
        content: message.content,
        provider: "gpt",
      });
    }

    if (choice.finish_reason !== "tool_calls" || !message.tool_calls?.length) {
      break;
    }

    const toolMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [];

    for (const tc of message.tool_calls) {
      if (tc.type !== "function") continue;
      const fn = tc.function;

      let input: Record<string, unknown> = {};
      try {
        input = JSON.parse(fn.arguments) as Record<string, unknown>;
      } catch {
        input = { raw: fn.arguments };
      }

      const call: ToolCall = {
        id: tc.id,
        name: fn.name,
        input,
      };
      allToolCalls.push(call);

      emit({
        id: `${label}-tool-${tc.id}`,
        timestamp: Date.now(),
        agent: label,
        type: "tool_call",
        content: `${fn.name}(${fn.arguments})`,
        tool: fn.name,
        provider: "gpt",
      });

      const result = await executeTool(call);

      emit({
        id: `${label}-result-${tc.id}`,
        timestamp: Date.now(),
        agent: label,
        type: "tool_result",
        content: result.output,
        tool: fn.name,
        provider: "gpt",
      });

      toolMessages.push({
        role: "tool",
        tool_call_id: tc.id,
        content: result.output,
      });
    }

    messages.push(message);
    messages.push(...toolMessages);
  }

  // If all iterations were consumed by tool calls, force one final text response
  if (!finalText && allToolCalls.length > 0) {
    const finalResponse = await client.chat.completions.create({
      model: "gpt-4o",
      max_tokens: 2048,
      messages,
    });
    finalText = finalResponse.choices[0].message.content ?? "";
    if (finalText) {
      emit({
        id: `${label}-final`,
        timestamp: Date.now(),
        agent: label,
        type: "thought",
        content: finalText,
        provider: "gpt",
      });
    }
  }

  return { text: finalText, toolCalls: allToolCalls, events: allEvents };
}
