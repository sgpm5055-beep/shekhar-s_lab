import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";
import { getJudgePrompt } from "../agents/prompts";
import { JudgeScores, Provider, TraceEvent } from "../agents/types";

export async function runJudge(
  topic: string,
  transcript: string,
  judgeModel: Provider,
  onEvent: (event: TraceEvent) => void,
  apiKeys: { claude?: string; gpt?: string }
): Promise<JudgeScores> {
  const prompt = getJudgePrompt(topic, transcript, judgeModel);

  onEvent({
    id: "judge-start",
    timestamp: Date.now(),
    agent: "judge",
    type: "system",
    content: `Judge (${judgeModel === "claude" ? "Claude" : "GPT-4o"}) is evaluating the debate...`,
    provider: judgeModel,
  });

  let raw = "";

  if (judgeModel === "claude") {
    const client = new Anthropic({ apiKey: apiKeys.claude });
    const res = await client.messages.create({
      model: "claude-sonnet-4-5",
      max_tokens: 1024,
      messages: [{ role: "user", content: prompt }],
    });
    raw = (res.content[0] as Anthropic.TextBlock).text;
  } else {
    const client = new OpenAI({ apiKey: apiKeys.gpt });
    const res = await client.chat.completions.create({
      model: "gpt-4o",
      max_tokens: 1024,
      messages: [{ role: "user", content: prompt }],
    });
    raw = res.choices[0].message.content ?? "";
  }

  onEvent({
    id: "judge-verdict",
    timestamp: Date.now(),
    agent: "judge",
    type: "output",
    content: raw,
    provider: judgeModel,
  });

  const fallback: JudgeScores = {
    claudeScore: 50,
    gptScore: 50,
    winner: "tie",
    reasoning: "Unable to parse judge verdict.",
    criteria: {
      accuracy: { claude: 5, gpt: 5 },
      depth: { claude: 5, gpt: 5 },
      clarity: { claude: 5, gpt: 5 },
      persuasion: { claude: 5, gpt: 5 },
    },
  };

  const jsonMatch = raw.match(/\{[\s\S]*\}/);
  if (!jsonMatch) return fallback;

  try {
    return JSON.parse(jsonMatch[0]) as JudgeScores;
  } catch {
    return fallback;
  }
}
