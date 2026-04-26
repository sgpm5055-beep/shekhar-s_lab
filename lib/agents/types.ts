export type Provider = "claude" | "gpt";

export type AgentRole =
  | "supervisor"
  | "researcher"
  | "writer"
  | "critic"
  | "revisor"
  | "judge"
  | "claude"
  | "gpt";

export type AgentStatus = "idle" | "thinking" | "tool_use" | "done" | "error";

export interface TraceEvent {
  id: string;
  timestamp: number;
  agent: string;
  type:
    | "thought"
    | "tool_call"
    | "tool_result"
    | "handoff"
    | "output"
    | "error"
    | "system";
  content: string;
  tool?: string;
  provider?: Provider;
}

export interface OutputBubble {
  id: string;
  agent: string;
  role: AgentRole;
  content: string;
  provider?: Provider;
  round?: number;
  timestamp: number;
}

export interface ToolCall {
  id: string;
  name: string;
  input: Record<string, unknown>;
}

export interface ToolResult {
  id: string;
  name: string;
  output: string;
}

export interface AgentResult {
  text: string;
  toolCalls: ToolCall[];
  events: TraceEvent[];
}

export interface JudgeScores {
  claudeScore: number;
  gptScore: number;
  winner: "claude" | "gpt" | "tie";
  reasoning: string;
  criteria: {
    accuracy: { claude: number; gpt: number };
    depth: { claude: number; gpt: number };
    clarity: { claude: number; gpt: number };
    persuasion: { claude: number; gpt: number };
  };
}

export interface BattleConfig {
  topic: string;
  rounds: number;
  judgeModel: Provider;
  claudeApiKey?: string;
  gptApiKey?: string;
}

export interface OrchestrationConfig {
  task: string;
  pattern: string;
  provider: Provider;
  claudeApiKey?: string;
  gptApiKey?: string;
}

export interface RunRecord {
  id: string;
  mode: "battle" | "orchestrate";
  timestamp: number;
  topic?: string;
  task?: string;
  pattern?: string;
  winner?: string;
  output?: string;
  events: TraceEvent[];
}
