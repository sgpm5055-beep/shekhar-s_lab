import { create } from "zustand";
import { TraceEvent, OutputBubble, JudgeScores, Provider, AgentStatus } from "../agents/types";

interface PlatformState {
  // Mode
  mode: "battle" | "orchestrate";
  setMode: (mode: "battle" | "orchestrate") => void;

  // Shared execution state
  running: boolean;
  started: boolean;
  events: TraceEvent[];
  bubbles: OutputBubble[];
  agentStatuses: Record<string, AgentStatus>;
  toolCounts: Record<string, number>;
  error: string | null;
  elapsed: number;
  activeTab: "output" | "trace";

  // Battle-specific
  topic: string;
  rounds: number;
  judgeModel: Provider;
  scores: JudgeScores | null;

  // Orchestration-specific
  orchTask: string;
  orchPattern: string;
  orchProvider: Provider;
  finalOutput: string;

  // Actions
  setRunning: (v: boolean) => void;
  setStarted: (v: boolean) => void;
  addEvent: (e: TraceEvent) => void;
  addBubble: (b: OutputBubble) => void;
  setAgentStatus: (agent: string, status: AgentStatus) => void;
  setToolCount: (agent: string, count: number) => void;
  setError: (e: string | null) => void;
  setElapsed: (n: number) => void;
  setActiveTab: (t: "output" | "trace") => void;
  setScores: (s: JudgeScores) => void;
  setFinalOutput: (s: string) => void;
  setTopic: (s: string) => void;
  setRounds: (n: number) => void;
  setJudgeModel: (m: Provider) => void;
  setOrchTask: (s: string) => void;
  setOrchPattern: (s: string) => void;
  setOrchProvider: (p: Provider) => void;
  reset: () => void;
}

const initialState = {
  running: false,
  started: false,
  events: [] as TraceEvent[],
  bubbles: [] as OutputBubble[],
  agentStatuses: {} as Record<string, AgentStatus>,
  toolCounts: {} as Record<string, number>,
  error: null,
  elapsed: 0,
  activeTab: "output" as const,
  scores: null,
  finalOutput: "",
};

export const usePlatformStore = create<PlatformState>((set) => ({
  mode: "battle",
  topic: "AI will do more harm than good to humanity",
  rounds: 2,
  judgeModel: "claude" as Provider,
  orchTask: "",
  orchPattern: "research-report",
  orchProvider: "claude" as Provider,
  ...initialState,

  setMode: (mode) => set({ mode }),
  setRunning: (running) => set({ running }),
  setStarted: (started) => set({ started }),
  addEvent: (e) => set((s) => ({ events: [...s.events, e] })),
  addBubble: (b) => set((s) => ({ bubbles: [...s.bubbles, b] })),
  setAgentStatus: (agent, status) =>
    set((s) => ({ agentStatuses: { ...s.agentStatuses, [agent]: status } })),
  setToolCount: (agent, count) =>
    set((s) => ({ toolCounts: { ...s.toolCounts, [agent]: count } })),
  setError: (error) => set({ error }),
  setElapsed: (elapsed) => set({ elapsed }),
  setActiveTab: (activeTab) => set({ activeTab }),
  setScores: (scores) => set({ scores }),
  setFinalOutput: (finalOutput) => set({ finalOutput }),
  setTopic: (topic) => set({ topic }),
  setRounds: (rounds) => set({ rounds }),
  setJudgeModel: (judgeModel) => set({ judgeModel }),
  setOrchTask: (orchTask) => set({ orchTask }),
  setOrchPattern: (orchPattern) => set({ orchPattern }),
  setOrchProvider: (orchProvider) => set({ orchProvider }),
  reset: () => set({ ...initialState }),
}));
