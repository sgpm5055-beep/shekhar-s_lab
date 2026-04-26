export type PatternId = "research-report" | "parallel-research" | "adversarial";

export interface Pattern {
  id: PatternId;
  name: string;
  description: string;
  agents: string[];
  flow: "sequential" | "parallel" | "adversarial";
  crossModel: boolean;
  icon: string;
  color: string;
}

export const PATTERNS: Pattern[] = [
  {
    id: "research-report",
    name: "Research → Analyze → Write → Critique",
    description:
      "A sequential pipeline where each agent hands off to the next, building a comprehensive research report from scratch.",
    agents: ["Supervisor", "Researcher", "Writer", "Critic"],
    flow: "sequential",
    crossModel: false,
    icon: "📋",
    color: "blue",
  },
  {
    id: "parallel-research",
    name: "Parallel Research → Merge → Critique",
    description:
      "Two researchers work simultaneously using different AI providers, then merge findings. Maximizes coverage and cross-model diversity.",
    agents: ["Researcher A (Claude)", "Researcher B (GPT)", "Writer", "Critic"],
    flow: "parallel",
    crossModel: true,
    icon: "⚡",
    color: "purple",
  },
  {
    id: "adversarial",
    name: "Propose → Critique → Revise",
    description:
      "Writer drafts, then an opposing AI provider critiques adversarially, then the writer revises. Produces battle-hardened output.",
    agents: ["Writer", "Critic (opposing model)", "Revisor"],
    flow: "adversarial",
    crossModel: true,
    icon: "⚔️",
    color: "red",
  },
];

export function getPattern(id: string): Pattern | undefined {
  return PATTERNS.find((p) => p.id === id);
}
