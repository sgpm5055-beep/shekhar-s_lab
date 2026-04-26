import { ToolCall, ToolResult } from "./types";

const TOOL_DEFS = [
  {
    name: "web_search",
    description:
      "Search the web for current information on a topic. Returns relevant excerpts and sources.",
    parameters: {
      type: "object" as const,
      properties: {
        query: { type: "string", description: "The search query" },
        num_results: {
          type: "number",
          description: "Number of results to return (1-5)",
        },
      },
      required: ["query"],
    },
  },
  {
    name: "fact_check",
    description:
      "Verify a specific claim or statement for accuracy. Returns verdict and evidence.",
    parameters: {
      type: "object" as const,
      properties: {
        claim: { type: "string", description: "The claim to verify" },
        context: {
          type: "string",
          description: "Optional context around the claim",
        },
      },
      required: ["claim"],
    },
  },
  {
    name: "analyze",
    description:
      "Perform structured analysis on data or text. Returns insights and patterns.",
    parameters: {
      type: "object" as const,
      properties: {
        content: { type: "string", description: "Content to analyze" },
        analysis_type: {
          type: "string",
          enum: ["sentiment", "themes", "statistics", "comparison", "summary"],
          description: "Type of analysis to perform",
        },
      },
      required: ["content", "analysis_type"],
    },
  },
  {
    name: "write_section",
    description:
      "Write a structured section of a document with proper formatting.",
    parameters: {
      type: "object" as const,
      properties: {
        title: { type: "string", description: "Section title" },
        content: { type: "string", description: "Section content to write" },
        format: {
          type: "string",
          enum: ["paragraph", "bullet_points", "numbered_list"],
        },
      },
      required: ["title", "content"],
    },
  },
  {
    name: "critique",
    description:
      "Critically evaluate a piece of text or argument. Returns strengths, weaknesses, and suggestions.",
    parameters: {
      type: "object" as const,
      properties: {
        text: { type: "string", description: "Text to critique" },
        focus: {
          type: "string",
          enum: ["logic", "evidence", "clarity", "completeness", "overall"],
        },
      },
      required: ["text"],
    },
  },
];

// Anthropic format: { name, description, input_schema }
export function toAnthropicTools() {
  return TOOL_DEFS.map((t) => ({
    name: t.name,
    description: t.description,
    input_schema: t.parameters,
  }));
}

// OpenAI format: { type: "function", function: { name, description, parameters } }
export function toOpenAITools() {
  return TOOL_DEFS.map((t) => ({
    type: "function" as const,
    function: {
      name: t.name,
      description: t.description,
      parameters: t.parameters,
    },
  }));
}

// Simulated tool execution (Phase 2: replace with real APIs)
export async function executeTool(call: ToolCall): Promise<ToolResult> {
  await new Promise((r) => setTimeout(r, 300 + Math.random() * 700));

  let output = "";

  switch (call.name) {
    case "web_search": {
      const q = call.input.query as string;
      output = `Search results for "${q}":
1. [Source: research-journal.org] Recent studies show significant developments in ${q}, with experts noting key breakthroughs in the past 18 months.
2. [Source: techreview.com] Analysis of ${q} reveals three main perspectives: technological, economic, and social implications.
3. [Source: academic-press.edu] Peer-reviewed findings suggest that ${q} has measurable impact on global systems, with 67% of researchers agreeing on core findings.`;
      break;
    }
    case "fact_check": {
      const claim = call.input.claim as string;
      const verdict = Math.random() > 0.3 ? "SUPPORTED" : "PARTIALLY SUPPORTED";
      output = `Fact-check verdict: ${verdict}
Claim: "${claim}"
Evidence: Multiple credible sources corroborate the core assertion. Primary sources include peer-reviewed literature and official reports. Minor caveats noted regarding scope and timeframe.
Confidence: ${Math.floor(70 + Math.random() * 25)}%`;
      break;
    }
    case "analyze": {
      const type = call.input.analysis_type as string;
      output = `Analysis (${type}):
Key findings:
• Primary theme: Strong coherence in the main argument structure
• Secondary patterns: Three recurring motifs identified across the content
• Statistical significance: Effect size Cohen's d = 0.72 (medium-large)
• Recommendation: The analysis supports the central thesis with 85% confidence`;
      break;
    }
    case "write_section": {
      const title = call.input.title as string;
      output = `Section "${title}" drafted successfully.
Word count: ${Math.floor(150 + Math.random() * 200)}
Structure: Introduction → Evidence → Analysis → Conclusion
Quality score: 8.4/10`;
      break;
    }
    case "critique": {
      const focus = (call.input.focus as string) || "overall";
      output = `Critique (${focus}):
Strengths: Clear structure, well-supported claims, logical flow
Weaknesses: Could benefit from more diverse sources; some assertions lack quantitative backing
Suggestions: Add 2-3 counter-arguments to strengthen credibility; expand the conclusion section
Overall score: ${Math.floor(6 + Math.random() * 3)}/10`;
      break;
    }
    default:
      output = `Tool "${call.name}" executed successfully.`;
  }

  return { id: call.id, name: call.name, output };
}
