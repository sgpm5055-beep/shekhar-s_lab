export function stripMarkdown(text: string): string {
  return text
    // Remove headers (## Title → Title, with newline preserved)
    .replace(/^#{1,6}\s+(.+)$/gm, "$1")
    // Remove bold + italic combined
    .replace(/\*{3}([^*]+)\*{3}/g, "$1")
    .replace(/_{3}([^_]+)_{3}/g, "$1")
    // Remove bold
    .replace(/\*{2}([^*]+)\*{2}/g, "$1")
    .replace(/_{2}([^_]+)_{2}/g, "$1")
    // Remove italic
    .replace(/\*([^*\n]+)\*/g, "$1")
    .replace(/_([^_\n]+)_/g, "$1")
    // Remove inline code
    .replace(/`([^`]+)`/g, "$1")
    // Remove fenced code blocks (keep content)
    .replace(/```[^\n]*\n([\s\S]*?)```/g, "$1")
    // Remove blockquotes
    .replace(/^>\s*/gm, "")
    // Replace unordered list markers with bullet
    .replace(/^\s*[-*+]\s+/gm, "• ")
    // Remove ordered list numbers (keep text)
    .replace(/^\s*\d+\.\s+/gm, "")
    // Remove horizontal rules
    .replace(/^[-*_]{3,}\s*$/gm, "")
    // Remove markdown links — keep label text
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    // Remove image syntax
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1")
    // Collapse 3+ blank lines to 2
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export function formatElapsed(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

export function truncate(text: string, max = 120): string {
  return text.length > max ? text.slice(0, max) + "…" : text;
}

export function agentColor(role: string): string {
  const map: Record<string, string> = {
    claude:     "var(--claude)",
    gpt:        "var(--gpt)",
    judge:      "var(--judge)",
    supervisor: "var(--supervisor)",
    researcher: "var(--researcher)",
    writer:     "var(--writer)",
    critic:     "var(--critic)",
    revisor:    "var(--writer)",
    system:     "var(--t4)",
  };
  return map[role] ?? "var(--t3)";
}

export function agentColorDark(role: string): string {
  const map: Record<string, string> = {
    claude:     "#92600A",
    gpt:        "#065F46",
    judge:      "#5B21B6",
    supervisor: "#92600A",
    researcher: "#1E3A8A",
    writer:     "#065F46",
    critic:     "#7F1D1D",
    revisor:    "#065F46",
    system:     "#27272A",
  };
  return map[role] ?? "#27272A";
}

export function agentIcon(role: string): string {
  const icons: Record<string, string> = {
    supervisor: "S", researcher: "R", writer: "W",
    critic: "C", revisor: "V", judge: "J",
    claude: "C", gpt: "G", system: "⚙",
  };
  return icons[role] ?? "?";
}
