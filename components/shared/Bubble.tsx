"use client";

import { OutputBubble } from "@/lib/agents/types";
import { agentColor, agentColorDark, agentIcon, stripMarkdown } from "@/lib/utils/formatting";

interface BubbleProps {
  bubble: OutputBubble;
  mode?: "battle" | "orchestrate";
  index?: number;
}

export function Bubble({ bubble, mode = "battle", index = 0 }: BubbleProps) {
  const role   = bubble.role ?? (bubble.provider === "claude" ? "claude" : "gpt");
  const color  = agentColor(role);
  const colorDark = agentColorDark(role);
  const initial = agentIcon(role);
  const isLeft = bubble.provider === "claude" || ["supervisor","researcher","claude"].includes(role);

  const radius = isLeft
    ? "4px 16px 16px 16px"   // left-aligned
    : "16px 4px 16px 16px";  // right-aligned

  return (
    <div style={{
      display: "flex",
      flexDirection: isLeft ? "row" : "row-reverse",
      gap: 10,
      animation: "fadeUp 0.4s ease both",
      animationDelay: index < 10 ? `${index * 0.04}s` : "0s",
    }}>
      {/* Avatar */}
      <div style={{
        width: 32, height: 32, borderRadius: 8, flexShrink: 0,
        background: `linear-gradient(135deg, ${color}, ${colorDark})`,
        boxShadow: `0 4px 12px ${color}30`,
        display: "flex", alignItems: "center", justifyContent: "center",
      }}>
        <span style={{ color: "#fff", fontWeight: 700, fontSize: 12, fontFamily: "var(--font-display)" }}>
          {initial}
        </span>
      </div>

      <div style={{ maxWidth: "80%", display: "flex", flexDirection: "column", gap: 4, alignItems: isLeft ? "flex-start" : "flex-end" }}>
        {/* Name row */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexDirection: isLeft ? "row" : "row-reverse" }}>
          <span style={{ fontSize: 11, fontWeight: 600, color: "var(--t2)", fontFamily: "var(--font-display)" }}>
            {bubble.agent}
          </span>
          {bubble.provider && (
            <span style={{
              fontSize: 9, fontFamily: "var(--font-mono)", textTransform: "uppercase", letterSpacing: "0.14em",
              color: agentColor(bubble.provider),
              background: `${agentColor(bubble.provider)}15`,
              padding: "2px 8px", borderRadius: 20,
            }}>
              {bubble.provider === "claude" ? "Claude" : "GPT-4o"}
            </span>
          )}
          {bubble.round !== undefined && mode === "battle" && (
            <span style={{
              fontSize: 9, fontFamily: "var(--font-mono)", textTransform: "uppercase", letterSpacing: "0.14em",
              color: "var(--t4)", background: "var(--s2)", padding: "2px 8px", borderRadius: 20,
            }}>
              R{bubble.round}
            </span>
          )}
        </div>

        {/* Bubble */}
        <div style={{
          background: "var(--glass)",
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
          border: `1px solid ${color}20`,
          borderRadius: radius,
          boxShadow: `0 4px 24px ${color}08`,
          padding: "12px 16px",
          fontSize: 13,
          lineHeight: 1.72,
          color: "var(--t1)",
          fontFamily: "var(--font-display)",
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        }}>
          {stripMarkdown(bubble.content)}
        </div>
      </div>
    </div>
  );
}
