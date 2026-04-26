"use client";

import { TraceEvent } from "@/lib/agents/types";
import { agentColor } from "@/lib/utils/formatting";

const typeColor: Record<string, string> = {
  thought:     "var(--t4)",
  tool_call:   "var(--tool)",
  tool_result: "var(--gpt)",
  handoff:     "var(--judge)",
  output:      "var(--gpt)",
  error:       "var(--critic)",
  system:      "var(--t4)",
};

export function TraceEventRow({ event, index = 0 }: { event: TraceEvent; index?: number }) {
  const color  = typeColor[event.type] ?? "var(--t4)";
  const aColor = agentColor(event.agent);

  const time = new Date(event.timestamp).toLocaleTimeString("en-US", {
    hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit",
  });

  return (
    <div style={{
      background: `linear-gradient(135deg, ${aColor}08, ${aColor}04)`,
      border: `1px solid ${aColor}15`,
      borderRadius: 10,
      padding: "10px 12px",
      animation: "slideIn 0.3s ease both",
      animationDelay: `${index * 0.015}s`,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 6 }}>
        {/* Left accent dot */}
        <div style={{ width: 5, height: 5, borderRadius: "50%", background: color, boxShadow: `0 0 8px ${color}60`, flexShrink: 0 }} />
        <span style={{ fontSize: 9, fontFamily: "var(--font-mono)", color: "var(--t4)", letterSpacing: "0.14em" }}>{time}</span>
        <span style={{ fontSize: 9, fontFamily: "var(--font-mono)", color: "var(--t3)", textTransform: "uppercase", letterSpacing: "0.14em" }}>
          {event.agent}
        </span>
        <span style={{
          fontSize: 8, fontFamily: "var(--font-mono)", textTransform: "uppercase", letterSpacing: "0.14em",
          color, background: `${color}15`, padding: "2px 8px", borderRadius: 20,
        }}>
          {event.type.replace("_", " ")}
        </span>
        {event.tool && (
          <span style={{
            fontSize: 8, fontFamily: "var(--font-mono)", textTransform: "uppercase", letterSpacing: "0.14em",
            color: "var(--tool)", background: "rgba(129,140,248,0.15)", padding: "2px 8px", borderRadius: 20,
          }}>
            {event.tool}
          </span>
        )}
        {event.provider && (
          <span style={{
            fontSize: 8, fontFamily: "var(--font-mono)", textTransform: "uppercase", letterSpacing: "0.14em",
            color: agentColor(event.provider), background: `${agentColor(event.provider)}15`, padding: "2px 8px", borderRadius: 20,
          }}>
            {event.provider}
          </span>
        )}
      </div>
      <p style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--t2)", lineHeight: 1.7, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
        {event.content.length > 500 ? event.content.slice(0, 500) + "…" : event.content}
      </p>
    </div>
  );
}
