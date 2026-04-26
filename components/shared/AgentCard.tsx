"use client";

import { AgentStatus, Provider } from "@/lib/agents/types";
import { agentColor, agentColorDark, agentIcon } from "@/lib/utils/formatting";

interface AgentCardProps {
  label: string;
  role: string;
  provider?: Provider;
  status: AgentStatus;
  toolCount?: number;
}

export function AgentCard({ label, role, provider, status, toolCount = 0 }: AgentCardProps) {
  const isActive = status === "thinking" || status === "tool_use";
  const isDone   = status === "done";
  const color    = agentColor(role);
  const colorDark = agentColorDark(role);
  const initial  = agentIcon(role);

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: 10,
      padding: "14px 16px",
      borderRadius: 14,
      background: isActive ? `${color}08` : "var(--glass)",
      backdropFilter: "blur(20px)",
      WebkitBackdropFilter: "blur(20px)",
      border: `1px solid ${isActive ? `${color}25` : "var(--b1)"}`,
      boxShadow: isActive ? `0 0 24px ${color}20, inset 0 0 20px ${color}05` : "none",
      transition: "all 0.3s cubic-bezier(.4,0,.2,1)",
      minWidth: 100,
      position: "relative",
    }}>
      {/* Running indicator dot top-right */}
      {isActive && (
        <div style={{
          position: "absolute",
          top: 10, right: 10,
          width: 8, height: 8,
          borderRadius: "50%",
          background: color,
          boxShadow: `0 0 8px ${color}60`,
          animation: "pulseDot 1.5s ease infinite",
        }} />
      )}
      {isDone && (
        <div style={{
          position: "absolute",
          top: 10, right: 10,
          width: 8, height: 8,
          borderRadius: "50%",
          background: color,
          boxShadow: `0 0 8px ${color}60`,
        }} />
      )}

      {/* Gradient icon square */}
      <div style={{
        width: 36, height: 36,
        borderRadius: 10,
        background: `linear-gradient(135deg, ${color}, ${colorDark})`,
        display: "flex", alignItems: "center", justifyContent: "center",
        boxShadow: `0 4px 12px ${color}30`,
      }}>
        <span style={{ color: "#fff", fontWeight: 700, fontSize: 13, fontFamily: "var(--font-display)" }}>
          {initial}
        </span>
      </div>

      <div style={{ textAlign: "center" }}>
        <p style={{ color: "var(--t1)", fontWeight: 600, fontSize: 11, fontFamily: "var(--font-display)", letterSpacing: "-0.01em" }}>
          {label}
        </p>
        <p style={{ color: "var(--t4)", fontSize: 9, fontFamily: "var(--font-mono)", textTransform: "uppercase", letterSpacing: "0.14em", marginTop: 2 }}>
          {role}
        </p>
      </div>

      {/* Status */}
      <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
        <div style={{
          width: 5, height: 5, borderRadius: "50%",
          background: isActive ? color : isDone ? "var(--gpt)" : "var(--t4)",
          boxShadow: isActive ? `0 0 8px ${color}60` : isDone ? "0 0 8px var(--gpt)60" : "none",
          animation: isActive ? "pulseDot 1.5s ease infinite" : "none",
        }} />
        <span style={{ fontSize: 9, fontFamily: "var(--font-mono)", color: "var(--t4)", textTransform: "uppercase", letterSpacing: "0.14em" }}>
          {isActive ? "running" : isDone ? "done" : "idle"}
        </span>
      </div>

      {/* Tool count chip */}
      {toolCount > 0 && (
        <div style={{
          fontSize: 9, fontFamily: "var(--font-mono)",
          color: "var(--tool)", background: "rgba(129,140,248,0.15)",
          padding: "3px 10px", borderRadius: 20,
        }}>
          {toolCount} tools
        </div>
      )}

      {/* Provider chip */}
      {provider && (
        <div style={{
          fontSize: 9, fontFamily: "var(--font-mono)", textTransform: "uppercase", letterSpacing: "0.14em",
          color: agentColor(provider),
          background: `${agentColor(provider)}15`,
          padding: "3px 10px", borderRadius: 20,
        }}>
          {provider === "claude" ? "Claude" : "GPT-4o"}
        </div>
      )}
    </div>
  );
}
