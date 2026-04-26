"use client";

import { formatElapsed } from "@/lib/utils/formatting";

interface StatusBarProps { running: boolean; started: boolean; elapsed: number; eventCount: number; error: string | null; }

export function StatusBar({ running, started, elapsed, eventCount, error }: StatusBarProps) {
  if (error) {
    return (
      <div style={{
        display: "flex", alignItems: "center", gap: 10,
        padding: "10px 16px", borderRadius: 10,
        background: "rgba(239,68,68,0.08)",
        border: "1px solid rgba(239,68,68,0.20)",
      }}>
        <div style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--critic)", boxShadow: "0 0 8px rgba(239,68,68,0.6)", flexShrink: 0 }} />
        <span style={{ fontSize: 11, fontWeight: 600, color: "var(--critic)", fontFamily: "var(--font-display)" }}>Error:</span>
        <span style={{ fontSize: 11, color: "rgba(239,68,68,0.8)", fontFamily: "var(--font-mono)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{error}</span>
      </div>
    );
  }
  if (!started) return null;

  const color = running ? "var(--claude)" : "var(--gpt)";

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 16,
      padding: "10px 16px", borderRadius: 10,
      background: "var(--s1)", border: "1px solid var(--b1)",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{
          width: 6, height: 6, borderRadius: "50%", background: color,
          boxShadow: `0 0 8px ${color}60`,
          animation: running ? "pulseDot 1.5s ease infinite" : "none",
        }} />
        <span style={{ fontSize: 12, fontWeight: 600, color: "var(--t2)", fontFamily: "var(--font-display)" }}>
          {running ? "Running…" : "Complete"}
        </span>
      </div>
      <div style={{ width: 1, height: 12, background: "var(--b1)" }} />
      <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--t4)" }}>{formatElapsed(elapsed)}</span>
      <div style={{ width: 1, height: 12, background: "var(--b1)" }} />
      <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--t4)" }}>{eventCount} events</span>
    </div>
  );
}
