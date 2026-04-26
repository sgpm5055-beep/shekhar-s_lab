"use client";

interface ScoreBarProps { label: string; score: number; max?: number; color: string; }

export function ScoreBar({ label, score, max = 10, color }: ScoreBarProps) {
  const pct = Math.round((score / max) * 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
      <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--t3)", width: 72, flexShrink: 0 }}>{label}</span>
      <div style={{ flex: 1, height: 6, background: "var(--s2)", borderRadius: 20, overflow: "hidden" }}>
        <div style={{
          height: "100%",
          width: `${pct}%`,
          background: `linear-gradient(90deg, ${color}, ${color}AA)`,
          borderRadius: 20,
          boxShadow: `0 0 12px ${color}40`,
          transition: "width 1s cubic-bezier(.4,0,.2,1)",
        }} />
      </div>
      <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", fontWeight: 600, color: "var(--t2)", width: 20, textAlign: "right" }}>{score}</span>
    </div>
  );
}
