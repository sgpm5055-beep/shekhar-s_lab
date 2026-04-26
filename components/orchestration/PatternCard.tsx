"use client";

import { Pattern } from "@/lib/orchestration/patterns";

const patternColors: Record<string, string> = {
  blue:   "var(--researcher)",
  purple: "var(--judge)",
  red:    "var(--critic)",
};

interface PatternCardProps { pattern: Pattern; selected: boolean; onClick: () => void; disabled?: boolean; }

export function PatternCard({ pattern, selected, onClick, disabled }: PatternCardProps) {
  const color = patternColors[pattern.color] ?? "var(--t3)";

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        width: "100%", textAlign: "left",
        background: selected ? `${color}08` : "var(--s1)",
        backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)",
        border: `1px solid ${selected ? `${color}25` : "var(--b1)"}`,
        borderRadius: 10, padding: "12px 14px",
        cursor: disabled ? "not-allowed" : "pointer",
        transition: "all 0.2s cubic-bezier(.4,0,.2,1)",
        boxShadow: selected ? `0 0 16px ${color}15` : "none",
        opacity: disabled ? 0.55 : 1,
      }}
    >
      <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
        <span style={{ fontSize: 18, flexShrink: 0, marginTop: 1 }}>{pattern.icon}</span>
        <div style={{ minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 4 }}>
            <span style={{
              fontSize: 12, fontWeight: 600, fontFamily: "var(--font-display)",
              color: selected ? "var(--t0)" : "var(--t1)", letterSpacing: "-0.01em",
            }}>
              {pattern.name}
            </span>
            {pattern.crossModel && (
              <span style={{
                fontSize: 8, fontFamily: "var(--font-mono)", textTransform: "uppercase", letterSpacing: "0.14em",
                color: "var(--judge)", background: "rgba(167,139,250,0.15)", padding: "2px 8px", borderRadius: 20,
              }}>
                cross-model
              </span>
            )}
          </div>
          <p style={{ fontSize: 11, color: "var(--t3)", fontFamily: "var(--font-display)", lineHeight: 1.6 }}>
            {pattern.description}
          </p>
        </div>
      </div>
    </button>
  );
}
