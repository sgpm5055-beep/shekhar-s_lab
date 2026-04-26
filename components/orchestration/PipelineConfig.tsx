"use client";

import { usePlatformStore } from "@/lib/store/platform";
import { Button } from "@/components/ui/Button";
import { PatternCard } from "./PatternCard";
import { PATTERNS } from "@/lib/orchestration/patterns";
import { Play, RotateCcw } from "lucide-react";

const chipBase: React.CSSProperties = {
  flex: 1, padding: "9px 12px", borderRadius: 8, fontSize: 11,
  fontFamily: "var(--font-mono)", fontWeight: 600, cursor: "pointer",
  transition: "all 0.2s cubic-bezier(.4,0,.2,1)", letterSpacing: "0.05em",
};

const microLabel: React.CSSProperties = {
  fontSize: 9, fontFamily: "var(--font-mono)", color: "var(--t3)",
  textTransform: "uppercase" as const, letterSpacing: "0.18em", display: "block", marginBottom: 8,
};

interface PipelineConfigProps { onStart: () => void; onReset: () => void; disabled?: boolean; started?: boolean; }

export function PipelineConfig({ onStart, onReset, disabled, started }: PipelineConfigProps) {
  const { orchTask, setOrchTask, orchPattern, setOrchPattern, orchProvider, setOrchProvider } = usePlatformStore();

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 20, overflowY: "auto", paddingRight: 2 }}>

        {/* Task */}
        <div>
          <span style={microLabel}>Research Task</span>
          <textarea
            value={orchTask}
            onChange={(e) => setOrchTask(e.target.value)}
            rows={4}
            disabled={disabled}
            placeholder="Describe the research task or question…"
            style={{
              width: "100%", borderRadius: 10, resize: "none",
              background: "var(--s1)", border: "1px solid var(--b1)",
              color: disabled ? "var(--t4)" : "var(--t1)", fontSize: 13, lineHeight: 1.72,
              padding: "10px 14px", fontFamily: "var(--font-display)",
              outline: "none", transition: "border-color 0.2s ease",
              cursor: disabled ? "not-allowed" : "text",
              opacity: disabled ? 0.6 : 1,
            }}
            onFocus={(e) => { if (!disabled) e.target.style.borderColor = "rgba(255,255,255,0.16)"; }}
            onBlur={(e) => { e.target.style.borderColor = "var(--b1)"; }}
          />
        </div>

        {/* Provider */}
        <div>
          <span style={microLabel}>Primary Provider</span>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, opacity: disabled ? 0.5 : 1, pointerEvents: disabled ? "none" : "auto" }}>
            {([["claude", "var(--claude)"], ["gpt", "var(--gpt)"]] as const).map(([p, c]) => {
              const sel = orchProvider === p;
              return (
                <button key={p} onClick={() => setOrchProvider(p)} disabled={disabled} style={{
                  ...chipBase,
                  background: sel ? `${c}12` : "transparent",
                  border: `1px solid ${sel ? `${c}40` : "var(--b1)"}`,
                  color: sel ? c : "var(--t3)",
                  boxShadow: sel ? `0 0 12px ${c}12` : "none",
                  cursor: disabled ? "not-allowed" : "pointer",
                }}>
                  {p === "claude" ? "Claude" : "GPT-4o"}
                </button>
              );
            })}
          </div>
        </div>

        {/* Pattern */}
        <div>
          <span style={microLabel}>Pipeline Pattern</span>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {PATTERNS.map((p) => (
              <PatternCard
                key={p.id}
                pattern={p}
                selected={orchPattern === p.id}
                onClick={() => setOrchPattern(p.id)}
                disabled={disabled}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Actions */}
      <div style={{ paddingTop: 20, borderTop: "1px solid var(--b1)", marginTop: 20, display: "flex", flexDirection: "column", gap: 10 }}>
        <Button onClick={onStart} disabled={disabled || !orchTask.trim()} size="lg">
          <Play size={14} />
          {disabled ? "Running…" : "Run Pipeline"}
        </Button>
        {started && (
          <button onClick={onReset} style={{
            display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
            background: "none", border: "none", cursor: "pointer",
            fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--t4)",
            padding: "6px", transition: "color 0.2s",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.color = "var(--t2)")}
          onMouseLeave={(e) => (e.currentTarget.style.color = "var(--t4)")}
          >
            <RotateCcw size={11} /> reset
          </button>
        )}
      </div>
    </div>
  );
}
