"use client";

import { JudgeScores } from "@/lib/agents/types";
import { ScoreBar } from "./ScoreBar";

interface VerdictProps { scores: JudgeScores; judgeModel: "claude" | "gpt"; }

export function Verdict({ scores, judgeModel }: VerdictProps) {
  const winnerColor = scores.winner === "claude" ? "var(--claude)" : scores.winner === "gpt" ? "var(--gpt)" : "var(--judge)";
  const winnerLabel = scores.winner === "claude" ? "Claude Wins" : scores.winner === "gpt" ? "GPT-4o Wins" : "It's a Tie";

  return (
    <div style={{
      background: "var(--glass)", backdropFilter: "blur(24px)", WebkitBackdropFilter: "blur(24px)",
      border: "1px solid var(--b1)", borderRadius: 16, padding: 24,
      display: "flex", flexDirection: "column", gap: 20,
      boxShadow: `0 0 40px ${winnerColor}10, 0 8px 32px rgba(0,0,0,0.2)`,
      animation: "fadeUp 0.5s ease both",
    }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{
          width: 32, height: 32, borderRadius: 8,
          background: "linear-gradient(135deg, #F59E0B, #EF4444)",
          boxShadow: "0 4px 12px rgba(245,158,11,0.30)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 16,
        }}>🏆</div>
        <div>
          <p style={{ color: "var(--t0)", fontWeight: 700, fontSize: 14, fontFamily: "var(--font-display)", letterSpacing: "-0.03em" }}>Verdict</p>
          <p style={{ fontSize: 9, fontFamily: "var(--font-mono)", color: "var(--t4)", textTransform: "uppercase", letterSpacing: "0.14em" }}>
            judged by {judgeModel === "claude" ? "Claude" : "GPT-4o"}
          </p>
        </div>
      </div>

      {/* Winner */}
      <div style={{
        textAlign: "center", padding: "20px",
        background: `${winnerColor}08`, border: `1px solid ${winnerColor}20`,
        borderRadius: 12,
      }}>
        <p style={{ fontSize: 24, fontWeight: 700, color: winnerColor, fontFamily: "var(--font-display)", letterSpacing: "-0.03em" }}>{winnerLabel}</p>
        <div style={{ display: "flex", justifyContent: "center", gap: 60, marginTop: 16 }}>
          {[
            { label: "Claude", score: scores.claudeScore, color: "var(--claude)" },
            { label: "GPT-4o", score: scores.gptScore,   color: "var(--gpt)"    },
          ].map(({ label, score, color }) => (
            <div key={label} style={{ textAlign: "center" }}>
              <p style={{ fontSize: 32, fontWeight: 700, color, fontFamily: "var(--font-display)", letterSpacing: "-0.03em" }}>{score}</p>
              <p style={{ fontSize: 9, fontFamily: "var(--font-mono)", color: "var(--t4)", textTransform: "uppercase", letterSpacing: "0.14em", marginTop: 4 }}>{label}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Reasoning */}
      <div style={{ background: "var(--s1)", border: "1px solid var(--b1)", borderRadius: 10, padding: "14px 16px" }}>
        <p style={{ fontSize: 13, color: "var(--t1)", lineHeight: 1.72, fontFamily: "var(--font-display)" }}>{scores.reasoning}</p>
      </div>

      {/* Criteria */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        {[
          { label: "Claude", color: "var(--claude)", crit: scores.criteria, key: "claude" as const },
          { label: "GPT-4o", color: "var(--gpt)",   crit: scores.criteria, key: "gpt"    as const },
        ].map(({ label, color, crit, key }) => (
          <div key={label}>
            <p style={{ fontSize: 9, fontFamily: "var(--font-mono)", color, textTransform: "uppercase", letterSpacing: "0.14em", marginBottom: 10 }}>{label}</p>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <ScoreBar label="Accuracy"   score={crit.accuracy[key]}   color={color} />
              <ScoreBar label="Depth"      score={crit.depth[key]}      color={color} />
              <ScoreBar label="Clarity"    score={crit.clarity[key]}    color={color} />
              <ScoreBar label="Persuasion" score={crit.persuasion[key]} color={color} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
