"use client";

import { useState } from "react";
import { Copy, Check } from "lucide-react";

export function FinalOutput({ output }: { output: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => { await navigator.clipboard.writeText(output); setCopied(true); setTimeout(() => setCopied(false), 2000); };
  if (!output) return null;

  return (
    <div style={{
      background: "var(--glass)", backdropFilter: "blur(24px)", WebkitBackdropFilter: "blur(24px)",
      border: "1px solid var(--b1)", borderRadius: 16, padding: 24,
      boxShadow: "0 0 40px rgba(16,185,129,0.08), 0 8px 32px rgba(0,0,0,0.2)",
      animation: "fadeUp 0.5s ease both",
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: "linear-gradient(135deg, var(--gpt), #065F46)",
            boxShadow: "0 4px 12px rgba(16,185,129,0.30)",
            display: "flex", alignItems: "center", justifyContent: "center", fontSize: 15,
          }}>📄</div>
          <div>
            <p style={{ color: "var(--t0)", fontWeight: 700, fontSize: 14, fontFamily: "var(--font-display)", letterSpacing: "-0.03em" }}>Final Output</p>
            <p style={{ fontSize: 9, fontFamily: "var(--font-mono)", color: "var(--t4)", textTransform: "uppercase", letterSpacing: "0.14em" }}>pipeline result</p>
          </div>
        </div>
        <button
          onClick={copy}
          style={{
            display: "flex", alignItems: "center", gap: 6,
            background: "var(--s2)", border: "1px solid var(--b1)",
            borderRadius: 8, padding: "6px 12px", cursor: "pointer",
            fontSize: 11, fontFamily: "var(--font-mono)", color: copied ? "var(--gpt)" : "var(--t3)",
            transition: "all 0.2s ease",
          }}
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
          {copied ? "copied" : "copy"}
        </button>
      </div>
      <div style={{
        background: "var(--s1)", border: "1px solid var(--b1)", borderRadius: 10,
        padding: "16px 20px", fontSize: 13, color: "var(--t1)", lineHeight: 1.72,
        fontFamily: "var(--font-display)", whiteSpace: "pre-wrap", maxHeight: 500, overflowY: "auto",
      }}>
        {output}
      </div>
    </div>
  );
}
