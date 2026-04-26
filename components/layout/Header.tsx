"use client";

import { usePlatformStore } from "@/lib/store/platform";
import { Swords, GitBranch } from "lucide-react";
import { cn } from "@/lib/utils/cn";

export function Header() {
  const { mode, setMode } = usePlatformStore();

  return (
    <header
      className="glass-intense sticky top-0 z-50 h-14"
      style={{ borderRadius: 0, borderLeft: 0, borderRight: 0, borderTop: 0 }}
    >
      <div className="h-full flex items-center justify-between px-5">

        {/* Logo + Brand */}
        <div className="flex items-center gap-3">
          <div
            style={{
              width: 32, height: 32,
              borderRadius: 10,
              background: "linear-gradient(135deg, #F59E0B, #EF4444)",
              boxShadow: "0 4px 16px rgba(245,158,11,0.20)",
              display: "flex", alignItems: "center", justifyContent: "center",
              flexShrink: 0,
            }}
          >
            <span style={{ color: "#fff", fontWeight: 900, fontSize: 15, fontFamily: "var(--font-display)" }}>S</span>
          </div>
          <div>
            <span style={{ color: "var(--t0)", fontWeight: 700, fontSize: 13, letterSpacing: "-0.03em", fontFamily: "var(--font-display)" }}>
              Shekhar&apos;s Research Lab
            </span>
            <span
              style={{
                marginLeft: 8,
                fontSize: 9,
                fontFamily: "var(--font-mono)",
                color: "var(--t4)",
                letterSpacing: "0.14em",
                textTransform: "uppercase",
              }}
            >
              v3
            </span>
          </div>
        </div>

        {/* Mode Switcher — segmented control */}
        <div
          style={{
            display: "flex",
            gap: 2,
            background: "var(--s1)",
            border: "1px solid var(--b1)",
            borderRadius: 10,
            padding: 3,
          }}
        >
          {([
            { id: "battle",      label: "Battle Arena",    Icon: Swords     },
            { id: "orchestrate", label: "Orchestration",   Icon: GitBranch  },
          ] as const).map(({ id, label, Icon }) => (
            <button
              key={id}
              onClick={() => setMode(id)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                padding: "6px 16px",
                borderRadius: 8,
                fontSize: 12,
                fontWeight: 600,
                fontFamily: "var(--font-display)",
                letterSpacing: "-0.01em",
                transition: "all 0.2s cubic-bezier(.4,0,.2,1)",
                border: "none",
                cursor: "pointer",
                ...(mode === id
                  ? {
                      background: "var(--glass)",
                      boxShadow: "0 2px 8px rgba(0,0,0,0.20)",
                      color: "var(--t0)",
                    }
                  : {
                      background: "transparent",
                      color: "var(--t3)",
                    }),
              }}
            >
              <Icon size={13} />
              {label}
            </button>
          ))}
        </div>

        {/* Provider status pills */}
        <div className="flex items-center gap-4">
          {[
            { label: "Claude", color: "var(--claude)" },
            { label: "GPT-4o", color: "var(--gpt)"   },
          ].map(({ label, color }) => (
            <div key={label} className="flex items-center gap-1.5">
              <div style={{
                width: 6, height: 6, borderRadius: "50%",
                background: color,
                boxShadow: `0 0 8px ${color}60`,
              }} />
              <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--t3)" }}>
                {label}
              </span>
            </div>
          ))}
        </div>
      </div>
    </header>
  );
}
