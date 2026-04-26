"use client";

import { ReactNode } from "react";

interface Tab { id: string; label: ReactNode; }
interface TabsProps { tabs: Tab[]; active: string; onChange: (id: string) => void; }

export function Tabs({ tabs, active, onChange }: TabsProps) {
  return (
    <div style={{
      display: "flex",
      gap: 2,
      background: "var(--s1)",
      border: "1px solid var(--b1)",
      borderRadius: 10,
      padding: 3,
    }}>
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          style={{
            flex: 1,
            padding: "7px 16px",
            borderRadius: 8,
            fontSize: 12,
            fontWeight: 600,
            fontFamily: "var(--font-display)",
            letterSpacing: "-0.01em",
            border: "none",
            cursor: "pointer",
            transition: "all 0.2s cubic-bezier(.4,0,.2,1)",
            ...(active === tab.id
              ? { background: "var(--glass)", boxShadow: "0 2px 8px rgba(0,0,0,0.20)", color: "var(--t0)" }
              : { background: "transparent", color: "var(--t3)" }),
          }}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
