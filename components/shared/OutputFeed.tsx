"use client";

import { useEffect, useRef } from "react";
import { OutputBubble } from "@/lib/agents/types";
import { Bubble } from "./Bubble";

export function OutputFeed({ bubbles, mode = "battle" }: { bubbles: OutputBubble[]; mode?: "battle" | "orchestrate" }) {
  const bottomRef = useRef<HTMLDivElement>(null);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [bubbles.length]);

  if (!bubbles.length) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 160,
        fontSize: 12, color: "var(--t4)", fontFamily: "var(--font-mono)" }}>
        {mode === "battle" ? "debate output will appear here" : "pipeline output will appear here"}
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, paddingRight: 4 }}>
      {bubbles.map((b, i) => <Bubble key={b.id} bubble={b} mode={mode} index={i} />)}
      <div ref={bottomRef} />
    </div>
  );
}
