"use client";

import { useEffect, useRef } from "react";
import { TraceEvent } from "@/lib/agents/types";
import { TraceEventRow } from "./TraceEvent";

export function ExecutionTrace({ events }: { events: TraceEvent[] }) {
  const bottomRef = useRef<HTMLDivElement>(null);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [events.length]);

  if (!events.length) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 160,
        fontSize: 12, color: "var(--t4)", fontFamily: "var(--font-mono)" }}>
        trace will appear here
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6, paddingRight: 4 }}>
      {events.map((e, i) => <TraceEventRow key={e.id} event={e} index={i} />)}
      <div ref={bottomRef} />
    </div>
  );
}
