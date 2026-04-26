"use client";

import { useEffect, useRef, useState } from "react";
import { usePlatformStore } from "@/lib/store/platform";
import { BattleConfig } from "./BattleConfig";
import { Verdict } from "./Verdict";
import { OutputFeed } from "@/components/shared/OutputFeed";
import { ExecutionTrace } from "@/components/shared/ExecutionTrace";
import { StatusBar } from "@/components/shared/StatusBar";
import { DAGView } from "@/components/shared/DAGView";
import { Tabs } from "@/components/ui/Tabs";
import { createSSEStream } from "@/lib/utils/sse";
import { AgentStatus } from "@/lib/agents/types";

const sidebarW = 280;

export function BattlePanel() {
  const store = usePlatformStore();
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);
  const cancelRef = useRef<(() => void) | null>(null);
  const [localStatuses, setLocalStatuses] = useState<Record<string, AgentStatus>>({ claude: "idle", gpt: "idle", judge: "idle" });

  const dagNodes = [
    { id: "claude", label: "Claude",  role: "claude", provider: "claude" as const, status: localStatuses["claude"] ?? "idle", toolCount: store.toolCounts["claude"] },
    { id: "gpt",    label: "GPT-4o",  role: "gpt",    provider: "gpt"    as const, status: localStatuses["gpt"]    ?? "idle", toolCount: store.toolCounts["gpt"]    },
    { id: "judge",  label: "Judge",   role: "judge",  provider: store.judgeModel,  status: localStatuses["judge"]  ?? "idle" },
  ];

  const start = () => {
    store.reset();
    setLocalStatuses({ claude: "idle", gpt: "idle", judge: "idle" });
    store.setStarted(true); store.setRunning(true);
    startTimeRef.current = Date.now();
    timerRef.current = setInterval(() => store.setElapsed(Date.now() - startTimeRef.current), 500);

    cancelRef.current = createSSEStream(
      "/api/battle",
      { topic: store.topic, rounds: store.rounds, judgeModel: store.judgeModel },
      (msg) => {
        if      (msg.type === "event")        store.addEvent(msg.payload as never);
        else if (msg.type === "bubble")       store.addBubble(msg.payload as never);
        else if (msg.type === "agent_status") setLocalStatuses((p) => ({ ...p, [msg.agent as string]: msg.status as AgentStatus }));
        else if (msg.type === "tool_count")   store.setToolCount(msg.agent as string, msg.count as number);
        else if (msg.type === "scores")     { store.setScores(msg.payload as never); setLocalStatuses((p) => ({ ...p, judge: "done" })); }
        else if (msg.type === "done")       { store.setRunning(false); clearInterval(timerRef.current!); }
        else if (msg.type === "error")      { store.setError(msg.message as string); store.setRunning(false); clearInterval(timerRef.current!); }
      },
      () => { store.setRunning(false); clearInterval(timerRef.current!); },
      (err) => { store.setError(err); store.setRunning(false); clearInterval(timerRef.current!); }
    );
  };

  const reset = () => { cancelRef.current?.(); clearInterval(timerRef.current!); store.reset(); setLocalStatuses({ claude: "idle", gpt: "idle", judge: "idle" }); };
  useEffect(() => () => { clearInterval(timerRef.current!); cancelRef.current?.(); }, []);

  return (
    <div style={{ display: "flex", height: "calc(100vh - 56px)", overflow: "hidden" }}>
      {/* Sidebar */}
      <aside style={{
        width: sidebarW, flexShrink: 0,
        background: "var(--glass)", backdropFilter: "blur(24px)", WebkitBackdropFilter: "blur(24px)",
        borderRight: "1px solid var(--b1)",
        display: "flex", flexDirection: "column", padding: 20,
      }}>
        <div style={{ marginBottom: 20 }}>
          <p style={{ color: "var(--t0)", fontWeight: 700, fontSize: 15, fontFamily: "var(--font-display)", letterSpacing: "-0.03em" }}>Battle Arena</p>
          <p style={{ fontSize: 9, fontFamily: "var(--font-mono)", color: "var(--t4)", textTransform: "uppercase", letterSpacing: "0.14em", marginTop: 2 }}>Claude vs GPT-4o</p>
        </div>
        <BattleConfig onStart={start} onReset={reset} disabled={store.running} started={store.started} />
      </aside>

      {/* Main */}
      <main style={{ flex: 1, overflowY: "auto", background: "transparent" }}>
        {!store.started ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", textAlign: "center", padding: "0 32px" }}>
            <div style={{
              width: 72, height: 72, borderRadius: 20,
              background: "var(--glass)", backdropFilter: "blur(20px)", WebkitBackdropFilter: "blur(20px)",
              border: "1px solid var(--b1)", display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 32, marginBottom: 20,
              boxShadow: "0 8px 32px rgba(0,0,0,0.2)",
            }}>⚔️</div>
            <p style={{ color: "var(--t0)", fontWeight: 700, fontSize: 18, fontFamily: "var(--font-display)", letterSpacing: "-0.03em", marginBottom: 8 }}>
              Ready to battle
            </p>
            <p style={{ color: "var(--t4)", fontSize: 13, fontFamily: "var(--font-display)", lineHeight: 1.7, maxWidth: 300 }}>
              Set your topic, rounds, and judge in the sidebar, then hit Start Battle.
            </p>
          </div>
        ) : (
          <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 16 }}>
            {/* DAG */}
            <div style={{
              background: "var(--glass)", backdropFilter: "blur(20px)", WebkitBackdropFilter: "blur(20px)",
              border: "1px solid var(--b1)", borderRadius: 14, padding: 20,
            }}>
              <DAGView nodes={dagNodes} mode="battle" />
            </div>

            <StatusBar running={store.running} started={store.started} elapsed={store.elapsed} eventCount={store.events.length} error={store.error} />

            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <Tabs
                tabs={[{ id: "output", label: "Output" }, { id: "trace", label: `Trace (${store.events.length})` }]}
                active={store.activeTab}
                onChange={(t) => store.setActiveTab(t as "output" | "trace")}
              />
              <div style={{
                background: "var(--glass)", backdropFilter: "blur(20px)", WebkitBackdropFilter: "blur(20px)",
                border: "1px solid var(--b1)", borderRadius: 14, padding: 20, minHeight: 300,
              }}>
                {store.activeTab === "output"
                  ? <OutputFeed bubbles={store.bubbles} mode="battle" />
                  : <ExecutionTrace events={store.events} />}
              </div>
            </div>

            {store.scores && <Verdict scores={store.scores} judgeModel={store.judgeModel} />}
          </div>
        )}
      </main>
    </div>
  );
}
