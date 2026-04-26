"use client";

import { useEffect, useRef, useState } from "react";
import { usePlatformStore } from "@/lib/store/platform";
import { PipelineConfig } from "./PipelineConfig";
import { FinalOutput } from "./FinalOutput";
import { OutputFeed } from "@/components/shared/OutputFeed";
import { ExecutionTrace } from "@/components/shared/ExecutionTrace";
import { StatusBar } from "@/components/shared/StatusBar";
import { DAGView } from "@/components/shared/DAGView";
import { Tabs } from "@/components/ui/Tabs";
import { createSSEStream } from "@/lib/utils/sse";
import { AgentStatus, Provider } from "@/lib/agents/types";
import { getPattern } from "@/lib/orchestration/patterns";
import { oppositeProvider } from "@/lib/agents/dispatcher";
import { agentColor } from "@/lib/utils/formatting";

function buildDAGNodes(pattern: string, provider: Provider, statuses: Record<string, AgentStatus>, toolCounts: Record<string, number>) {
  const alt = oppositeProvider(provider);
  const pL  = provider === "claude" ? "Claude" : "GPT-4o";
  const aL  = alt      === "claude" ? "Claude" : "GPT-4o";
  const defs: Record<string, { id: string; label: string; role: string; provider: Provider }[]> = {
    "research-report":  [
      { id: "supervisor",   label: "Supervisor",         role: "supervisor",  provider },
      { id: "researcher",   label: "Researcher",         role: "researcher",  provider },
      { id: "writer",       label: "Writer",             role: "writer",      provider },
      { id: "critic",       label: "Critic",             role: "critic",      provider },
    ],
    "parallel-research": [
      { id: "researcher-a", label: `Researcher A (${pL})`, role: "researcher", provider },
      { id: "researcher-b", label: `Researcher B (${aL})`, role: "researcher", provider: alt },
      { id: "writer",       label: "Writer",               role: "writer",     provider },
      { id: "critic",       label: "Critic",               role: "critic",     provider: alt },
    ],
    adversarial: [
      { id: "writer",  label: "Writer",          role: "writer",  provider },
      { id: "critic",  label: `Critic (${aL})`,  role: "critic",  provider: alt },
      { id: "revisor", label: "Revisor",          role: "revisor", provider },
    ],
  };
  return (defs[pattern] ?? []).map((n) => ({ ...n, status: statuses[n.id] ?? ("idle" as AgentStatus), toolCount: toolCounts[n.id] }));
}

const sidebarW = 300;

export function OrchestrationPanel() {
  const store = usePlatformStore();
  const timerRef     = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);
  const cancelRef    = useRef<(() => void) | null>(null);
  const [localStatuses, setLocalStatuses] = useState<Record<string, AgentStatus>>({});

  const dagNodes = buildDAGNodes(store.orchPattern, store.orchProvider, localStatuses, store.toolCounts);

  const start = () => {
    store.reset(); setLocalStatuses({});
    store.setStarted(true); store.setRunning(true);
    startTimeRef.current = Date.now();
    timerRef.current = setInterval(() => store.setElapsed(Date.now() - startTimeRef.current), 500);

    cancelRef.current = createSSEStream(
      "/api/orchestrate",
      { task: store.orchTask, pattern: store.orchPattern, provider: store.orchProvider },
      (msg) => {
        if      (msg.type === "event")         store.addEvent(msg.payload as never);
        else if (msg.type === "bubble")        store.addBubble(msg.payload as never);
        else if (msg.type === "agent_status")  setLocalStatuses((p) => ({ ...p, [msg.agent as string]: msg.status as AgentStatus }));
        else if (msg.type === "tool_count")    store.setToolCount(msg.agent as string, msg.count as number);
        else if (msg.type === "final_output")  store.setFinalOutput(msg.output as string);
        else if (msg.type === "done")        { store.setRunning(false); clearInterval(timerRef.current!); }
        else if (msg.type === "error")       { store.setError(msg.message as string); store.setRunning(false); clearInterval(timerRef.current!); }
      },
      () => { store.setRunning(false); clearInterval(timerRef.current!); },
      (err) => { store.setError(err); store.setRunning(false); clearInterval(timerRef.current!); }
    );
  };

  const reset = () => { cancelRef.current?.(); clearInterval(timerRef.current!); store.reset(); setLocalStatuses({}); };
  useEffect(() => () => { clearInterval(timerRef.current!); cancelRef.current?.(); }, []);

  const pattern = getPattern(store.orchPattern);

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
          <p style={{ color: "var(--t0)", fontWeight: 700, fontSize: 15, fontFamily: "var(--font-display)", letterSpacing: "-0.03em" }}>Orchestration</p>
          <p style={{ fontSize: 9, fontFamily: "var(--font-mono)", color: "var(--t4)", textTransform: "uppercase", letterSpacing: "0.14em", marginTop: 2 }}>Multi-agent pipelines</p>
        </div>
        <PipelineConfig onStart={start} onReset={reset} disabled={store.running} started={store.started} />
      </aside>

      {/* Main */}
      <main style={{ flex: 1, overflowY: "auto", background: "transparent" }}>
        {!store.started ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", textAlign: "center", padding: "0 32px" }}>
            <div style={{
              width: 72, height: 72, borderRadius: 20,
              background: "var(--glass)", backdropFilter: "blur(20px)", WebkitBackdropFilter: "blur(20px)",
              border: "1px solid var(--b1)", display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 32, marginBottom: 20, boxShadow: "0 8px 32px rgba(0,0,0,0.2)",
            }}>🔬</div>
            <p style={{ color: "var(--t0)", fontWeight: 700, fontSize: 18, fontFamily: "var(--font-display)", letterSpacing: "-0.03em", marginBottom: 8 }}>
              Ready to run
            </p>
            <p style={{ color: "var(--t4)", fontSize: 13, fontFamily: "var(--font-display)", lineHeight: 1.7, maxWidth: 300 }}>
              Enter a research task, pick a pattern, and hit Run Pipeline.
            </p>
          </div>
        ) : (
          <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 16 }}>
            {/* DAG + pattern badge */}
            <div style={{
              background: "var(--glass)", backdropFilter: "blur(20px)", WebkitBackdropFilter: "blur(20px)",
              border: "1px solid var(--b1)", borderRadius: 14, padding: 20,
            }}>
              {pattern && (
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
                  <span style={{ fontSize: 16 }}>{pattern.icon}</span>
                  <span style={{ fontSize: 12, fontWeight: 600, color: "var(--t1)", fontFamily: "var(--font-display)" }}>{pattern.name}</span>
                  {pattern.crossModel && (
                    <span style={{
                      fontSize: 8, fontFamily: "var(--font-mono)", textTransform: "uppercase", letterSpacing: "0.14em",
                      color: "var(--judge)", background: "rgba(167,139,250,0.15)", padding: "2px 8px", borderRadius: 20,
                    }}>cross-model</span>
                  )}
                </div>
              )}
              <DAGView nodes={dagNodes} mode="orchestrate" />
            </div>

            <StatusBar running={store.running} started={store.started} elapsed={store.elapsed} eventCount={store.events.length} error={store.error} />

            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <Tabs
                tabs={[{ id: "output", label: "Pipeline Output" }, { id: "trace", label: `Trace (${store.events.length})` }]}
                active={store.activeTab}
                onChange={(t) => store.setActiveTab(t as "output" | "trace")}
              />
              <div style={{
                background: "var(--glass)", backdropFilter: "blur(20px)", WebkitBackdropFilter: "blur(20px)",
                border: "1px solid var(--b1)", borderRadius: 14, padding: 20, minHeight: 300,
              }}>
                {store.activeTab === "output"
                  ? <OutputFeed bubbles={store.bubbles} mode="orchestrate" />
                  : <ExecutionTrace events={store.events} />}
              </div>
            </div>

            {store.finalOutput && <FinalOutput output={store.finalOutput} />}
          </div>
        )}
      </main>
    </div>
  );
}
