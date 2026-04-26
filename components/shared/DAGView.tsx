"use client";

import { AgentCard } from "./AgentCard";
import { AgentStatus, Provider } from "@/lib/agents/types";

interface AgentNode {
  id: string;
  label: string;
  role: string;
  provider?: Provider;
  status: AgentStatus;
  toolCount?: number;
}

export function DAGView({ nodes, mode }: { nodes: AgentNode[]; mode: "battle" | "orchestrate" }) {
  if (!nodes.length) return null;

  return (
    <div>
      <p style={{
        fontSize: 9, fontFamily: "var(--font-mono)", color: "var(--t4)",
        textTransform: "uppercase", letterSpacing: "0.18em", marginBottom: 12,
      }}>
        {mode === "battle" ? "Debate Agents" : "Pipeline DAG"}
      </p>
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", justifyContent: "center" }}>
        {nodes.map((node, i) => (
          <div key={node.id} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <AgentCard
              label={node.label}
              role={node.role}
              provider={node.provider}
              status={node.status}
              toolCount={node.toolCount}
            />
            {i < nodes.length - 1 && (
              <span style={{ color: "var(--t4)", fontSize: 16, fontWeight: 200, userSelect: "none" }}>→</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
