"use client";

import { usePlatformStore } from "@/lib/store/platform";
import { Header } from "@/components/layout/Header";
import { BattlePanel } from "@/components/battle/BattlePanel";
import { OrchestrationPanel } from "@/components/orchestration/OrchestrationPanel";

export default function Home() {
  const mode = usePlatformStore((s) => s.mode);

  return (
    <div className="h-screen flex flex-col bg-slate-950 overflow-hidden">
      <Header />
      {mode === "battle" ? <BattlePanel /> : <OrchestrationPanel />}
    </div>
  );
}
