"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/Button";
import { Key, Eye, EyeOff, AlertCircle } from "lucide-react";

const CLAUDE_KEY = "lab_claude_key";
const GPT_KEY = "lab_gpt_key";

interface ApiKeyGateProps {
  children: React.ReactNode;
  requireGPT?: boolean;
}

export function ApiKeyGate({ children, requireGPT = true }: ApiKeyGateProps) {
  const [claudeKey, setClaudeKey] = useState("");
  const [gptKey, setGptKey] = useState("");
  const [showClaude, setShowClaude] = useState(false);
  const [showGpt, setShowGpt] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    const c = localStorage.getItem(CLAUDE_KEY) ?? "";
    const g = localStorage.getItem(GPT_KEY) ?? "";
    setClaudeKey(c);
    setGptKey(g);
    setSaved(!!c && (!requireGPT || !!g));
  }, [requireGPT]);

  const save = () => {
    localStorage.setItem(CLAUDE_KEY, claudeKey);
    localStorage.setItem(GPT_KEY, gptKey);
    setSaved(true);
  };

  const clear = () => {
    localStorage.removeItem(CLAUDE_KEY);
    localStorage.removeItem(GPT_KEY);
    setClaudeKey("");
    setGptKey("");
    setSaved(false);
  };

  if (saved) {
    return <>{children}</>;
  }

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center p-4">
      <div className="w-full max-w-md space-y-6">
        {/* Logo */}
        <div className="text-center space-y-2">
          <div className="text-4xl">🧪</div>
          <h1 className="text-2xl font-bold text-slate-100">Shekhar&apos;s Research Lab</h1>
          <p className="text-slate-400 text-sm">Enter your API keys to get started</p>
        </div>

        <div className="rounded-xl border border-slate-700/50 bg-slate-900/80 backdrop-blur p-6 space-y-4">
          <div className="flex items-center gap-2 text-slate-300 text-sm font-medium">
            <Key className="w-4 h-4" />
            API Keys
          </div>

          {/* Anthropic Key */}
          <div className="space-y-1.5">
            <label className="text-xs text-slate-400">Anthropic API Key</label>
            <div className="relative">
              <input
                type={showClaude ? "text" : "password"}
                value={claudeKey}
                onChange={(e) => setClaudeKey(e.target.value)}
                placeholder="sk-ant-..."
                className="w-full rounded-lg bg-slate-800 border border-slate-700 px-3 py-2.5 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-orange-500/50 pr-10"
              />
              <button
                onClick={() => setShowClaude(!showClaude)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
              >
                {showClaude ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          {/* OpenAI Key */}
          <div className="space-y-1.5">
            <label className="text-xs text-slate-400">
              OpenAI API Key {!requireGPT && <span className="text-slate-500">(optional)</span>}
            </label>
            <div className="relative">
              <input
                type={showGpt ? "text" : "password"}
                value={gptKey}
                onChange={(e) => setGptKey(e.target.value)}
                placeholder="sk-..."
                className="w-full rounded-lg bg-slate-800 border border-slate-700 px-3 py-2.5 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-emerald-500/50 pr-10"
              />
              <button
                onClick={() => setShowGpt(!showGpt)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
              >
                {showGpt ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <div className="flex items-start gap-2 text-xs text-slate-500 bg-slate-800/50 rounded-lg p-3">
            <AlertCircle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
            <p>Keys are stored in localStorage only — never sent to any server except the respective AI providers during API calls.</p>
          </div>

          <Button
            onClick={save}
            disabled={!claudeKey || (requireGPT && !gptKey)}
            className="w-full"
          >
            Enter Lab
          </Button>
        </div>

        <button
          onClick={clear}
          className="w-full text-xs text-slate-600 hover:text-slate-400 transition-colors"
        >
          Clear saved keys
        </button>
      </div>
    </div>
  );
}

export function getStoredKeys(): { claude?: string; gpt?: string } {
  if (typeof window === "undefined") return {};
  return {
    claude: localStorage.getItem(CLAUDE_KEY) ?? undefined,
    gpt: localStorage.getItem(GPT_KEY) ?? undefined,
  };
}
