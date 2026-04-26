"use client";

import { ButtonHTMLAttributes, ReactNode } from "react";
import { cn } from "@/lib/utils/cn";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md" | "lg";
  children: ReactNode;
}

export function Button({ variant = "primary", size = "md", className, style, children, ...props }: ButtonProps) {
  const base: React.CSSProperties = {
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    fontFamily: "var(--font-display)",
    fontWeight: 700,
    letterSpacing: "-0.01em",
    borderRadius: 10,
    border: "none",
    cursor: "pointer",
    transition: "all 0.2s cubic-bezier(.4,0,.2,1)",
    gap: 8,
    ...(size === "sm" && { fontSize: 11, padding: "7px 16px" }),
    ...(size === "md" && { fontSize: 13, padding: "10px 24px" }),
    ...(size === "lg" && { fontSize: 13, padding: "11px 28px" }),
  };

  const variants: Record<string, React.CSSProperties> = {
    primary: {
      background: "linear-gradient(135deg, #F59E0B, #EF4444)",
      color: "#fff",
      boxShadow: "0 4px 20px rgba(245,158,11,0.25)",
    },
    secondary: {
      background: "var(--glass)",
      backdropFilter: "blur(16px)",
      WebkitBackdropFilter: "blur(16px)",
      border: "1px solid var(--b1)",
      color: "var(--t2)",
    },
    ghost: {
      background: "transparent",
      color: "var(--t3)",
    },
  };

  return (
    <button
      style={{ ...base, ...variants[variant], ...style }}
      className={cn("disabled:opacity-40 disabled:cursor-not-allowed w-full", className)}
      {...props}
    >
      {children}
    </button>
  );
}
