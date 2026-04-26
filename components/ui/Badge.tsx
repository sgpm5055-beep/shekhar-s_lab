import { ReactNode, CSSProperties } from "react";

interface BadgeProps {
  children: ReactNode;
  color?: string;   // hex or css var, e.g. "var(--claude)"
  className?: string;
}

export function Badge({ children, color = "var(--t4)" }: BadgeProps) {
  const style: CSSProperties = {
    display: "inline-flex",
    alignItems: "center",
    gap: 4,
    fontSize: 9,
    fontFamily: "var(--font-mono)",
    fontWeight: 500,
    letterSpacing: "0.14em",
    textTransform: "uppercase",
    color,
    background: `${color}15`,
    padding: "3px 10px",
    borderRadius: 20,
  };
  return <span style={style}>{children}</span>;
}
