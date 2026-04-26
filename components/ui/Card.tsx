import { ReactNode, CSSProperties } from "react";

interface CardProps {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
  intensity?: "normal" | "intense" | "subtle";
  accentColor?: string;
}

export function Card({ children, className, style, intensity = "normal", accentColor }: CardProps) {
  const intensityMap = {
    normal: { background: "var(--glass)", blur: 20 },
    intense: { background: "rgba(255,255,255,0.06)", blur: 24 },
    subtle: { background: "var(--s1)", blur: 12 },
  };
  const { background, blur } = intensityMap[intensity];

  return (
    <div
      className={className}
      style={{
        background,
        backdropFilter: `blur(${blur}px)`,
        WebkitBackdropFilter: `blur(${blur}px)`,
        border: "1px solid var(--b1)",
        borderRadius: 14,
        ...(accentColor && { boxShadow: `0 0 24px ${accentColor}10, 0 4px 32px rgba(0,0,0,0.12)` }),
        ...style,
      }}
    >
      {children}
    </div>
  );
}
