import type { Metadata } from "next";
import { Instrument_Sans, Geist_Mono } from "next/font/google";
import "./globals.css";

const instrumentSans = Instrument_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-display",
  display: "swap",
});

const geistMono = Geist_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Shekhar's Research Lab v3",
  description: "Unified multi-model AI agent platform — Battle Arena & Orchestration Workbench",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${instrumentSans.variable} ${geistMono.variable} h-full`}>
      <body
        className="h-full antialiased"
        style={{ fontFamily: "var(--font-display, 'Instrument Sans', sans-serif)", background: "#09090B" }}
      >
        {/* Ambient light bleed */}
        <div
          aria-hidden
          style={{
            position: "fixed", inset: 0,
            background: `
              radial-gradient(ellipse at 20% 10%, rgba(245,158,11,0.04) 0%, transparent 50%),
              radial-gradient(ellipse at 80% 90%, rgba(16,185,129,0.03)  0%, transparent 50%),
              radial-gradient(ellipse at 50% 50%, rgba(167,139,250,0.02) 0%, transparent 60%)
            `,
            pointerEvents: "none",
            zIndex: 0,
          }}
        />
        <div style={{ position: "relative", zIndex: 1, height: "100%" }}>
          {children}
        </div>
      </body>
    </html>
  );
}
