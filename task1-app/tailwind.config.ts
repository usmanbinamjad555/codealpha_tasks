import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ["'Sora'", "sans-serif"],
        mono: ["'JetBrains Mono'", "monospace"],
      },
      colors: {
        glass: {
          border: "rgba(255,255,255,0.08)",
          bg: "rgba(255,255,255,0.04)",
          "bg-hover": "rgba(255,255,255,0.07)",
        },
        teal: {
          400: "#2DD4BF",
          500: "#14B8A6",
          600: "#0D9488",
        },
        navy: {
          900: "#060B18",
          800: "#0A1128",
          700: "#0F1A3E",
        },
      },
      backdropBlur: {
        xs: "2px",
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "fade-in": "fadeIn 0.4s ease forwards",
        "slide-up": "slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards",
      },
      keyframes: {
        fadeIn: {
          from: { opacity: "0" },
          to: { opacity: "1" },
        },
        slideUp: {
          from: { opacity: "0", transform: "translateY(16px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
      boxShadow: {
        glass: "0 0 0 1px rgba(255,255,255,0.06), 0 8px 40px rgba(0,0,0,0.4)",
        "glass-hover":
          "0 0 0 1px rgba(45,212,191,0.2), 0 8px 40px rgba(0,0,0,0.5)",
        "teal-glow": "0 0 30px rgba(45,212,191,0.15)",
      },
    },
  },
  plugins: [],
};
export default config;
