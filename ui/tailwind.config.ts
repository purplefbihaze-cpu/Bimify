import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./pages/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./layouts/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          bg: "var(--bg)",
          bg2: "var(--bg2)",
          fg: "var(--fg)",
        },
        accent: {
          300: "var(--accent-soft)",
          400: "var(--accent)",
          500: "var(--accent-strong)",
        },
      },
      boxShadow: {
        glass: "inset 0 1px 0 rgba(255,255,255,0.08), 0 10px 30px rgba(0,0,0,0.35)",
        depth: "0 10px 30px rgba(0,0,0,0.35)",
        soft: "0 4px 16px rgba(0,0,0,0.22)",
      },
      backdropBlur: {
        xs: "2px",
      },
      borderRadius: {
        xl: "1.25rem",
      },
      keyframes: {
        floaty: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-2px)" },
        },
        pulseSoft: {
          "0%, 100%": { opacity: "0.8" },
          "50%": { opacity: "1" },
        },
      },
      animation: {
        floaty: "floaty 6s ease-in-out infinite",
        pulseSoft: "pulseSoft 2.8s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;


