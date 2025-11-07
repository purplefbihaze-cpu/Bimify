import { createContext, ReactNode, useContext, useEffect, useLayoutEffect, useMemo, useState } from "react";
import { paletteDefinitions, PaletteName, highContrastOverrides, tokenKeys } from "@/lib/theme/palettes";

type ThemeMode = "light" | "dark" | "system";

type ThemeContextValue = {
  palette: PaletteName;
  mode: ThemeMode;
  highContrast: boolean;
  setPalette: (palette: PaletteName) => void;
  setMode: (mode: ThemeMode) => void;
  toggleHighContrast: () => void;
};

const ThemeContext = createContext<ThemeContextValue | null>(null);

const PALETTE_KEY = "bimify-theme-palette";
const MODE_KEY = "bimify-theme-mode";
const CONTRAST_KEY = "bimify-theme-contrast";

const defaultPalette: PaletteName = "base";

const useIsomorphicLayoutEffect = typeof window !== "undefined" ? useLayoutEffect : useEffect;

function resolveMode(mode: ThemeMode): "light" | "dark" {
  if (mode === "system") {
    if (typeof window !== "undefined") {
      return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    }
    return "dark";
  }
  return mode;
}

function applyTheme(palette: PaletteName, mode: ThemeMode, highContrast: boolean) {
  if (typeof document === "undefined") return;
  const resolved = resolveMode(mode);
  const paletteDef = paletteDefinitions[palette];
  const tone = paletteDef[resolved];
  const root = document.documentElement;

  tokenKeys.forEach((key) => {
    const baseValue = tone[key];
    const override = highContrast && highContrastOverrides[key as keyof typeof highContrastOverrides];
    root.style.setProperty(`--${key}`, override || baseValue);
  });

  root.dataset.themePalette = palette;
  root.dataset.themeMode = resolved;
  root.dataset.highContrast = String(highContrast);
  root.classList.toggle("dark", resolved === "dark");
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [palette, setPaletteState] = useState<PaletteName>(() => {
    if (typeof window === "undefined") return defaultPalette;
    return (window.localStorage.getItem(PALETTE_KEY) as PaletteName | null) || defaultPalette;
  });

  const [mode, setModeState] = useState<ThemeMode>(() => {
    if (typeof window === "undefined") return "system";
    return ((window.localStorage.getItem(MODE_KEY) as ThemeMode | null) || "system");
  });

  const [highContrast, setHighContrast] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    return window.localStorage.getItem(CONTRAST_KEY) === "true";
  });

  useIsomorphicLayoutEffect(() => {
    applyTheme(palette, mode, highContrast);
  }, [palette, mode, highContrast]);

  useEffect(() => {
    if (mode !== "system") return;
    if (typeof window === "undefined") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => applyTheme(palette, mode, highContrast);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [palette, mode, highContrast]);

  const api = useMemo<ThemeContextValue>(() => ({
    palette,
    mode,
    highContrast,
    setPalette: (next) => {
      setPaletteState(next);
      if (typeof window !== "undefined") window.localStorage.setItem(PALETTE_KEY, next);
    },
    setMode: (next) => {
      setModeState(next);
      if (typeof window !== "undefined") window.localStorage.setItem(MODE_KEY, next);
    },
    toggleHighContrast: () => {
      setHighContrast((prev) => {
        const next = !prev;
        if (typeof window !== "undefined") window.localStorage.setItem(CONTRAST_KEY, String(next));
        return next;
      });
    },
  }), [palette, mode, highContrast]);

  return <ThemeContext.Provider value={api}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
  return ctx;
}



