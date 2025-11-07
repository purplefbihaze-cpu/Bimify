export type PaletteName = "base" | "warm" | "cool";

export const tokenKeys = [
  "bg",
  "bg2",
  "fg",
  "accent",
  "accent-soft",
  "accent-strong",
  "glass",
  "glass-strong",
  "gradient-a",
  "gradient-b",
] as const;

export type TokenKey = (typeof tokenKeys)[number];

type Tone = Record<TokenKey, string>;

export type PaletteDefinition = {
  name: PaletteName;
  light: Tone;
  dark: Tone;
};

export const paletteDefinitions: Record<PaletteName, PaletteDefinition> = {
  base: {
    name: "base",
    light: {
      bg: "#f6f7fb",
      bg2: "#eef1f9",
      fg: "#0b1325",
      accent: "#1f6f4c",
      "accent-soft": "#4a8f6c",
      "accent-strong": "#d4b76f",
      glass: "rgba(255,255,255,0.58)",
      "glass-strong": "rgba(255,255,255,0.72)",
      "gradient-a": "rgba(82,140,112,0.22)",
      "gradient-b": "rgba(230,205,150,0.3)",
    },
    dark: {
      bg: "#0b0f14",
      bg2: "#0f141b",
      fg: "#e5ecf5",
      accent: "#1f6f4c",
      "accent-soft": "#4a8f6c",
      "accent-strong": "#d4b76f",
      glass: "rgba(15,19,27,0.55)",
      "glass-strong": "rgba(18,24,34,0.72)",
      "gradient-a": "rgba(31,111,76,0.2)",
      "gradient-b": "rgba(212,183,111,0.24)",
    },
  },
  warm: {
    name: "warm",
    light: {
      bg: "#faf6f2",
      bg2: "#f4eae1",
      fg: "#2b1611",
      accent: "#1f6f4c",
      "accent-soft": "#4a8f6c",
      "accent-strong": "#d4b76f",
      glass: "rgba(255,245,238,0.6)",
      "glass-strong": "rgba(255,235,220,0.78)",
      "gradient-a": "rgba(255,176,132,0.38)",
      "gradient-b": "rgba(255,133,133,0.42)",
    },
    dark: {
      bg: "#14100d",
      bg2: "#1c1611",
      fg: "#fbe9de",
      accent: "#1f6f4c",
      "accent-soft": "#4a8f6c",
      "accent-strong": "#d4b76f",
      glass: "rgba(38,24,18,0.6)",
      "glass-strong": "rgba(52,30,20,0.78)",
      "gradient-a": "rgba(255,143,101,0.22)",
      "gradient-b": "rgba(255,99,99,0.28)",
    },
  },
  cool: {
    name: "cool",
    light: {
      bg: "#f2f8fa",
      bg2: "#e6f1f4",
      fg: "#0e1f24",
      accent: "#1f6f4c",
      "accent-soft": "#4a8f6c",
      "accent-strong": "#d4b76f",
      glass: "rgba(230,247,255,0.6)",
      "glass-strong": "rgba(215,241,255,0.75)",
      "gradient-a": "rgba(142,246,255,0.38)",
      "gradient-b": "rgba(89,189,255,0.42)",
    },
    dark: {
      bg: "#061018",
      bg2: "#071823",
      fg: "#e2faff",
      accent: "#1f6f4c",
      "accent-soft": "#4a8f6c",
      "accent-strong": "#d4b76f",
      glass: "rgba(6,25,36,0.6)",
      "glass-strong": "rgba(12,32,44,0.78)",
      "gradient-a": "rgba(111,237,255,0.22)",
      "gradient-b": "rgba(92,160,255,0.28)",
    },
  },
};
export const highContrastOverrides: Partial<Tone> = {
  fg: "#ffffff",
  accent: "#2da169",
  "accent-soft": "#57c792",
  "accent-strong": "#f1d48a",
  glass: "rgba(0,0,0,0.7)",
  "glass-strong": "rgba(0,0,0,0.85)",
};


