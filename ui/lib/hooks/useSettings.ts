import { useEffect, useState } from "react";
import { getSettings, saveSettings } from "@/lib/api";

const LS_KEY = "roboflowApiKey";
export type MatrixConfig = {
  enabled: boolean;
  speed: number;
  density: number;
  opacity: number;
  color: string;
};

export const MATRIX_DEFAULTS: MatrixConfig = {
  enabled: true,
  speed: 1,
  density: 1,
  opacity: 0.35,
  color: "#1f6f4c",
};
export const MATRIX_LIMITS = {
  speed: { min: 0.02, max: 3 },
  density: { min: 0.5, max: 2 },
  opacity: { min: 0.05, max: 0.6 },
} as const;
const MATRIX_EVENT = "ifc-matrix-config-updated";

const clamp = (value: number, min: number, max: number) => {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
};

export function useSettings() {
  const [apiKey, setApiKey] = useState<string>("");
  const [project, setProject] = useState<string>("");
  const [version, setVersion] = useState<number | undefined>(undefined);
  const [hasServerKey, setHasServerKey] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);
  const [saving, setSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [matrixEnabled, setMatrixEnabled] = useState<boolean>(MATRIX_DEFAULTS.enabled);
  const [matrixSpeed, setMatrixSpeed] = useState<number>(MATRIX_DEFAULTS.speed);
  const [matrixDensity, setMatrixDensity] = useState<number>(MATRIX_DEFAULTS.density);
  const [matrixOpacity, setMatrixOpacity] = useState<number>(MATRIX_DEFAULTS.opacity);
  const [matrixColor, setMatrixColor] = useState<string>(MATRIX_DEFAULTS.color);
  const [savingMatrix, setSavingMatrix] = useState<boolean>(false);
  const [matrixError, setMatrixError] = useState<string | null>(null);

  useEffect(() => {
    const ls = (typeof window !== "undefined" && window.localStorage.getItem(LS_KEY)) || "";
    if (ls) setApiKey(ls);
    (async () => {
      try {
        const settings = await getSettings();
        setHasServerKey(Boolean(settings.has_roboflow_api_key));
        if (settings.roboflow_project) setProject(settings.roboflow_project);
        if (settings.roboflow_version != null) setVersion(settings.roboflow_version);
        if (settings.matrix_enabled != null) setMatrixEnabled(Boolean(settings.matrix_enabled));
        if (settings.matrix_speed != null)
          setMatrixSpeed(clamp(Number(settings.matrix_speed), MATRIX_LIMITS.speed.min, MATRIX_LIMITS.speed.max));
        if (settings.matrix_density != null)
          setMatrixDensity(clamp(Number(settings.matrix_density), MATRIX_LIMITS.density.min, MATRIX_LIMITS.density.max));
        if (settings.matrix_opacity != null)
          setMatrixOpacity(clamp(Number(settings.matrix_opacity), MATRIX_LIMITS.opacity.min, MATRIX_LIMITS.opacity.max));
        if (settings.matrix_color != null && settings.matrix_color.trim()) setMatrixColor(settings.matrix_color);
      } catch (e: any) {
        // ignore; offline or server not running
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  async function save(key: string, proj?: string, ver?: number) {
    setSaving(true);
    setError(null);
    try {
      const trimmed = (key || "").trim();
      await saveSettings({
        roboflow_api_key: trimmed,
        roboflow_project: (((proj ?? project) || "").trim()) || null,
        roboflow_version: ver ?? version ?? null,
      });
      if (typeof window !== "undefined") {
        if (trimmed) window.localStorage.setItem(LS_KEY, trimmed);
        else window.localStorage.removeItem(LS_KEY);
      }
      setApiKey(trimmed);
      setHasServerKey(Boolean(trimmed));
      if (proj != null) setProject(proj);
      if (ver != null) setVersion(ver);
    } catch (e: any) {
      setError(e.message || String(e));
      throw e;
    } finally {
      setSaving(false);
    }
  }

  async function saveMatrix(options?: {
    enabled?: boolean;
    speed?: number;
    density?: number;
    opacity?: number;
    color?: string;
  }) {
    setSavingMatrix(true);
    setMatrixError(null);
    try {
      const nextEnabled = options?.enabled ?? matrixEnabled;
      const desiredSpeed = options?.speed ?? matrixSpeed;
      const desiredDensity = options?.density ?? matrixDensity;
      const desiredOpacity = options?.opacity ?? matrixOpacity;
      const nextSpeed = clamp(desiredSpeed, MATRIX_LIMITS.speed.min, MATRIX_LIMITS.speed.max);
      const nextDensity = clamp(desiredDensity, MATRIX_LIMITS.density.min, MATRIX_LIMITS.density.max);
      const nextOpacity = clamp(desiredOpacity, MATRIX_LIMITS.opacity.min, MATRIX_LIMITS.opacity.max);
      const nextColor = (options?.color ?? matrixColor)?.trim() || MATRIX_DEFAULTS.color;
      await saveSettings({
        matrix_enabled: nextEnabled,
        matrix_speed: nextSpeed,
        matrix_density: nextDensity,
        matrix_opacity: nextOpacity,
        matrix_color: nextColor,
      });
      setMatrixEnabled(Boolean(nextEnabled));
      setMatrixSpeed(Number(nextSpeed));
      setMatrixDensity(Number(nextDensity));
      setMatrixOpacity(Number(nextOpacity));
      setMatrixColor(nextColor);
      if (typeof window !== "undefined") {
        const detail: MatrixConfig = {
          enabled: Boolean(nextEnabled),
          speed: Number(nextSpeed),
          density: Number(nextDensity),
          opacity: Number(nextOpacity),
          color: nextColor,
        };
        window.dispatchEvent(new CustomEvent<MatrixConfig>(MATRIX_EVENT, { detail }));
      }
    } catch (e: any) {
      setMatrixError(e.message || String(e));
      throw e;
    } finally {
      setSavingMatrix(false);
    }
  }

  return {
    apiKey,
    setApiKey,
    project,
    setProject,
    version,
    setVersion,
    hasServerKey,
    loading,
    saving,
    error,
    save,
    matrixEnabled,
    setMatrixEnabled,
    matrixSpeed,
    setMatrixSpeed,
    matrixDensity,
    setMatrixDensity,
    matrixOpacity,
    setMatrixOpacity,
    matrixColor,
    setMatrixColor,
    saveMatrix,
    savingMatrix,
    matrixError,
  };
}

export function useMatrixBackgroundConfig() {
  const [config, setConfig] = useState<MatrixConfig>({ ...MATRIX_DEFAULTS });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const settings = await getSettings();
        if (cancelled) return;
        setConfig({
          enabled: settings.matrix_enabled != null ? Boolean(settings.matrix_enabled) : MATRIX_DEFAULTS.enabled,
          speed:
            settings.matrix_speed != null
              ? clamp(Number(settings.matrix_speed), MATRIX_LIMITS.speed.min, MATRIX_LIMITS.speed.max)
              : MATRIX_DEFAULTS.speed,
          density:
            settings.matrix_density != null
              ? clamp(Number(settings.matrix_density), MATRIX_LIMITS.density.min, MATRIX_LIMITS.density.max)
              : MATRIX_DEFAULTS.density,
          opacity:
            settings.matrix_opacity != null
              ? clamp(Number(settings.matrix_opacity), MATRIX_LIMITS.opacity.min, MATRIX_LIMITS.opacity.max)
              : MATRIX_DEFAULTS.opacity,
          color:
            settings.matrix_color != null && settings.matrix_color.trim()
              ? settings.matrix_color
              : MATRIX_DEFAULTS.color,
        });
      } catch (error) {
        // ignore fetch issues; fall back to defaults
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = (event: Event) => {
      const custom = event as CustomEvent<MatrixConfig>;
      if (!custom.detail) return;
      setConfig({
        enabled: custom.detail.enabled,
        speed: clamp(custom.detail.speed, MATRIX_LIMITS.speed.min, MATRIX_LIMITS.speed.max),
        density: clamp(custom.detail.density, MATRIX_LIMITS.density.min, MATRIX_LIMITS.density.max),
        opacity: clamp(custom.detail.opacity, MATRIX_LIMITS.opacity.min, MATRIX_LIMITS.opacity.max),
        color: custom.detail.color,
      });
    };
    window.addEventListener(MATRIX_EVENT, handler as EventListener);
    return () => {
      window.removeEventListener(MATRIX_EVENT, handler as EventListener);
    };
  }, []);

  return { config, loading };
}


