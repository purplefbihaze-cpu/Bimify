import { useEffect } from "react";
import GlassPanel from "@/components/GlassPanel";
import DepthButton from "@/components/DepthButton";
import { useSettings, MATRIX_DEFAULTS, MATRIX_LIMITS } from "@/lib/hooks/useSettings";

export default function SettingsPage() {
  const {
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
  } = useSettings();

  useEffect(() => {
    // no-op; just ensuring mount behavior
  }, []);

  return (
    <div className="grid gap-6 md:grid-cols-2 items-start">
      <GlassPanel className="p-4 space-y-4">
        <div className="text-lg font-medium">Einstellungen</div>
        <div className="space-y-2">
          <label htmlFor="rf-key" className="text-sm opacity-75">
            Roboflow API Key
          </label>
          <input
            id="rf-key"
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Private API Key (ohne rf_ Präfix)"
            className="w-full rounded-md bg-white/5 border border-white/10 px-3 py-2 focus-ring"
            autoComplete="off"
          />
          <div className="text-xs opacity-70">
            Verwende bevorzugt den privaten Roboflow-Key (ohne "rf_" Präfix). Der Publishable-Key funktioniert nur für freigegebene Projekte.
          </div>
          <div className="text-xs opacity-70">
            {loading ? "Lade..." : hasServerKey ? "Gespeichert auf Server" : "Noch nicht gespeichert"}
          </div>
          {error && <div className="text-sm text-red-300">{error}</div>}
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <label htmlFor="rf-project" className="text-sm opacity-75">
              Roboflow Project (workspace/project)
            </label>
            <input
              id="rf-project"
              type="text"
              value={project}
              onChange={(e) => setProject(e.target.value)}
              placeholder="workspace/project"
              className="w-full rounded-md bg-white/5 border border-white/10 px-3 py-2 focus-ring"
              autoComplete="off"
            />
          </div>
          <div className="space-y-2">
            <label htmlFor="rf-version" className="text-sm opacity-75">
              Version
            </label>
            <input
              id="rf-version"
              type="number"
              value={version ?? ""}
              onChange={(e) => setVersion(e.target.value ? parseInt(e.target.value, 10) : undefined)}
              placeholder="z.B. 31"
              className="w-full rounded-md bg-white/5 border border-white/10 px-3 py-2 focus-ring"
              autoComplete="off"
            />
          </div>
        </div>
        <div className="flex gap-3">
          <DepthButton disabled={saving} onClick={() => save(apiKey, project, version)}>
            {saving ? "Speichere..." : "Speichern"}
          </DepthButton>
          <DepthButton disabled={saving} onClick={() => save("")}>Entfernen</DepthButton>
        </div>
        <div className="text-xs opacity-70">
          Der Schlüssel wird sicher lokal gespeichert und serverseitig persistiert, und gilt sofort für neue Jobs.
        </div>
      </GlassPanel>
      <GlassPanel className="p-4 space-y-4">
        <div className="text-lg font-medium">Matrix-Hintergrund</div>
        <div className="flex items-center justify-between">
          <span className="text-sm opacity-75">Aktiv</span>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              className="h-4 w-4 rounded border border-white/20 bg-white/10 text-accent-400 focus-ring"
              checked={matrixEnabled}
              onChange={(event) => setMatrixEnabled(event.target.checked)}
            />
            <span>{matrixEnabled ? "An" : "Aus"}</span>
          </label>
        </div>
        <div className="space-y-3">
          <div className="space-y-1">
            <label htmlFor="matrix-speed" className="flex items-center justify-between text-sm opacity-75">
              <span>Geschwindigkeit</span>
              <span className="font-mono text-xs opacity-90">{matrixSpeed.toFixed(2)}</span>
            </label>
            <input
              id="matrix-speed"
              type="range"
              min={MATRIX_LIMITS.speed.min}
              max={MATRIX_LIMITS.speed.max}
              step={0.01}
              value={matrixSpeed}
              onChange={(event) => setMatrixSpeed(parseFloat(event.target.value))}
              className="w-full accent-accent-400"
            />
          </div>
          <div className="space-y-1">
            <label htmlFor="matrix-density" className="flex items-center justify-between text-sm opacity-75">
              <span>Dichte</span>
              <span className="font-mono text-xs opacity-90">{matrixDensity.toFixed(2)}</span>
            </label>
            <input
              id="matrix-density"
              type="range"
              min={MATRIX_LIMITS.density.min}
              max={MATRIX_LIMITS.density.max}
              step={0.05}
              value={matrixDensity}
              onChange={(event) => setMatrixDensity(parseFloat(event.target.value))}
              className="w-full accent-accent-400"
            />
          </div>
          <div className="space-y-1">
            <label htmlFor="matrix-opacity" className="flex items-center justify-between text-sm opacity-75">
              <span>Opacity</span>
              <span className="font-mono text-xs opacity-90">{matrixOpacity.toFixed(2)}</span>
            </label>
            <input
              id="matrix-opacity"
              type="range"
              min={MATRIX_LIMITS.opacity.min}
              max={MATRIX_LIMITS.opacity.max}
              step={0.01}
              value={matrixOpacity}
              onChange={(event) => setMatrixOpacity(parseFloat(event.target.value))}
              className="w-full accent-accent-400"
            />
          </div>
          <div className="space-y-1">
            <label htmlFor="matrix-color" className="text-sm opacity-75">
              Farbe
            </label>
            <div className="flex items-center gap-3">
              <input
                id="matrix-color"
                type="color"
                value={matrixColor}
                onChange={(event) => setMatrixColor(event.target.value)}
                className="h-9 w-14 cursor-pointer rounded border border-white/20 bg-transparent"
              />
              <input
                type="text"
                value={matrixColor}
                onChange={(event) => setMatrixColor(event.target.value)}
                className="flex-1 rounded-md bg-white/5 border border-white/10 px-3 py-2 font-mono text-sm uppercase tracking-wide focus-ring"
                placeholder="#1f6f4c"
              />
            </div>
          </div>
        </div>
        {matrixError && <div className="text-sm text-red-300">{matrixError}</div>}
        <div className="flex flex-wrap gap-3">
          <DepthButton disabled={savingMatrix} onClick={() => saveMatrix()}>
            {savingMatrix ? "Speichere..." : "Speichern"}
          </DepthButton>
          <DepthButton
            type="button"
            disabled={savingMatrix}
            onClick={() => {
              setMatrixEnabled(MATRIX_DEFAULTS.enabled);
              setMatrixSpeed(MATRIX_DEFAULTS.speed);
              setMatrixDensity(MATRIX_DEFAULTS.density);
              setMatrixOpacity(MATRIX_DEFAULTS.opacity);
              setMatrixColor(MATRIX_DEFAULTS.color);
              void saveMatrix({
                enabled: MATRIX_DEFAULTS.enabled,
                speed: MATRIX_DEFAULTS.speed,
                density: MATRIX_DEFAULTS.density,
                opacity: MATRIX_DEFAULTS.opacity,
                color: MATRIX_DEFAULTS.color,
              });
            }}
          >
            Zurücksetzen
          </DepthButton>
        </div>
        <div className="text-xs opacity-70">
          Passe Geschwindigkeit, Dichte, Transparenz und Farbe der Matrix an. Die Animation reagiert sanft auf diese Werte und bleibt hinter allen Inhalten.
        </div>
      </GlassPanel>
    </div>
  );
}


