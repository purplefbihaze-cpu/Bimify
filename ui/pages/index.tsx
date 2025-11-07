import { useEffect, useMemo, useState } from "react";
import UploadDropzone from "@/components/UploadDropzone";
import GlassPanel from "@/components/GlassPanel";
import DepthButton from "@/components/DepthButton";
import type { ScaleCalibration } from "@/components/ScaleOverlay";
import PredictionViewer from "@/components/PredictionViewer";
import IfcJsViewerClient from "@/components/IfcJsViewerClient";
import IfcTopView from "@/components/IfcTopView";
import { useAnalyze } from "@/lib/hooks/useAnalyze";
import { useSettings } from "@/lib/hooks/useSettings";
import { exportIfcAsync, getIfcTopView, getJob, repairIfc } from "@/lib/api";
import type {
  AnalyzeOptions,
  ExportIFCJobResponse,
  ExportIFCResponse,
  IfcTopViewResponsePayload,
  IfcRepairResponsePayload,
  JobStatusResponse,
} from "@/lib/api";

const WINDOW_HEAD_MM = 2000;
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const POLL_INTERVAL_MS = 2000;
const POLL_TIMEOUT_MS = 10 * 60 * 1000;

const delay = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

export default function Home() {
  const { run, loading, error, result, reset } = useAnalyze();
  const { hasServerKey } = useSettings();
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0.5);
  const [drawLabels, setDrawLabels] = useState(true);
  const [drawShapes, setDrawShapes] = useState(true);
  const [showSensors, setShowSensors] = useState(false);
  const [activeTab, setActiveTab] = useState<"viewer" | "annotated" | "json">("viewer");
  const [zonesInput, setZonesInput] = useState<string>("");
  const [linesInput, setLinesInput] = useState<string>("");
  const [optionsError, setOptionsError] = useState<string | null>(null);
  const [storeyHeight, setStoreyHeight] = useState(3000);
  const [doorHeight, setDoorHeight] = useState(2100);
  const [windowHeight, setWindowHeight] = useState(1000);
  const [pxPerMm, setPxPerMm] = useState<string>("");
  const [createLoading, setCreateLoading] = useState(false);
  const [ifcError, setIfcError] = useState<string | null>(null);
  const [ifcResponse, setIfcResponse] = useState<ExportIFCResponse | null>(null);
  const [ifcImprovedUrl, setIfcImprovedUrl] = useState<string | null>(null);
  const [ifcStatsUrl, setIfcStatsUrl] = useState<string | null>(null);
  const [ifcJobId, setIfcJobId] = useState<string | null>(null);
  const [ifcJobProgress, setIfcJobProgress] = useState(0);
  const [scaleToolActive, setScaleToolActive] = useState(false);
  const [calibration, setCalibration] = useState<ScaleCalibration | null>(null);
  const [ifcBaseUrl, setIfcBaseUrl] = useState<string | null>(null);
  const [ifcTopViewUrl, setIfcTopViewUrl] = useState<string | null>(null);
  const [topViewError, setTopViewError] = useState<string | null>(null);
  const [repairLoading, setRepairLoading] = useState(false);
  const [repairError, setRepairError] = useState<string | null>(null);
  const [repairResponse, setRepairResponse] = useState<IfcRepairResponsePayload | null>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const selectFile = (value: File) => {
    setFile(value);
    setActiveTab("viewer");
    reset();
    setIfcResponse(null);
    setIfcError(null);
    setCreateLoading(false);
    setIfcBaseUrl(null);
    setIfcImprovedUrl(null);
    setIfcStatsUrl(null);
    setTopViewError(null);
    setCalibration(null);
    setScaleToolActive(false);
    setRepairResponse(null);
    setRepairError(null);
    setRepairLoading(false);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const url = URL.createObjectURL(value);
    setPreviewUrl(url);
  };

  const analyze = async () => {
    if (!file) return;
    setOptionsError(null);
    setScaleToolActive(false);
    const opts: AnalyzeOptions = { confidence };
    try {
      if (zonesInput.trim()) {
        const parsed = JSON.parse(zonesInput);
        if (!Array.isArray(parsed)) throw new Error("Zones JSON muss ein Array sein");
        opts.zones = parsed;
      }
      if (linesInput.trim()) {
        const parsed = JSON.parse(linesInput);
        if (!Array.isArray(parsed)) throw new Error("Lines JSON muss ein Array sein");
        opts.lines = parsed;
      }
    } catch (err: any) {
      setOptionsError(err?.message ? String(err.message) : "Ungültige Analyseoptionen");
      return;
    }
    await run(file, opts);
    setActiveTab("viewer");
    setIfcResponse(null);
    setIfcError(null);
    setIfcImprovedUrl(null);
    setIfcStatsUrl(null);
    setRepairResponse(null);
    setRepairError(null);
  };

  const createIfc = async () => {
    console.log("[createIfc] Starting...", { hasResult: !!result, predictionsCount: result?.predictions?.length });
    if (!result?.predictions?.length) {
      console.warn("[createIfc] No predictions available");
      return;
    }
    setCreateLoading(true);
    setIfcError(null);
    setIfcJobId(null);
    setIfcJobProgress(0);
    setRepairResponse(null);
    setRepairError(null);
    try {
      setIfcTopViewUrl(null);
      setTopViewError(null);
      const safeStorey = Number.isFinite(storeyHeight) && storeyHeight > 0 ? storeyHeight : 3000;
      const safeDoor = Number.isFinite(doorHeight) && doorHeight > 0 ? doorHeight : 2100;
      const safeWindow = Number.isFinite(windowHeight) && windowHeight > 0 ? windowHeight : 1000;
      const pxValue = (pxPerMm || "").trim();
      let safePxPerMm: number | null = null;
      if (pxValue !== "") {
        const pxNumber = Number(pxValue);
        if (Number.isFinite(pxNumber) && pxNumber > 0 && !Number.isNaN(pxNumber)) {
          safePxPerMm = pxNumber;
        }
      }
      console.log("[createIfc] pxPerMm validation", { raw: pxPerMm, trimmed: pxValue, parsed: safePxPerMm });
      const payload = {
        predictions: result.predictions,
        image: null,
        storey_height_mm: safeStorey,
        door_height_mm: safeDoor,
        window_height_mm: Math.min(safeWindow, WINDOW_HEAD_MM - 50),
        window_head_elevation_mm: WINDOW_HEAD_MM,
        ...(safePxPerMm !== null ? { px_per_mm: safePxPerMm } : {}),
        calibration: calibration
          ? {
              px_per_mm: calibration.pxPerMm,
              pixel_distance: calibration.pixelDistance,
              real_distance_mm: calibration.realDistanceMm,
              point_a: calibration.points[0],
              point_b: calibration.points[1],
              unit: calibration.unit,
            }
          : null,
      };
      console.log("[createIfc] Calling exportIfcAsync API...", { payloadSize: JSON.stringify(payload).length });
      const jobResponse: ExportIFCJobResponse = await exportIfcAsync(payload);
      console.log("[createIfc] Job created", { job_id: jobResponse.job_id });
      setIfcJobId(jobResponse.job_id);
      setIfcJobProgress(5);

      const maxAttempts = Math.ceil(POLL_TIMEOUT_MS / POLL_INTERVAL_MS);
      let jobResult: ExportIFCResponse | null = null;
      for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
        if (attempt > 0) {
          await delay(POLL_INTERVAL_MS);
        }

        let status: JobStatusResponse;
        try {
          status = await getJob(jobResponse.job_id);
        } catch (pollErr: any) {
          console.warn("[createIfc] getJob polling failed", pollErr);
          if (attempt === maxAttempts - 1) {
            throw pollErr;
          }
          continue;
        }

        setIfcJobProgress(typeof status.progress === "number" ? status.progress : 0);
        if (status.status === "succeeded") {
          if (!status.result) {
            throw new Error("IFC-Job abgeschlossen, aber kein Ergebnis erhalten.");
          }
          jobResult = status.result;
          setIfcJobProgress(100);
          break;
        }

        if (status.status === "failed") {
          const message = status.error ? String(status.error) : "IFC Export fehlgeschlagen";
          throw new Error(message);
        }
      }

      if (!jobResult) {
        throw new Error("IFC Export hat die maximale Wartezeit überschritten.");
      }

      console.log("[createIfc] Job completed", { file_name: jobResult.file_name, ifc_url: jobResult.ifc_url });
      setIfcResponse(jobResult);
      const baseUrl = `${API_BASE}${jobResult.ifc_url}`;
      setIfcBaseUrl(baseUrl);
      setIfcImprovedUrl(jobResult.improved_ifc_url ? `${API_BASE}${jobResult.improved_ifc_url}` : null);
      setIfcStatsUrl(jobResult.improved_ifc_stats_url ? `${API_BASE}${jobResult.improved_ifc_stats_url}` : null);

      try {
        const topView: IfcTopViewResponsePayload = await getIfcTopView({ file_name: jobResult.file_name });
        setIfcTopViewUrl(`${API_BASE}${topView.topview_url}`);
        setTopViewError(null);
      } catch (topErr: any) {
        setIfcTopViewUrl(null);
        setTopViewError(topErr?.message ? String(topErr.message) : "TopView konnte nicht erzeugt werden");
      }
    } catch (err: any) {
      console.error("[createIfc] Error occurred", err);
      setIfcError(err?.message ? String(err.message) : "IFC Export fehlgeschlagen");
      setIfcResponse(null);
      setIfcBaseUrl(null);
      setIfcImprovedUrl(null);
      setIfcStatsUrl(null);
      setIfcTopViewUrl(null);
      setTopViewError(null);
    } finally {
      setCreateLoading(false);
      setIfcJobId(null);
      setIfcJobProgress(0);
    }
  };

  // xBIM removed – no preprocess handler

  const handleCalibrationComplete = (value: ScaleCalibration) => {
    setCalibration(value);
    setScaleToolActive(false);
    setPxPerMm(value.pxPerMm > 0 ? value.pxPerMm.toFixed(4) : "");
  };

  const resetCalibration = () => {
    setCalibration(null);
  };

  const counts = useMemo(() => {
    if (!result) return [] as Array<[string, number]>;
    return Object.entries(result.per_class || {}).sort((a, b) => b[1] - a[1]);
  }, [result]);

  const runRepairLevelOne = async () => {
    const targetFileName = repairResponse?.file_name || ifcResponse?.file_name;
    if (!targetFileName) {
      setRepairError("Bitte zuerst eine IFC erzeugen.");
      return;
    }

    setRepairLoading(true);
    setRepairError(null);

    try {
      const response = await repairIfc({ file_name: targetFileName, level: 1 });
      setRepairResponse(response);

      const absoluteUrl = `${API_BASE}${response.ifc_url}`;
      setIfcBaseUrl(absoluteUrl);
      setIfcImprovedUrl(null);
      setIfcStatsUrl(null);

      if (response.topview_url) {
        setIfcTopViewUrl(`${API_BASE}${response.topview_url}`);
        setTopViewError(null);
      } else {
        setIfcTopViewUrl(null);
        setTopViewError("TopView konnte nicht aktualisiert werden.");
      }

      if (response.warnings && response.warnings.length === 0) {
        setRepairError(null);
      }
    } catch (err: any) {
      console.error("[repairIfc] Error occurred", err);
      setRepairError(err?.message ? String(err.message) : "Reparatur fehlgeschlagen");
    } finally {
      setRepairLoading(false);
    }
  };

  const hasIfc = Boolean(ifcBaseUrl);
  const canRepair = Boolean(ifcResponse?.file_name || repairResponse?.file_name);

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-[minmax(0,360px)_1fr] items-start">
        <div className="space-y-4">
          <UploadDropzone onSelect={selectFile} disabled={loading} />
          <GlassPanel className="space-y-4 p-4">
            <div>
              <div className="text-sm opacity-75">Ausgewähltes Bild</div>
              <div className="text-sm opacity-90 mt-1 break-all">{file ? file.name : "Kein Bild ausgewählt"}</div>
              {!hasServerKey && (
                <div className="text-xs text-amber-300 mt-2">
                  Kein Roboflow API Key gesetzt. Bitte in den Einstellungen eintragen.
                </div>
              )}
            </div>
            <div>
              <label className="text-sm opacity-75" htmlFor="confidence">
                Confidence Threshold
              </label>
              <input
                id="confidence"
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={confidence}
                onChange={(e) => setConfidence(parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="text-sm mt-1 opacity-80">{confidence.toFixed(2)}</div>
            </div>
            <div className="flex flex-wrap gap-3 text-sm">
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={drawLabels} onChange={(e) => setDrawLabels(e.target.checked)} />
                Draw Labels
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={drawShapes} onChange={(e) => setDrawShapes(e.target.checked)} />
                Draw Shapes
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={showSensors} onChange={(e) => setShowSensors(e.target.checked)} />
                Sensor Predictions
              </label>
            </div>
            <div className="space-y-2">
              <label className="text-xs uppercase tracking-[0.3em] opacity-60">Zonen (JSON)</label>
              <textarea
                value={zonesInput}
                onChange={(e) => setZonesInput(e.target.value)}
                placeholder='[ { "name": "Zone A", "points": [[0,0],[100,0],[100,100],[0,100]] } ]'
                className="w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-xs font-mono leading-relaxed focus-ring"
                rows={3}
              />
              <label className="text-xs uppercase tracking-[0.3em] opacity-60">Linien (JSON)</label>
              <textarea
                value={linesInput}
                onChange={(e) => setLinesInput(e.target.value)}
                placeholder='[ { "name": "Linie 1", "start": [0,0], "end": [640,0] } ]'
                className="w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-xs font-mono leading-relaxed focus-ring"
                rows={3}
              />
            </div>
            {optionsError && <div className="text-xs text-amber-300">{optionsError}</div>}
            {error && <div className="text-sm text-red-300">{error}</div>}
            <DepthButton disabled={!file || loading} onClick={analyze} className="w-full">
              {loading ? "Analysiere..." : "Analysieren"}
            </DepthButton>
          </GlassPanel>
          {(result?.total ?? 0) > 0 && (
            <GlassPanel className="p-4 space-y-2">
              <div className="text-sm font-medium">Erkannte Elemente</div>
              <div className="text-xs opacity-70">Total: {result?.total ?? 0}</div>
              <ul className="text-sm grid gap-1">
                {counts.map(([label, value]) => (
                  <li key={label} className="flex justify-between">
                    <span className="capitalize">{label || "unbekannt"}</span>
                    <span className="opacity-80">{value}</span>
                  </li>
                ))}
              </ul>
            </GlassPanel>
          )}
          {result?.predictions && result.predictions.length > 0 && (
            <GlassPanel className="p-4 space-y-4">
              <div className="flex items-center justify-between">
                <div className="text-sm font-medium">IFC Export</div>
                {ifcResponse && (
                  <span className="text-xs" style={{ color: "var(--accent-soft)" }}>
                    Bereit
                  </span>
                )}
              </div>
              <div className="grid gap-3 text-sm">
                <label className="grid gap-1">
                  <span className="opacity-70">Geschosshöhe (mm)</span>
                  <input
                    type="number"
                    min={2000}
                    step={50}
                    value={storeyHeight}
                    onChange={(e) => {
                      const next = Number(e.target.value);
                      setStoreyHeight(Number.isFinite(next) ? next : storeyHeight);
                    }}
                    className="rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm focus-ring"
                  />
                </label>
                <label className="grid gap-1">
                  <span className="opacity-70">Türhöhe (mm)</span>
                  <input
                    type="number"
                    min={1800}
                    step={25}
                    value={doorHeight}
                    onChange={(e) => {
                      const next = Number(e.target.value);
                      setDoorHeight(Number.isFinite(next) ? next : doorHeight);
                    }}
                    className="rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm focus-ring"
                  />
                </label>
                <label className="grid gap-1">
                  <span className="opacity-70">Fensterhöhe (mm)</span>
                  <input
                    type="number"
                    min={500}
                    step={25}
                    value={windowHeight}
                    onChange={(e) => {
                      const next = Number(e.target.value);
                      setWindowHeight(Number.isFinite(next) ? next : windowHeight);
                    }}
                    className="rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm focus-ring"
                  />
                </label>
                <label className="grid gap-1">
                  <span className="opacity-70">Fenstersturz (mm)</span>
                  <input
                    type="number"
                    value={WINDOW_HEAD_MM}
                    readOnly
                    className="rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm opacity-70"
                  />
                </label>
                <label className="grid gap-1">
                  <span className="opacity-70">px/mm (optional)</span>
                  <input
                    type="number"
                    min={0}
                    step={0.01}
                    value={pxPerMm}
                    onChange={(e) => setPxPerMm(e.target.value)}
                    placeholder="z. B. 2.4"
                    className="rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm focus-ring"
                  />
                  {calibration && (
                    <div
                      className="flex items-center justify-between text-xs"
                      style={{ color: "var(--accent-soft)" }}
                    >
                      <span>Kalibriert: {calibration.pxPerMm.toFixed(4)} px/mm</span>
                      <button
                        type="button"
                        className="underline-offset-4 hover:underline opacity-80"
                        onClick={resetCalibration}
                        style={{ color: "var(--accent-strong)" }}
                      >
                        Reset
                      </button>
                    </div>
                  )}
                </label>
              </div>
              {ifcError && <div className="text-xs text-red-300">{ifcError}</div>}
              {ifcResponse?.warnings && ifcResponse.warnings.length > 0 && (
                <div className="rounded-lg border border-amber-400/30 bg-amber-500/10 p-2 text-xs text-amber-200 space-y-1">
                  {ifcResponse.warnings.map((warning, idx) => (
                    <div key={`${warning}-${idx}`}>{warning}</div>
                  ))}
                </div>
              )}
              <DepthButton
                disabled={createLoading || !result?.predictions?.length}
                onClick={createIfc}
                className="w-full"
              >
                {createLoading
                  ? `IFC wird erzeugt${ifcJobProgress > 0 ? ` (${ifcJobProgress}%)` : "..."}`
                  : "IFC erzeugen"}
              </DepthButton>
              <div className="pt-4">
                <div className="flex flex-col items-center gap-3">
                  <button
                    type="button"
                    onClick={runRepairLevelOne}
                    disabled={repairLoading || !canRepair}
                    className={`w-48 rounded-xl border bg-white/5 px-4 py-3 text-[11px] uppercase tracking-[0.35em] transition-all duration-300 ${
                      repairLoading
                        ? "animate-pulse"
                        : canRepair
                        ? "hover:-translate-y-0.5 hover:shadow-[0_16px_40px_-16px_rgba(56,189,248,0.45)]"
                        : "opacity-40 cursor-not-allowed"
                    }`}
                    style={{
                      color: "var(--accent)",
                      borderColor: "var(--accent-soft)",
                    }}
                  >
                    {repairLoading ? "Reparatur läuft" : "Reparieren Level 1"}
                  </button>
                  <div className="flex flex-wrap justify-center gap-2 text-[10px] uppercase tracking-[0.35em] text-white/30">
                    {[2, 3, 4, 5].map((level) => (
                      <div
                        key={level}
                        className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 backdrop-blur-sm"
                      >
                        Level {level}
                      </div>
                    ))}
                  </div>
                </div>
                <div className="mt-3 space-y-2 text-center text-xs">
                  {repairError && <div className="text-red-300">{repairError}</div>}
                  {!repairError && repairResponse && !repairLoading && (
                    <div className="text-emerald-300">Reparatur abgeschlossen (Level {repairResponse.level})</div>
                  )}
                  {repairResponse?.warnings && repairResponse.warnings.length > 0 && (
                    <div className="mx-auto max-w-sm space-y-1 rounded-lg border border-amber-400/30 bg-amber-500/10 p-2 text-amber-200/90">
                      {repairResponse.warnings.map((warning, idx) => (
                        <div key={`${warning}-${idx}`}>{warning}</div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
              {createLoading && ifcJobId && (
                <div className="text-xs" style={{ color: "var(--accent-soft)" }}>
                  Job {ifcJobId} · Fortschritt {ifcJobProgress}%
                </div>
              )}
              {ifcResponse && (
                <div className="text-xs break-all" style={{ color: "var(--accent-soft)" }}>
                  Datei: {ifcResponse.file_name}
                </div>
              )}
            </GlassPanel>
          )}
          {result?.zones && result.zones.length > 0 && (
            <GlassPanel className="p-4 space-y-3">
              <div className="text-sm font-medium">Zone Counts</div>
              <div className="grid gap-2 text-sm">
                {result.zones.map((zone) => (
                  <div key={zone.name} className="rounded-lg border border-white/10 p-3">
                    <div className="flex justify-between text-xs uppercase tracking-[0.25em] opacity-60">
                      <span>{zone.name}</span>
                      <span>Total {zone.total}</span>
                    </div>
                    <ul className="mt-2 text-xs grid gap-1">
                      {Object.entries(zone.per_class || {}).map(([label, value]) => (
                        <li key={label} className="flex justify-between opacity-80">
                          <span className="capitalize">{label}</span>
                          <span>{value}</span>
                        </li>
                      ))}
                      {!Object.keys(zone.per_class || {}).length && <li className="opacity-60">Keine Treffer</li>}
                    </ul>
                  </div>
                ))}
              </div>
            </GlassPanel>
          )}
          {result?.lines && result.lines.length > 0 && (
            <GlassPanel className="p-4 space-y-3">
              <div className="text-sm font-medium">Line Counts</div>
              <div className="grid gap-2 text-sm">
                {result.lines.map((line) => (
                  <div key={line.name} className="rounded-lg border border-white/10 p-3 space-y-2">
                    <div className="text-xs uppercase tracking-[0.25em] opacity-60">{line.name}</div>
                    <div className="flex gap-3 text-xs">
                      {Object.entries(line.counts || {}).map(([side, value]) => (
                        <div key={side} className="flex-1">
                          <div className="opacity-60">{side}</div>
                          <div className="text-sm">{value}</div>
                        </div>
                      ))}
                    </div>
                    {Object.keys(line.per_class || {}).length > 0 && (
                      <div className="text-[11px] opacity-75">
                        {Object.entries(line.per_class).map(([label, sides]) => (
                          <div key={label} className="flex gap-3">
                            <span className="capitalize w-20">{label}</span>
                            <span className="flex gap-2">
                              {Object.entries(sides).map(([side, val]) => (
                                <span key={side}>
                                  {side}:{" "}
                                  {val}
                                </span>
                              ))}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </GlassPanel>
          )}
        </div>

        <div className="space-y-4">
          <GlassPanel className="p-4 space-y-4 min-h-[320px]">
            <div className="flex flex-col gap-3 border-b border-white/10 pb-3 text-sm sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-3">
                <button
                  className={`rounded-lg px-3 py-1 transition ${activeTab === "viewer" ? "bg-white/10" : "hover:bg-white/5"}`}
                  onClick={() => setActiveTab("viewer")}
                >
                  Viewer
                </button>
                <button
                  className={`rounded-lg px-3 py-1 transition ${activeTab === "annotated" ? "bg-white/10" : "hover:bg-white/5"}`}
                  onClick={() => setActiveTab("annotated")}
                >
                  Annotated
                </button>
                <button
                  className={`rounded-lg px-3 py-1 transition ${activeTab === "json" ? "bg-white/10" : "hover:bg-white/5"}`}
                  onClick={() => setActiveTab("json")}
                >
                  JSON
                </button>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  className={`rounded-lg px-3 py-1 transition ${scaleToolActive ? "bg-accent text-slate-900" : "hover:bg-white/5"}`}
                  onClick={() => setScaleToolActive((prev) => !prev)}
                  disabled={!previewUrl}
                >
                  Maßstab
                </button>
              </div>
            </div>

            {activeTab === "viewer" && (
              <div className="relative">
                {previewUrl ? (
                  <PredictionViewer
                    imageSrc={previewUrl}
                    imageMeta={result?.image}
                    predictions={result?.predictions ?? []}
                    drawLabels={drawLabels}
                    drawShapes={drawShapes}
                    showSensors={showSensors}
                    threshold={confidence}
                    scaleToolActive={scaleToolActive}
                    calibration={calibration}
                    onCalibrationComplete={handleCalibrationComplete}
                    onCalibrationCancel={() => setScaleToolActive(false)}
                  />
                ) : (
                  <div className="text-sm opacity-70">Bitte ein Bild auswählen und analysieren.</div>
                )}
              </div>
            )}

            {activeTab === "annotated" && (
              <div className="flex items-center justify-center">
                {result?.annotated_image ? (
                  <img
                    src={`data:image/png;base64,${result.annotated_image}`}
                    alt="Annotated"
                    className="max-h-[600px] w-full rounded-xl border border-white/10 object-contain"
                  />
                ) : (
                  <div className="text-sm opacity-70">Noch keine annotierte Darstellung vorhanden.</div>
                )}
              </div>
            )}

            {activeTab === "json" && (
              <div className="max-h-[480px] overflow-auto rounded-lg bg-black/40 p-3 text-xs leading-relaxed">
                {result ? <pre className="whitespace-pre-wrap break-words">{JSON.stringify(result.raw, null, 2)}</pre> : <span className="opacity-70">Noch keine Analyse durchgeführt.</span>}
              </div>
            )}
          </GlassPanel>
        {hasIfc && (
          <GlassPanel className="space-y-4 p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="text-sm font-medium">Modell Viewer</div>
                <div className="text-xs uppercase tracking-[0.3em] text-white/40">IFC Artefakte</div>
              </div>
              <div className="flex flex-wrap gap-3 text-xs">
                {ifcBaseUrl && (
                  <a href={ifcBaseUrl} target="_blank" rel="noreferrer" className="text-accent hover:underline">
                    IFC herunterladen
                  </a>
                )}
                {ifcImprovedUrl && ifcImprovedUrl !== ifcBaseUrl && (
                  <a href={ifcImprovedUrl} target="_blank" rel="noreferrer" className="text-accent hover:underline">
                    Optimiertes IFC
                  </a>
                )}
                {ifcStatsUrl && (
                  <a href={ifcStatsUrl} target="_blank" rel="noreferrer" className="text-accent hover:underline">
                    Stats JSON
                  </a>
                )}
              </div>
            </div>
            {/* xBIM controls removed */}
            <div className="grid gap-4 lg:grid-cols-2">
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs uppercase tracking-[0.3em] text-white/50">
                  <span>Top View (2D)</span>
                  {topViewError && <span className="text-[10px] normal-case tracking-normal text-rose-300/90">{topViewError}</span>}
                </div>
                <IfcTopView topviewUrl={ifcTopViewUrl ?? undefined} height="clamp(20rem, 60vh, 32rem)" />
              </div>
              <div className="space-y-2">
                <div className="text-xs uppercase tracking-[0.3em] text-white/50">IFC (3D)</div>
                <IfcJsViewerClient ifcUrl={ifcBaseUrl} height="clamp(20rem, 60vh, 32rem)" />
              </div>
            </div>
            {/* Secondary 3D panel removed to keep UI compact */}
          </GlassPanel>
        )}
        </div>
      </div>

    </div>

  );
}


