import { useEffect, useMemo, useRef, useState } from "react";
import UploadDropzone from "@/components/UploadDropzone";
import GlassPanel from "@/components/GlassPanel";
import type { ScaleCalibration } from "@/components/ScaleOverlay";
import PredictionViewer from "@/components/PredictionViewer";
import Modal from "@/components/Modal";
import IfcJsViewerClient from "@/components/IfcJsViewerClient";
import IfcTopView from "@/components/IfcTopView";
import { useAnalyze } from "@/lib/hooks/useAnalyze";
import { useSettings } from "@/lib/hooks/useSettings";
import { exportIfcAsync, exportIfcV2Async, getIfcTopView, getJob, repairIfcAsync } from "@/lib/api";
import type {
  AnalyzeOptions,
  AnalyzeResponse,
  ExportIFCJobResponse,
  ExportIFCResponse,
  ExportIFCV2JobResponse,
  ExportIFCV2Response,
  IfcTopViewResponsePayload,
  IfcRepairResponsePayload,
  JobStatusResponse,
} from "@/lib/api";

const GEOMETRY_TITLE = "Geometrie";
const GEOMETRY_BUTTON = "Geometrie analysieren";

const WINDOW_HEAD_MM = 2000;
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const POLL_INTERVAL_MS = 2000;
const POLL_TIMEOUT_MS = 10 * 60 * 1000;

const delay = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

export default function Home() {
  const { run, loading, error, reset } = useAnalyze();
  const { hasServerKey } = useSettings();
  const previewContainerRef = useRef<HTMLDivElement | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"viewer" | "annotated" | "json">("viewer");
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
  const [repairJobId, setRepairJobId] = useState<string | null>(null);
  const [repairJobProgress, setRepairJobProgress] = useState(0);
  const [createV2Loading, setCreateV2Loading] = useState(false);
  const [ifcV2Error, setIfcV2Error] = useState<string | null>(null);
  const [ifcV2Response, setIfcV2Response] = useState<ExportIFCV2Response | null>(null);
  const [ifcV2ViewerUrl, setIfcV2ViewerUrl] = useState<string | null>(null);
  const [ifcV2JobId, setIfcV2JobId] = useState<string | null>(null);
  const [ifcV2JobProgress, setIfcV2JobProgress] = useState(0);
  const [ifcV2BaseUrl, setIfcV2BaseUrl] = useState<string | null>(null);
  const [ifcV2TopViewUrl, setIfcV2TopViewUrl] = useState<string | null>(null);
  const [topViewV2Error, setTopViewV2Error] = useState<string | null>(null);
  const [geometryResult, setGeometryResult] = useState<AnalyzeResponse | null>(null);
  const [geometryConfidence, setGeometryConfidence] = useState<number>(0.5);
  const [analyzedConfidence, setAnalyzedConfidence] = useState<number | null>(null);
  const [drawLabels, setDrawLabels] = useState<boolean>(true);
  const [drawShapes, setDrawShapes] = useState<boolean>(true);
  const [showSensors, setShowSensors] = useState<boolean>(false);
  const [geometryLocked, setGeometryLocked] = useState<boolean>(false);
  const [controlsReady, setControlsReady] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    const rafId = window.requestAnimationFrame(() => setControlsReady(true));
    return () => window.cancelAnimationFrame(rafId);
  }, []);

  useEffect(() => {
    if (modalOpen && !geometryResult) {
      setModalOpen(false);
    }
  }, [modalOpen, geometryResult]);

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
    setModalOpen(false);
    setGeometryResult(null);
    setGeometryConfidence(0.5);
    setDrawLabels(true);
    setDrawShapes(true);
    setShowSensors(false);
    setGeometryLocked(false);
    setAnalyzedConfidence(null);
    // V2 state reset
    setIfcV2Response(null);
    setIfcV2Error(null);
    setCreateV2Loading(false);
    setIfcV2BaseUrl(null);
    setIfcV2ViewerUrl(null);
    setIfcV2TopViewUrl(null);
    setTopViewV2Error(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const url = URL.createObjectURL(value);
    setPreviewUrl(url);
  };

  const analyzeGeometry = async () => {
    if (!file) return;
    if (geometryLocked && geometryResult) return;

    setScaleToolActive(false);
    const opts: AnalyzeOptions = {
      confidence: geometryConfidence,
      model_kind: "geometry",
    };

    setIsAnalyzing(true);

    try {
      const response = await run(file, opts);
      setGeometryResult(response);
      setGeometryLocked(true);
      setAnalyzedConfidence(geometryConfidence);
      setActiveTab("viewer");
      setIfcResponse(null);
      setIfcError(null);
      setIfcImprovedUrl(null);
      setIfcStatsUrl(null);
      setRepairResponse(null);
      setRepairError(null);
    } catch (err: any) {
      // error message handled by hook's error state
    } finally {
      setIsAnalyzing(false);
    }
  };


  const createIfc = async () => {
    console.log("[createIfc] Starting...", {
      hasGeometryResult: !!geometryResult,
      predictionsCount: geometryResult?.predictions?.length,
    });
    if (!geometryResult?.predictions?.length) {
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
        predictions: geometryResult.predictions,
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

  const createIfcV2 = async () => {
    console.log("[createIfcV2] Starting IFC export V2...", {
      hasGeometryResult: !!geometryResult,
      predictionsCount: geometryResult?.predictions?.length,
    });
    if (!geometryResult?.predictions?.length) {
      console.warn("[createIfcV2] No predictions available");
      return;
    }
    setCreateV2Loading(true);
    setIfcV2Error(null);
    setIfcV2JobId(null);
    setIfcV2JobProgress(0);
    setRepairResponse(null);
    setRepairError(null);
    try {
      setIfcV2TopViewUrl(null);
      setTopViewV2Error(null);
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
      console.log("[createIfcV2] pxPerMm validation", { raw: pxPerMm, trimmed: pxValue, parsed: safePxPerMm });
      const payload = {
        predictions: geometryResult.predictions,
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
      console.log("[createIfcV2] Calling exportIfcV2Async API - NEW IMPLEMENTATION...", { payloadSize: JSON.stringify(payload).length });
      const jobResponse: ExportIFCV2JobResponse = await exportIfcV2Async(payload);
      console.log("[createIfcV2] V2 Job created", { job_id: jobResponse.job_id });
      setIfcV2JobId(jobResponse.job_id);
      setIfcV2JobProgress(5);

      const maxAttempts = Math.ceil(POLL_TIMEOUT_MS / POLL_INTERVAL_MS);
      console.log("[createIfcV2] Starting polling", { job_id: jobResponse.job_id, maxAttempts, pollInterval: POLL_INTERVAL_MS });
      let jobResult: ExportIFCV2Response | null = null;
      for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
        if (attempt > 0) {
          await delay(POLL_INTERVAL_MS);
        }

        let status: JobStatusResponse;
        try {
          status = await getJob(jobResponse.job_id);
          console.log("[createIfcV2] Polling attempt", {
            attempt: attempt + 1,
            maxAttempts,
            status: status.status,
            progress: status.progress,
            hasResult: !!status.result,
            error: status.error,
          });
        } catch (pollErr: any) {
          console.warn("[createIfcV2] getJob polling failed", {
            attempt: attempt + 1,
            maxAttempts,
            error: pollErr?.message || String(pollErr),
            stack: pollErr?.stack,
          });
          if (attempt === maxAttempts - 1) {
            throw pollErr;
          }
          continue;
        }

        setIfcV2JobProgress(typeof status.progress === "number" ? status.progress : 0);
        
        if (status.status === "queued") {
          console.log("[createIfcV2] Job still queued", {
            attempt: attempt + 1,
            progress: status.progress,
          });
        }
        
        if (status.status === "succeeded") {
          if (!status.result) {
            throw new Error("IFC-V2-Job abgeschlossen, aber kein Ergebnis erhalten.");
          }
          jobResult = status.result as ExportIFCV2Response;
          setIfcV2JobProgress(100);
          console.log("[createIfcV2] Job succeeded", {
            attempt: attempt + 1,
            file_name: jobResult.file_name,
            ifc_url: jobResult.ifc_url,
          });
          break;
        }

        if (status.status === "failed") {
          const message = status.error ? String(status.error) : "IFC Export V2 fehlgeschlagen";
          console.error("[createIfcV2] Job failed", {
            attempt: attempt + 1,
            error: message,
            statusError: status.error,
          });
          throw new Error(message);
        }
      }

      if (!jobResult) {
        throw new Error("IFC Export V2 hat die maximale Wartezeit überschritten.");
      }

      console.log("[createIfcV2] Job completed", { file_name: jobResult.file_name, ifc_url: jobResult.ifc_url });
      setIfcV2Response(jobResult);
      const baseUrl = `${API_BASE}${jobResult.ifc_url}`;
      setIfcV2BaseUrl(baseUrl);
      setIfcV2ViewerUrl(jobResult.viewer_url ? `${API_BASE}${jobResult.viewer_url}` : null);
      setIfcV2TopViewUrl(jobResult.topview_url ? `${API_BASE}${jobResult.topview_url}` : null);
    } catch (err: any) {
      console.error("[createIfcV2] Error occurred", err);
      setIfcV2Error(err?.message ? String(err.message) : "IFC Export V2 fehlgeschlagen");
      setIfcV2Response(null);
      setIfcV2BaseUrl(null);
      setIfcV2ViewerUrl(null);
      setIfcV2TopViewUrl(null);
      setTopViewV2Error(null);
    } finally {
      setCreateV2Loading(false);
      setIfcV2JobId(null);
      setIfcV2JobProgress(0);
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
    if (!geometryResult) return [] as Array<[string, number]>;
    return Object.entries(geometryResult.per_class || {}).sort((a, b) => b[1] - a[1]);
  }, [geometryResult]);

  const runRepairLevelOne = async () => {
    const targetFileName = repairResponse?.file_name || ifcResponse?.file_name;
    if (!targetFileName) {
      setRepairError("Bitte zuerst eine IFC erzeugen.");
      return;
    }

    setRepairLoading(true);
    setRepairError(null);
    setRepairJobId(null);
    setRepairJobProgress(0);
    try {
      const jobResponse = await repairIfcAsync({ file_name: targetFileName, level: 1 });
      setRepairJobId(jobResponse.job_id);
      setRepairJobProgress(5);

      const maxAttempts = Math.ceil(POLL_TIMEOUT_MS / POLL_INTERVAL_MS);
      let jobResult: IfcRepairResponsePayload | null = null;
      for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
        if (attempt > 0) {
          await delay(POLL_INTERVAL_MS);
        }

        let status: JobStatusResponse;
        try {
          status = await getJob(jobResponse.job_id);
        } catch (pollErr: any) {
          console.warn("[runRepairLevelOne] getJob polling failed", pollErr);
          if (attempt === maxAttempts - 1) {
            throw pollErr;
          }
          continue;
        }

        setRepairJobProgress(typeof status.progress === "number" ? status.progress : 0);
        if (status.status === "succeeded") {
          if (!status.result) {
            throw new Error("Repair-Job abgeschlossen, aber kein Ergebnis erhalten.");
          }
          jobResult = status.result as IfcRepairResponsePayload;
          setRepairJobProgress(100);
          break;
        }

        if (status.status === "failed") {
          const message = status.error ? String(status.error) : "Reparatur fehlgeschlagen";
          throw new Error(message);
        }
      }

      if (!jobResult) {
        throw new Error("Reparatur hat die maximale Wartezeit überschritten.");
      }

      setRepairResponse(jobResult);
      const absoluteUrl = `${API_BASE}${jobResult.ifc_url}`;
      setIfcBaseUrl(absoluteUrl);
      setIfcImprovedUrl(null);
      setIfcStatsUrl(null);
      if (jobResult.topview_url) {
        setIfcTopViewUrl(`${API_BASE}${jobResult.topview_url}`);
        setTopViewError(null);
      }
    } catch (err: any) {
      console.error("[runRepairLevelOne] Error occurred", err);
      setRepairError(err?.message ? String(err.message) : "Reparatur fehlgeschlagen");
    } finally {
      setRepairLoading(false);
      setRepairJobId(null);
      setRepairJobProgress(0);
    }
  };

  const hasIfc = Boolean(ifcBaseUrl);
  const hasIfcV2 = Boolean(ifcV2BaseUrl);
  const canRepair = Boolean(ifcResponse?.file_name || repairResponse?.file_name);

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-[minmax(0,360px)_1fr] items-start">
        <div className="space-y-4" ref={previewContainerRef}>
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
            {error && (
              <div className="space-y-1">
                <div className="text-sm text-red-300">{error}</div>
                {error.includes("Roboflow API") && error.includes("403") && (
                  <div className="text-xs text-red-200/70">
                    Tipp: Überprüfe deinen API-Key in den{" "}
                    <a href="/settings" className="underline hover:text-red-100">
                      Einstellungen
                    </a>
                    .
                  </div>
                )}
              </div>
            )}
            {previewUrl && !geometryResult && (
              <div
                className={`rounded-2xl border border-white/10 bg-slate-950/40 p-4 shadow-[0_18px_38px_-24px_rgba(56,189,248,0.65)] backdrop-blur-sm transition-all duration-500 mb-4 ${
                  controlsReady ? "translate-y-0 opacity-100" : "translate-y-3 opacity-0"
                }`}
              >
                <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.35em] text-white/60 mb-4">
                  <span>Confidence Threshold</span>
                  <span className="font-mono text-sky-200">
                    {Math.round(geometryConfidence * 100)}%
                  </span>
                </div>
                <div className="relative h-2 rounded-full bg-slate-900/60">
                  <div
                    className="pointer-events-none absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-sky-500 via-cyan-400 to-emerald-400"
                    style={{ width: `${Math.max(0, Math.min(100, geometryConfidence * 100))}%` }}
                  />
                  <input
                    id="confidence-geometry-before-analysis"
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={geometryConfidence}
                    onChange={(e) => setGeometryConfidence(parseFloat(e.target.value))}
                    className="glass-range absolute inset-x-0 -top-2 h-6 w-full"
                  />
                </div>
                <div className="mt-3 text-[10px] text-white/50">
                  Stellen Sie den Confidence Threshold ein, bevor Sie die Geometrie analysieren. Niedrigere Werte zeigen mehr Vorhersagen, höhere Werte sind selektiver.
                </div>
              </div>
            )}
            <div className="space-y-2">
              <div className="space-y-1.5">
                <button
                  type="button"
                  disabled={!file || (loading && !isAnalyzing) || (geometryLocked && geometryResult)}
                  onClick={analyzeGeometry}
                  className={`w-full rounded-xl border bg-white/5 px-4 py-3 text-[11px] uppercase tracking-[0.35em] transition-all duration-300 ${
                    isAnalyzing
                      ? "animate-pulse"
                      : !file || (loading && !isAnalyzing) || (geometryLocked && geometryResult)
                      ? "opacity-40 cursor-not-allowed"
                      : "hover:-translate-y-0.5 hover:shadow-[0_16px_40px_-16px_rgba(56,189,248,0.45)]"
                  }`}
                  style={{
                    color: "var(--accent)",
                    borderColor: geometryResult ? "var(--accent)" : "var(--accent-soft)",
                    boxShadow: geometryResult ? "0 12px 24px -18px rgba(56,189,248,0.55)" : "none",
                  }}
                >
                  <div className="flex items-center justify-center">
                    <span className="tracking-[0.35em]">
                      {isAnalyzing ? "Analysiere..." : GEOMETRY_BUTTON}
                    </span>
                  </div>
                </button>
                {geometryResult?.model_id && (
                  <div className="pl-1 text-[10px] font-mono uppercase tracking-[0.2em] text-white/60">
                    {geometryResult.model_id}
                  </div>
                )}
              </div>
            </div>
          </GlassPanel>
          {(geometryResult?.total ?? 0) > 0 && (
            <GlassPanel className="p-4 space-y-2">
              <div className="text-sm font-medium">Erkannte Elemente</div>
              <div className="text-xs opacity-70">Total: {geometryResult?.total ?? 0}</div>
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
          {geometryResult?.predictions && geometryResult.predictions.length > 0 && (
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
              <button
                type="button"
                  disabled={createLoading || !geometryResult?.predictions?.length}
                onClick={createIfc}
                className={`w-full rounded-xl border bg-white/5 px-4 py-3 text-[11px] uppercase tracking-[0.35em] transition-all duration-300 ${
                  createLoading
                    ? "animate-pulse"
                    : !createLoading && geometryResult?.predictions?.length
                    ? "hover:-translate-y-0.5 hover:shadow-[0_16px_40px_-16px_rgba(56,189,248,0.45)]"
                    : "opacity-40 cursor-not-allowed"
                }`}
                style={{
                  color: "var(--accent)",
                  borderColor: "var(--accent-soft)",
                }}
              >
                {createLoading
                  ? `IFC wird erzeugt${ifcJobProgress > 0 ? ` (${ifcJobProgress}%)` : "..."}`
                  : "IFC erzeugen"}
              </button>
              {/* ============================================================================ */}
              {/* IFC EXPORT V2 - NEW IMPLEMENTATION BUTTON */}
              {/* ============================================================================ */}
              {/* This is a completely new IFC export implementation (V2). */}
              {/* It uses a different logic path than the original IFC export. */}
              {/* ============================================================================ */}
              <button
                type="button"
                disabled={createV2Loading || !geometryResult?.predictions?.length}
                onClick={createIfcV2}
                className={`w-full rounded-xl border bg-white/5 px-4 py-3 text-[11px] uppercase tracking-[0.35em] transition-all duration-300 ${
                  createV2Loading
                    ? "animate-pulse"
                    : !createV2Loading && geometryResult?.predictions?.length
                    ? "hover:-translate-y-0.5 hover:shadow-[0_16px_40px_-16px_rgba(139,92,246,0.45)]"
                    : "opacity-40 cursor-not-allowed"
                }`}
                style={{
                  color: "var(--accent)",
                  borderColor: "rgba(139,92,246,0.4)",
                }}
              >
                {createV2Loading
                  ? `IFC V2 wird erzeugt${ifcV2JobProgress > 0 ? ` (${ifcV2JobProgress}%)` : "..."}`
                  : "IFC erzeugen V2"}
              </button>
              <button
                type="button"
                onClick={runRepairLevelOne}
                disabled={repairLoading || !canRepair}
                className={`w-full rounded-xl border bg-white/5 px-4 py-3 text-[11px] uppercase tracking-[0.35em] transition-all duration-300 ${
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
              {createLoading && ifcJobId && (
                <div className="text-xs" style={{ color: "var(--accent-soft)" }}>
                  Job {ifcJobId} · Fortschritt {ifcJobProgress}%
                </div>
              )}
              {createV2Loading && ifcV2JobId && (
                <div className="text-xs" style={{ color: "rgba(139,92,246,0.6)" }}>
                  V2 Job {ifcV2JobId} · Fortschritt {ifcV2JobProgress}%
                </div>
              )}
              {ifcV2Error && <div className="text-xs text-red-300">{ifcV2Error}</div>}
              {ifcV2Response?.warnings && ifcV2Response.warnings.length > 0 && (
                <div className="rounded-lg border border-amber-400/30 bg-amber-500/10 p-2 text-xs text-amber-200 space-y-1">
                  {ifcV2Response.warnings.map((warning, idx) => (
                    <div key={`v2-${warning}-${idx}`}>{warning}</div>
                  ))}
                </div>
              )}
              {repairLoading && repairJobId && (
                <div className="text-xs" style={{ color: "var(--accent-soft)" }}>
                  Repair-Job {repairJobId} · Fortschritt {repairJobProgress}%
                </div>
              )}
              {ifcResponse && (
                <div className="text-xs break-all" style={{ color: "var(--accent-soft)" }}>
                  Datei: {ifcResponse.file_name}
                </div>
              )}
              {ifcV2Response && (
                <div className="text-xs break-all" style={{ color: "rgba(139,92,246,0.6)" }}>
                  V2 Datei: {ifcV2Response.file_name}
                </div>
              )}
            </GlassPanel>
          )}
          {geometryResult?.zones && geometryResult.zones.length > 0 && (
            <GlassPanel className="p-4 space-y-3">
              <div className="text-sm font-medium">Zone Counts</div>
              <div className="grid gap-2 text-sm">
                {geometryResult.zones.map((zone) => (
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
          {geometryResult?.lines && geometryResult.lines.length > 0 && (
            <GlassPanel className="p-4 space-y-3">
              <div className="text-sm font-medium">Line Counts</div>
              <div className="grid gap-2 text-sm">
                {geometryResult.lines.map((line) => (
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
              <div className="space-y-4">
                {!previewUrl ? (
                  <div className="text-sm opacity-70">Bitte ein Bild auswählen und analysieren.</div>
                ) : geometryResult ? (
                  <div className="space-y-3 rounded-xl border border-white/10 bg-slate-900/40 p-3">
                    <div className="flex items-center justify-between gap-3">
                      <div className="text-sm font-medium">{GEOMETRY_TITLE}</div>
                      {geometryResult && (
                        <div className="text-xs font-mono opacity-75">
                          Schwelle {Number.isFinite(geometryResult.confidence ?? geometryConfidence) ? (geometryResult.confidence ?? geometryConfidence).toFixed(2) : "--"}
                        </div>
                      )}
                    </div>
                    {geometryResult?.model_id && (
                      <div className="text-[11px] font-mono uppercase tracking-[0.25em] text-white/60">
                        {geometryResult.model_id}
                      </div>
                    )}
                    <div
                      className={`rounded-2xl border border-white/10 bg-slate-950/40 p-4 shadow-[0_18px_38px_-24px_rgba(56,189,248,0.65)] backdrop-blur-sm transition-all duration-500 ${
                        controlsReady ? "translate-y-0 opacity-100" : "translate-y-3 opacity-0"
                      }`}
                    >
                      <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.35em] text-white/60">
                        <span>Toleranz</span>
                        <span className="font-mono text-sky-200">
                          {Math.round((geometryResult.confidence ?? geometryConfidence) * 100)}%
                        </span>
                      </div>
                      <div className="relative mt-4 h-2 rounded-full bg-slate-900/60">
                        <div
                          className="pointer-events-none absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-sky-500 via-cyan-400 to-emerald-400"
                          style={{ width: `${Math.max(0, Math.min(100, geometryConfidence * 100))}%` }}
                        />
                        <input
                          id="confidence-geometry"
                          type="range"
                          min={0}
                          max={1}
                          step={0.01}
                          value={geometryConfidence}
                          onChange={(e) => setGeometryConfidence(parseFloat(e.target.value))}
                          className="glass-range absolute inset-x-0 -top-2 h-6 w-full"
                        />
                      </div>
                      <div className="mt-4 flex flex-wrap items-center gap-2">
                        <button
                          type="button"
                          aria-pressed={drawLabels}
                          onClick={() => setDrawLabels(!drawLabels)}
                          className={`inline-flex items-center justify-center rounded-full px-3 py-1 text-[10px] uppercase tracking-[0.35em] transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                            drawLabels
                              ? "border border-sky-400/40 bg-sky-500/20 text-sky-100 shadow-[0_14px_30px_-20px_rgba(56,189,248,0.75)]"
                              : "border border-white/10 bg-white/5 text-white/60 hover:bg-white/10 hover:text-white"
                          }`}
                        >
                          Labels
                        </button>
                        <button
                          type="button"
                          aria-pressed={drawShapes}
                          onClick={() => setDrawShapes(!drawShapes)}
                          className={`inline-flex items-center justify-center rounded-full px-3 py-1 text-[10px] uppercase tracking-[0.35em] transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 ${
                            drawShapes
                              ? "border border-emerald-400/40 bg-emerald-500/20 text-emerald-100 shadow-[0_14px_30px_-20px_rgba(16,185,129,0.7)]"
                              : "border border-white/10 bg-white/5 text-white/60 hover:bg-white/10 hover:text-white"
                          }`}
                        >
                          Shapes
                        </button>
                        <button
                          type="button"
                          aria-pressed={showSensors}
                          onClick={() => setShowSensors(!showSensors)}
                          className={`inline-flex items-center justify-center rounded-full px-3 py-1 text-[10px] uppercase tracking-[0.35em] transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-400 ${
                            showSensors
                              ? "border border-amber-400/50 bg-amber-400/20 text-amber-100 shadow-[0_14px_30px_-20px_rgba(251,191,36,0.6)]"
                              : "border border-white/10 bg-white/5 text-white/60 hover:bg-white/10 hover:text-white"
                          }`}
                        >
                          Sensoren
                        </button>
                        {geometryResult &&
                          analyzedConfidence !== null &&
                          Math.abs(geometryConfidence - analyzedConfidence) > 0.001 && (
                            <button
                              type="button"
                              onClick={() => {
                                setGeometryLocked(false);
                                analyzeGeometry();
                              }}
                              disabled={isAnalyzing}
                              className="inline-flex items-center justify-center rounded-full border border-cyan-400/50 bg-cyan-500/20 px-3 py-1 text-[10px] uppercase tracking-[0.35em] text-cyan-100 shadow-[0_14px_30px_-20px_rgba(6,182,212,0.7)] transition-all duration-200 hover:bg-cyan-500/30 hover:shadow-[0_16px_32px_-20px_rgba(6,182,212,0.8)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                              {isAnalyzing ? "Analysiere..." : "Neu analysieren"}
                            </button>
                          )}
                      </div>
                    </div>
                    <div className="relative group">
                      <PredictionViewer
                        imageSrc={previewUrl}
                        imageMeta={geometryResult?.image}
                        predictions={geometryResult?.predictions ?? []}
                        drawLabels={drawLabels}
                        drawShapes={drawShapes}
                        showSensors={showSensors}
                        threshold={geometryResult.confidence ?? geometryConfidence}
                        scaleToolActive={scaleToolActive}
                        calibration={calibration}
                        onCalibrationComplete={handleCalibrationComplete}
                        onCalibrationCancel={() => setScaleToolActive(false)}
                      />
                      <button
                        type="button"
                        onClick={() => geometryResult && previewUrl && setModalOpen(true)}
                        className="absolute right-4 top-4 z-20 rounded-full border border-white/20 bg-slate-950/60 px-3 py-1 text-[10px] uppercase tracking-[0.3em] text-white/70 opacity-0 shadow-[0_12px_32px_-12px_rgba(14,165,233,0.55)] transition-all duration-300 group-hover:opacity-100 hover:text-white hover:bg-slate-950/80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 focus-visible:opacity-100"
                        aria-label={`${GEOMETRY_TITLE} vergrößern`}
                      >
                        Vergrößern
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="flex h-56 items-center justify-center rounded-lg border border-dashed border-white/10 text-xs opacity-60">
                    Noch nicht analysiert.
                  </div>
                )}
              </div>
            )}

            {activeTab === "annotated" && (
              <div className="space-y-3">
                <div className="flex items-center justify-center">
                  {geometryResult?.annotated_image ? (
                    <img
                      src={`data:image/png;base64,${geometryResult.annotated_image}`}
                      alt={`${GEOMETRY_TITLE} Annotated`}
                      className="max-h-[600px] w-full rounded-xl border border-white/10 object-contain"
                    />
                  ) : (
                    <div className="text-sm opacity-70">
                      Noch keine annotierte Darstellung für {GEOMETRY_TITLE}.
                    </div>
                  )}
                </div>
              </div>
            )}

            {activeTab === "json" && (
              <div className="space-y-3">
                <div className="max-h-[480px] overflow-auto rounded-lg bg-black/40 p-3 text-xs leading-relaxed">
                  {geometryResult ? (
                    <pre className="whitespace-pre-wrap break-words">
                      {JSON.stringify(geometryResult.raw, null, 2)}
                    </pre>
                  ) : (
                    <span className="opacity-70">
                      Noch keine Analyse für {GEOMETRY_TITLE} durchgeführt.
                    </span>
                  )}
                </div>
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
        {/* ============================================================================ */}
        {/* IFC EXPORT V2 - NEW IMPLEMENTATION VIEWER */}
        {/* ============================================================================ */}
        {/* This is a completely new IFC export implementation (V2). */}
        {/* It uses a different logic path than the original IFC export. */}
        {/* ============================================================================ */}
        {hasIfcV2 && (
          <GlassPanel className="space-y-4 p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="text-sm font-medium">Modell Viewer V2</div>
                <div className="text-xs uppercase tracking-[0.3em] text-white/40">IFC Artefakte V2 - NEW IMPLEMENTATION</div>
              </div>
              <div className="flex flex-wrap gap-3 text-xs">
                {ifcV2BaseUrl && (
                  <a href={ifcV2BaseUrl} target="_blank" rel="noreferrer" className="text-accent hover:underline">
                    IFC V2 herunterladen
                  </a>
                )}
                {ifcV2ViewerUrl && (
                  <a href={ifcV2ViewerUrl} target="_blank" rel="noreferrer" className="text-accent hover:underline">
                    Viewer (bereinigte Predictions)
                  </a>
                )}
              </div>
            </div>
            <div className="grid gap-4 lg:grid-cols-2">
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs uppercase tracking-[0.3em] text-white/50">
                  <span>Top View V2 (2D)</span>
                  {topViewV2Error && <span className="text-[10px] normal-case tracking-normal text-rose-300/90">{topViewV2Error}</span>}
                </div>
                <IfcTopView topviewUrl={ifcV2TopViewUrl ?? undefined} height="clamp(20rem, 60vh, 32rem)" />
              </div>
              <div className="space-y-2">
                <div className="text-xs uppercase tracking-[0.3em] text-white/50">IFC V2 (3D)</div>
                <IfcJsViewerClient ifcUrl={ifcV2BaseUrl} height="clamp(20rem, 60vh, 32rem)" />
              </div>
            </div>
          </GlassPanel>
        )}
        </div>
      </div>

      <Modal open={modalOpen} onClose={() => setModalOpen(false)}>
        {geometryResult && previewUrl && (
          <div className="flex flex-col gap-6 p-6 md:p-10">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <div className="text-[11px] uppercase tracking-[0.35em] text-sky-200/80">
                  {GEOMETRY_TITLE}
                </div>
                <div className="mt-1 text-sm text-white/80">{geometryResult.model_id}</div>
              </div>
              <button
                type="button"
                onClick={() => setModalOpen(false)}
                className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-white/10 px-4 py-2 text-[10px] uppercase tracking-[0.35em] text-white/70 transition duration-200 hover:bg-white/20 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400"
              >
                Schließen
              </button>
            </div>
            <div
              className="rounded-2xl border border-white/10 bg-slate-950/40 p-4 shadow-[0_18px_38px_-24px_rgba(56,189,248,0.65)] backdrop-blur-sm"
            >
              <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.35em] text-white/60">
                <span>Toleranz</span>
                <span className="font-mono text-sky-200">
                  {Math.round((geometryResult.confidence ?? geometryConfidence) * 100)}%
                </span>
              </div>
              <div className="relative mt-4 h-2 rounded-full bg-slate-900/60">
                <div
                  className="pointer-events-none absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-sky-500 via-cyan-400 to-emerald-400"
                  style={{
                    width: `${Math.max(0, Math.min(100, geometryConfidence * 100))}%`,
                  }}
                />
                <input
                  id="modal-confidence-geometry"
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={geometryConfidence}
                  onChange={(e) => setGeometryConfidence(parseFloat(e.target.value))}
                  className="glass-range absolute inset-x-0 -top-2 h-6 w-full"
                />
              </div>
              <div className="mt-4 flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  aria-pressed={drawLabels}
                  onClick={() => setDrawLabels(!drawLabels)}
                  className={`inline-flex items-center justify-center rounded-full px-3 py-1 text-[10px] uppercase tracking-[0.35em] transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 ${
                    drawLabels
                      ? "border border-sky-400/40 bg-sky-500/20 text-sky-100 shadow-[0_14px_30px_-20px_rgba(56,189,248,0.75)]"
                      : "border border-white/10 bg-white/5 text-white/60 hover:bg-white/10 hover:text-white"
                  }`}
                >
                  Labels
                </button>
                <button
                  type="button"
                  aria-pressed={drawShapes}
                  onClick={() => setDrawShapes(!drawShapes)}
                  className={`inline-flex items-center justify-center rounded-full px-3 py-1 text-[10px] uppercase tracking-[0.35em] transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 ${
                    drawShapes
                      ? "border border-emerald-400/40 bg-emerald-500/20 text-emerald-100 shadow-[0_14px_30px_-20px_rgba(16,185,129,0.7)]"
                      : "border border-white/10 bg-white/5 text-white/60 hover:bg-white/10 hover:text-white"
                  }`}
                >
                  Shapes
                </button>
                <button
                  type="button"
                  aria-pressed={showSensors}
                  onClick={() => setShowSensors(!showSensors)}
                  className={`inline-flex items-center justify-center rounded-full px-3 py-1 text-[10px] uppercase tracking-[0.35em] transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-400 ${
                    showSensors
                      ? "border border-amber-400/50 bg-amber-400/20 text-amber-100 shadow-[0_14px_30px_-20px_rgba(251,191,36,0.6)]"
                      : "border border-white/10 bg-white/5 text-white/60 hover:bg-white/10 hover:text-white"
                  }`}
                >
                  Sensoren
                </button>
                {geometryResult &&
                  analyzedConfidence !== null &&
                  Math.abs(geometryConfidence - analyzedConfidence) > 0.001 && (
                    <button
                      type="button"
                      onClick={() => {
                        setModalOpen(false);
                        setGeometryLocked(false);
                        analyzeGeometry();
                      }}
                      disabled={isAnalyzing}
                      className="inline-flex items-center justify-center rounded-full border border-cyan-400/50 bg-cyan-500/20 px-3 py-1 text-[10px] uppercase tracking-[0.35em] text-cyan-100 shadow-[0_14px_30px_-20px_rgba(6,182,212,0.7)] transition-all duration-200 hover:bg-cyan-500/30 hover:shadow-[0_16px_32px_-20px_rgba(6,182,212,0.8)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isAnalyzing ? "Analysiere..." : "Neu analysieren"}
                    </button>
                  )}
              </div>
            </div>
            <div className="rounded-3xl border border-white/10 bg-slate-950/40 p-4 shadow-[inset_0_1px_0_rgba(148,163,184,0.12)]">
              <PredictionViewer
                imageSrc={previewUrl}
                imageMeta={geometryResult.image}
                predictions={geometryResult.predictions ?? []}
                drawLabels={drawLabels}
                drawShapes={drawShapes}
                showSensors={showSensors}
                threshold={geometryConfidence}
                scaleToolActive={false}
                calibration={calibration}
              />
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
