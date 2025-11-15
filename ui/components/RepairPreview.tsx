'use client';

import { useEffect, useMemo, useState } from "react";
import CanvasGeoViewer, { type CanvasLayer } from "./CanvasGeoViewer";
import { repairIfcPreviewAsync, repairIfcCommit, getJob, type IfcRepairPreviewResponse, type JobStatusResponse } from "../lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const POLL_INTERVAL_MS = 2000;
const POLL_TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes

const delay = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

type Feature = { type: "Feature"; properties?: Record<string, any> | null; geometry: any };
type FeatureCollection = { type: "FeatureCollection"; features: Feature[] };

type Props = {
  fileName?: string | null;
  ifcUrl?: string | null;
  jobId?: string | null;
  imageUrl?: string | null;
  onApplied?: (payload: { ifc_url: string; file_name: string; topview_url?: string | null }) => void;
};

export default function RepairPreview({ fileName, ifcUrl, jobId, imageUrl, onApplied }: Props) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<IfcRepairPreviewResponse | null>(null);
  const [overlay, setOverlay] = useState<FeatureCollection | null>(null);
  const [activeTool, setActiveTool] = useState<"pan" | "select">("pan");
  const [applying, setApplying] = useState(false);
  const [previewJobId, setPreviewJobId] = useState<string | null>(null);
  const [previewProgress, setPreviewProgress] = useState(0);

  useEffect(() => {
    const run = async () => {
      setLoading(true);
      setError(null);
      setData(null);
      setOverlay(null);
      setPreviewJobId(null);
      setPreviewProgress(0);
      
      try {
        console.log("[RepairPreview] Starte Preview-Request (Async)", { fileName, ifcUrl, jobId, imageUrl });
        const jobResponse = await repairIfcPreviewAsync({
          level: 1,
          file_name: fileName ?? undefined,
          ifc_url: ifcUrl ?? undefined,
          job_id: jobId ?? undefined,
          image_url: imageUrl ?? undefined,
        });
        console.log("[RepairPreview] Job erstellt", { job_id: jobResponse.job_id });
        setPreviewJobId(jobResponse.job_id);
        setPreviewProgress(5);

        const maxAttempts = Math.ceil(POLL_TIMEOUT_MS / POLL_INTERVAL_MS);
        let jobResult: IfcRepairPreviewResponse | null = null;
        for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
          if (attempt > 0) {
            await delay(POLL_INTERVAL_MS);
          }

          let status: JobStatusResponse;
          try {
            status = await getJob(jobResponse.job_id);
          } catch (pollErr: any) {
            console.warn("[RepairPreview] getJob polling failed", pollErr);
            if (attempt === maxAttempts - 1) {
              throw pollErr;
            }
            continue;
          }

          setPreviewProgress(typeof status.progress === "number" ? status.progress : 0);
          if (status.status === "succeeded") {
            if (!status.result) {
              throw new Error("Preview-Job abgeschlossen, aber kein Ergebnis erhalten.");
            }
            // Type guard to ensure it's IfcRepairPreviewResponse
            const previewResult = status.result as IfcRepairPreviewResponse;
            if (!previewResult.preview_id) {
              throw new Error("Preview-Ergebnis hat ungültiges Format.");
            }
            jobResult = previewResult;
            setPreviewProgress(100);
            break;
          }

          if (status.status === "failed") {
            const message = status.error ? String(status.error) : "Preview-Erstellung fehlgeschlagen";
            throw new Error(message);
          }
        }

        if (!jobResult) {
          throw new Error("Preview-Job hat die maximale Wartezeit überschritten.");
        }

        console.log("[RepairPreview] Preview-Response erhalten", { 
          preview_id: jobResult.preview_id, 
          overlay_url: jobResult.overlay_url, 
          metrics: jobResult.metrics,
          estimated_seconds: jobResult.estimated_seconds 
        });
        setData(jobResult);
        
        if (jobResult.overlay_url) {
          try {
            const overlayUrl = jobResult.overlay_url.startsWith("http") ? jobResult.overlay_url : `${API_BASE}${jobResult.overlay_url}`;
            console.log("[RepairPreview] Lade Overlay von", overlayUrl);
            
            // Add timeout for overlay fetch (10 seconds)
            const overlayController = new AbortController();
            const overlayTimeout = setTimeout(() => overlayController.abort(), 10000);
            
            const res = await fetch(overlayUrl, { 
              cache: "no-store",
              signal: overlayController.signal,
            });
            clearTimeout(overlayTimeout);
            
            if (res.ok) {
              const json = (await res.json()) as FeatureCollection;
              console.log("[RepairPreview] Overlay geladen", { 
                featureCount: json.features?.length || 0,
                hasFeatures: Boolean(json.features && json.features.length > 0)
              });
              
              // Validate overlay structure
              if (!json || typeof json !== "object") {
                throw new Error("Overlay-Daten haben ungültiges Format");
              }
              if (!json.features || !Array.isArray(json.features)) {
                throw new Error("Overlay enthält keine 'features' Array");
              }
              
              setOverlay(json);
            } else {
              const errorText = await res.text().catch(() => res.statusText);
              console.error(`[RepairPreview] Overlay fetch failed: ${res.status} ${res.statusText}`, errorText);
              setError(`Overlay konnte nicht geladen werden (${res.status} ${res.statusText}). ${errorText.substring(0, 200)}`);
            }
          } catch (fetchErr: any) {
            if (fetchErr.name === "AbortError") {
              console.error("[RepairPreview] Overlay fetch timeout");
              setError("Overlay-Laden hat zu lange gedauert (>10s). Die Datei könnte zu groß sein.");
            } else {
              console.error("[RepairPreview] Overlay fetch error:", fetchErr);
              setError(`Overlay konnte nicht geladen werden: ${fetchErr?.message || "Netzwerkfehler"}`);
            }
          }
        } else {
          console.warn("[RepairPreview] Keine overlay_url in Response", jobResult);
          setError("Keine Overlay-URL in der Server-Antwort erhalten.");
        }
      } catch (err: any) {
        console.error("[RepairPreview] Preview-Request fehlgeschlagen:", err);
        const errorMessage = err?.message || "Vorschau fehlgeschlagen.";
        // Parse error details if available
        let detailedError = errorMessage;
        try {
          if (errorMessage.includes(":")) {
            const parts = errorMessage.split(":");
            if (parts.length > 1) {
              detailedError = parts.slice(1).join(":").trim();
            }
          }
        } catch {
          // Keep original error message
        }
        setError(detailedError);
      } finally {
        setLoading(false);
        setPreviewJobId(null);
        setPreviewProgress(0);
      }
    };
    
    if (fileName || ifcUrl) run();
  }, [fileName, ifcUrl, jobId, imageUrl]);

  const layers = useMemo<CanvasLayer[]>(() => {
    if (!overlay) return [];
    
    // Validate overlay structure
    if (!overlay.features || !Array.isArray(overlay.features)) {
      console.warn("[RepairPreview] Overlay hat keine features Array", overlay);
      return [];
    }
    
    const walls: Feature[] = [];
    const axes: Feature[] = [];
    for (const f of overlay.features) {
      if (!f || typeof f !== "object") continue;
      const t = String(f.properties?.type || "");
      if (t === "WALL_SRC") walls.push(f);
      else if (t === "WALL_AXIS") axes.push(f);
    }
    
    console.log("[RepairPreview] Layers erstellt", { walls: walls.length, axes: axes.length });
    
    return [
      { id: "walls", label: "Wände (Quelle)", data: { type: "FeatureCollection", features: walls }, fill: "rgba(56,189,248,0.12)", stroke: "rgba(56,189,248,0.9)", lineWidth: 1.5, zIndex: 1 },
      { id: "axes", label: "Achsen (snapped)", data: { type: "FeatureCollection", features: axes }, stroke: "rgba(236,72,153,0.9)", lineWidth: 2.2, zIndex: 2 },
    ];
  }, [overlay]);

  const handleApply = async () => {
    if (!data?.preview_id) return;
    setApplying(true);
    setError(null);
    try {
      const out = await repairIfcCommit({ preview_id: data.preview_id, level: 1 });
      onApplied?.({ ifc_url: out.ifc_url, file_name: out.file_name, topview_url: out.topview_url });
    } catch (err: any) {
      setError(err?.message || "Übernehmen fehlgeschlagen.");
    } finally {
      setApplying(false);
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-[0.3em] text-white/50">Level 1 Vorschau</div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setActiveTool("pan")}
            className={`rounded-md border px-2 py-1 text-[11px] uppercase tracking-[0.25em] ${activeTool === "pan" ? "border-accent text-accent" : "border-white/20 text-white/60"}`}
          >
            Pan
          </button>
          <button
            type="button"
            onClick={() => setActiveTool("select")}
            className={`rounded-md border px-2 py-1 text-[11px] uppercase tracking-[0.25em] ${activeTool === "select" ? "border-accent text-accent" : "border-white/20 text-white/60"}`}
          >
            Select
          </button>
        </div>
      </div>
      <div className="rounded-xl border border-white/10 bg-slate-950/70 p-2">
        {loading ? (
          <div className="p-8 text-center text-sm opacity-70 space-y-2">
            <div>Vorschau wird erstellt…</div>
            {previewProgress > 0 && (
              <div className="text-xs opacity-50 mt-2">
                Fortschritt: {previewProgress}%
              </div>
            )}
            {previewJobId && (
              <div className="text-xs opacity-40 mt-1">
                Job {previewJobId.substring(0, 8)}…
              </div>
            )}
          </div>
        ) : error ? (
          <div className="p-8 text-center text-sm text-red-300 space-y-2">
            <div>{error}</div>
            {data && !data.overlay_url && (
              <div className="text-xs text-red-200/70 mt-2">
                Hinweis: Server hat keine Overlay-URL zurückgegeben. Möglicherweise enthält die IFC-Datei keine Wände.
              </div>
            )}
          </div>
        ) : layers.length ? (
          <CanvasGeoViewer layers={layers} activeTool={activeTool} height={420} />
        ) : data && data.overlay_url ? (
          <div className="p-8 text-center text-sm opacity-70 space-y-2">
            <div>Overlay geladen, aber keine sichtbaren Features gefunden.</div>
            {data.metrics && (
              <div className="text-xs opacity-60 mt-2">
                Wände: {data.metrics.total_walls_src ?? 0} · Achsen: {data.metrics.total_axes ?? 0}
              </div>
            )}
          </div>
        ) : (
          <div className="p-8 text-center text-sm opacity-70">Keine Vorschau verfügbar.</div>
        )}
      </div>
      <div className="flex items-center justify-between">
        <div className="text-[11px] text-white/50">
          {data?.metrics && (
            <span>
              Wände: {String(data.metrics.total_walls_src ?? 0)} · Achsen: {String(data.metrics.total_axes ?? 0)}
              {data.metrics.median_iou != null ? ` · Median IoU: ${Number(data.metrics.median_iou).toFixed(3)}` : ""}
            </span>
          )}
        </div>
        <button
          type="button"
          onClick={handleApply}
          disabled={!data?.preview_id || applying}
          className="rounded-xl border border-accent/40 bg-accent/10 px-4 py-2 text-[11px] uppercase tracking-[0.3em] text-accent transition hover:bg-accent/20 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {applying ? "Übernehmen…" : "Übernehmen"}
        </button>
      </div>
    </div>
  );
}


