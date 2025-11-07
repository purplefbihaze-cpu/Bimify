'use client';

import { useEffect, useMemo, useState } from "react";

import CanvasGeoViewer, { type CanvasLayer } from "./CanvasGeoViewer";

type Feature = {
  type: "Feature";
  properties?: Record<string, any> | null;
  geometry: {
    type: string;
    coordinates: any;
  } | null;
};

type FeatureCollection = {
  type: "FeatureCollection";
  features: Feature[];
};

type Props = {
  topviewUrl?: string | null;
  height?: number | string;
  onSelectProduct?: (productId: number | null, guid?: string | null) => void;
};

type LoadState = "idle" | "loading" | "ready" | "error";

const DEFAULT_HEIGHT = "clamp(22rem, 60vh, 34rem)";

export default function IfcTopView({ topviewUrl, height = DEFAULT_HEIGHT, onSelectProduct }: Props) {
  const [data, setData] = useState<FeatureCollection | null>(null);
  const [status, setStatus] = useState<LoadState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [activeTool, setActiveTool] = useState<"pan" | "select">("pan");

  useEffect(() => {
    let disposed = false;
    const url = typeof topviewUrl === "string" ? topviewUrl.trim() : "";
    if (!url) {
      setStatus("idle");
      setData(null);
      setError(null);
      return;
    }

    const load = async () => {
      setStatus("loading");
      setError(null);
      try {
        const response = await fetch(url, { cache: "no-store" });
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        const json = (await response.json()) as FeatureCollection;
        if (!disposed) {
          setData(json);
          setStatus("ready");
        }
      } catch (err: any) {
        if (disposed) return;
        setStatus("error");
        setData(null);
        setError(err?.message ?? "TopView konnte nicht geladen werden.");
      }
    };

    load();

    return () => {
      disposed = true;
    };
  }, [topviewUrl]);

  const layers = useMemo<CanvasLayer[]>(() => {
    if (!data) return [];
    const grouped: Record<string, Feature[]> = { WALL: [], DOOR: [], WINDOW: [] };
    for (const feature of data.features || []) {
      const type = String(feature?.properties?.type || "").toUpperCase();
      if (type in grouped) {
        grouped[type].push(feature);
      }
    }
    return [
      {
        id: "walls",
        label: "Wände",
        data: {
          type: "FeatureCollection",
          features: grouped.WALL,
        },
        fill: "#0ea5e9",
        stroke: "#0369a1",
        opacity: 0.28,
        lineWidth: 1.2,
        zIndex: 1,
      },
      {
        id: "doors",
        label: "Türen",
        data: {
          type: "FeatureCollection",
          features: grouped.DOOR,
        },
        fill: "#22c55e22",
        stroke: "#22c55e",
        lineWidth: 1.6,
        zIndex: 2,
      },
      {
        id: "windows",
        label: "Fenster",
        data: {
          type: "FeatureCollection",
          features: grouped.WINDOW,
        },
        fill: "#38bdf822",
        stroke: "#38bdf8",
        lineWidth: 1.4,
        dash: [6, 4],
        zIndex: 3,
      },
    ];
  }, [data]);

  const handleSelect = (payload: { layerId: string; feature: Feature } | null) => {
    if (!onSelectProduct) return;
    if (!payload?.feature?.properties) {
      onSelectProduct(null, null);
      return;
    }
    const productId = payload.feature.properties.productId;
    const guid = payload.feature.properties.guid;
    onSelectProduct(typeof productId === "number" ? productId : null, typeof guid === "string" ? guid : null);
  };

  const overlayControls = (
    <div className="pointer-events-none absolute left-3 top-3 z-20 flex gap-2 text-xs">
      {(
        [
          { key: "pan" as const, label: "Pan" },
          { key: "select" as const, label: "Select" },
        ]
      ).map(({ key, label }) => (
        <button
          key={key}
          type="button"
          onClick={() => setActiveTool(key)}
          className={`pointer-events-auto rounded-lg border px-2 py-1 uppercase tracking-[0.25em] transition ${
            activeTool === key ? "border-accent text-accent" : "border-white/20 text-white/60 hover:border-white/40 hover:text-white/80"
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );

  return (
    <div className="relative">
      <CanvasGeoViewer
        layers={layers}
        height={height}
        activeTool={activeTool}
        className="rounded-xl border border-white/10 bg-slate-950/80"
        onSelect={handleSelect}
        overlay={overlayControls}
      />
      {status === "loading" && <StatusOverlay message="TopView wird geladen…" />}
      {status === "error" && <StatusOverlay message={error ?? "TopView konnte nicht geladen werden."} tone="error" />}
      {status === "idle" && <StatusOverlay message="Noch keine TopView verfügbar." tone="muted" />}
    </div>
  );
}


function StatusOverlay({ message, tone = "info" }: { message: string; tone?: "info" | "error" | "muted" }) {
  const toneColor = tone === "error" ? "#fca5a5" : tone === "muted" ? "#94a3b8" : "#e2e8f0";
  return (
    <div
      className="pointer-events-none absolute inset-0 flex items-center justify-center px-4 text-center text-xs"
      style={{
        background: "linear-gradient(180deg, rgba(8,15,29,0.85) 0%, rgba(8,13,25,0.9) 65%, rgba(10,16,28,0.92) 100%)",
        color: toneColor,
      }}
    >
      {message}
    </div>
  );
}

