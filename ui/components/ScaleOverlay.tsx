'use client';

import { useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

export type ScaleCalibration = {
  pxPerMm: number;
  pixelDistance: number;
  realDistanceMm: number;
  points: [[number, number], [number, number]];
  unit: "mm" | "m";
};

type Props = {
  width: number;
  height: number;
  imageSrc?: string;
  onComplete: (calibration: ScaleCalibration) => void;
  onCancel: () => void;
};

type InteractionState = {
  start: [number, number] | null;
  current: [number, number] | null;
};

const ESCAPE_KEY = "Escape";
const MAGNIFIER_SIZE = 120;

export default function ScaleOverlay({ width, height, imageSrc, onComplete, onCancel }: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [interaction, setInteraction] = useState<InteractionState>({ start: null, current: null });
  const [dialogOpen, setDialogOpen] = useState(false);
  const [realValue, setRealValue] = useState<string>("1000");
  const [unit, setUnit] = useState<"mm" | "m">("mm");
  const [pendingPoints, setPendingPoints] = useState<[[number, number], [number, number]] | null>(null);
  const [hoverPoint, setHoverPoint] = useState<[number, number] | null>(null);
  const [hoverMeta, setHoverMeta] = useState<{
    screen: [number, number];
    rect: { width: number; height: number };
  } | null>(null);

  const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

  const pixelDistance = useMemo(() => {
    if (!pendingPoints) return 0;
    const [[x1, y1], [x2, y2]] = pendingPoints;
    const dx = x2 - x1;
    const dy = y2 - y1;
    return Math.hypot(dx, dy);
  }, [pendingPoints]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === ESCAPE_KEY) {
        event.preventDefault();
        event.stopPropagation();
        resetInteraction();
        onCancel();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onCancel]);

  const handlePointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
    const { clientX, clientY } = event;
    const point = svgPoint(event);
    if (!point) return;
    setHoverPoint(point);
    const svg = svgRef.current;
    if (svg) {
      const rect = svg.getBoundingClientRect();
      setHoverMeta({
        screen: [clientX - rect.left, clientY - rect.top],
        rect: { width: rect.width, height: rect.height },
      });
    }
    if (!interaction.start) return;
    setInteraction((prev) => ({ ...prev, current: point }));
  };

  const handlePointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
    const { clientX, clientY } = event;
    const point = svgPoint(event);
    if (!point) return;
    setHoverPoint(point);
    const svg = svgRef.current;
    if (svg) {
      const rect = svg.getBoundingClientRect();
      setHoverMeta({
        screen: [clientX - rect.left, clientY - rect.top],
        rect: { width: rect.width, height: rect.height },
      });
    }
    setInteraction((prev) => {
      if (!prev.start) {
        return { start: point, current: point };
      }
      const finished: [[number, number], [number, number]] = [prev.start, point];
      setPendingPoints(finished);
      setDialogOpen(true);
      return { start: prev.start, current: point };
    });
  };

  const svgPoint = (event: ReactPointerEvent<SVGSVGElement>): [number, number] | null => {
    const svg = svgRef.current;
    if (!svg) return null;
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return null;
    const transformed = point.matrixTransform(ctm.inverse());
    return [transformed.x, transformed.y];
  };

  const resetInteraction = () => {
    setInteraction({ start: null, current: null });
    setDialogOpen(false);
    setPendingPoints(null);
    setRealValue("1000");
    setUnit("mm");
    setHoverMeta(null);
  };

  const submitCalibration = (event: React.FormEvent) => {
    event.preventDefault();
    if (!pendingPoints) return;
    const parsed = Number(realValue);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      return;
    }
    const realDistanceMm = unit === "mm" ? parsed : parsed * 1000;
    const calibration: ScaleCalibration = {
      pxPerMm: pixelDistance / realDistanceMm,
      pixelDistance,
      realDistanceMm,
      points: pendingPoints,
      unit,
    };
    resetInteraction();
    onComplete(calibration);
  };

  return (
    <div className="absolute inset-0 cursor-crosshair">
      <svg
        ref={svgRef}
        className="absolute inset-0 h-full w-full cursor-crosshair"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerLeave={() => {
          setHoverPoint(null);
          setHoverMeta(null);
        }}
      >
        <defs>
          <marker id="scale-overlay-arrow" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto-start-reverse">
            <path d="M0,0 L6,3 L0,6 z" fill="var(--accent)" />
          </marker>
        </defs>
        {interaction.start && interaction.current && (
          <g>
            <line
              x1={interaction.start[0]}
              y1={interaction.start[1]}
              x2={interaction.current[0]}
              y2={interaction.current[1]}
              stroke="var(--accent)"
              strokeWidth={2}
              strokeDasharray="6 8"
              markerStart="url(#scale-overlay-arrow)"
              markerEnd="url(#scale-overlay-arrow)"
            />
            <circle cx={interaction.start[0]} cy={interaction.start[1]} r={6} fill="var(--accent)" />
            <circle cx={interaction.current[0]} cy={interaction.current[1]} r={6} fill="var(--accent)" />
          </g>
        )}
      </svg>

      {hoverPoint && hoverMeta && imageSrc && width > 0 && height > 0 && (() => {
        const displayWidth = hoverMeta.rect.width || width;
        const displayHeight = hoverMeta.rect.height || height;
        const screenX = hoverMeta.screen[0];
        const screenY = hoverMeta.screen[1];
        const clampedX = clamp(screenX, MAGNIFIER_SIZE / 2, Math.max(displayWidth - MAGNIFIER_SIZE / 2, MAGNIFIER_SIZE / 2));
        const clampedY = clamp(screenY, MAGNIFIER_SIZE / 2, Math.max(displayHeight - MAGNIFIER_SIZE / 2, MAGNIFIER_SIZE / 2));
        return (
          <div
            className="pointer-events-none absolute z-20 rounded-xl border border-white/20 bg-black/60 shadow-lg backdrop-blur"
            style={{
              width: MAGNIFIER_SIZE,
              height: MAGNIFIER_SIZE,
              left: `${clampedX - MAGNIFIER_SIZE / 2}px`,
              top: `${clampedY - MAGNIFIER_SIZE / 2}px`,
              backgroundImage: `url(${imageSrc})`,
              backgroundSize: `${displayWidth * 2}px ${displayHeight * 2}px`,
              backgroundPosition: `${-(screenX * 2 - MAGNIFIER_SIZE / 2)}px ${-(screenY * 2 - MAGNIFIER_SIZE / 2)}px`,
            }}
          >
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="h-full w-px bg-white/30" />
              <div className="h-full w-full">
                <div className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-white/30" />
              </div>
            </div>
          </div>
        );
      })()}

      {dialogOpen && pendingPoints && (
        <div className="absolute left-1/2 top-1/2 z-20 w-64 -translate-x-1/2 -translate-y-1/2 rounded-xl border border-white/15 bg-black/80 p-4 shadow-xl backdrop-blur-xl">
          <div className="text-sm font-medium">Maßstab festlegen</div>
          <div className="mt-1 text-xs opacity-70">Pixel-Distanz: {pixelDistance.toFixed(2)} px</div>
          <form className="mt-3 space-y-3" onSubmit={submitCalibration}>
            <label className="grid gap-1 text-sm">
              <span className="text-xs uppercase tracking-[0.3em] opacity-60">Reale Distanz</span>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min={0.001}
                  step={0.001}
                  value={realValue}
                  onChange={(event) => setRealValue(event.target.value)}
                  className="w-full rounded-lg border border-white/15 bg-white/5 px-3 py-2 text-sm focus-ring"
                  autoFocus
                />
                <select
                  value={unit}
                  onChange={(event) => setUnit(event.target.value as "mm" | "m")}
                  className="rounded-lg border border-white/15 bg-white/5 px-2 py-2 text-sm focus-ring"
                >
                  <option value="mm">mm</option>
                  <option value="m">m</option>
                </select>
              </div>
            </label>
            <div className="flex items-center justify-end gap-2 text-sm">
              <button
                type="button"
                onClick={() => {
                  resetInteraction();
                  onCancel();
                }}
                className="rounded-lg px-3 py-2 transition hover:bg-white/10"
              >
                Abbrechen
              </button>
              <button
                type="submit"
                className="rounded-lg bg-accent px-3 py-2 font-medium text-slate-950 transition hover:bg-accent/90"
              >
                Speichern
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
