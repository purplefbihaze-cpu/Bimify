'use client';

import { ReactNode, useEffect, useMemo, useRef, useState } from "react";

type Geometry = {
  type: string;
  coordinates: any;
};

type Feature = {
  type: "Feature";
  properties?: Record<string, any> | null;
  geometry: Geometry;
};

type FeatureCollection = {
  type: "FeatureCollection";
  features: Feature[];
};

export type CanvasLayer = {
  id: string;
  label?: string;
  data: FeatureCollection | null;
  stroke?: string;
  fill?: string;
  lineWidth?: number;
  dash?: number[];
  opacity?: number;
  visible?: boolean;
  zIndex?: number;
};

type PreparedFeature = {
  feature: Feature;
  path: Path2D;
};

type PreparedLayer = {
  layer: CanvasLayer;
  items: PreparedFeature[];
};

type Tool = "pan" | "select" | "measure";

type CanvasGeoViewerProps = {
  layers: CanvasLayer[];
  activeTool?: Tool;
  className?: string;
  height?: number | string;
  overlay?: ReactNode;
  onHover?: (payload: { layerId: string; feature: Feature } | null) => void;
  onSelect?: (payload: { layerId: string; feature: Feature } | null) => void;
  onMeasure?: (payload: { distance: number; points: [number, number][] }) => void;
};

type Transform = {
  scale: number;
  tx: number;
  ty: number;
};

const defaultTransform: Transform = { scale: 1, tx: 0, ty: 0 };

export default function CanvasGeoViewer({
  layers,
  activeTool = "pan",
  className = "",
  height = "clamp(22rem, 60vh, 34rem)",
  overlay,
  onHover,
  onSelect,
  onMeasure,
}: CanvasGeoViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fallbackHeight = typeof height === "number" ? height : 520;
  const [size, setSize] = useState({ width: 800, height: fallbackHeight });
  const [transform, setTransform] = useState<Transform>(defaultTransform);
  const fitTransformRef = useRef<Transform>(defaultTransform);
  const [measurePoints, setMeasurePoints] = useState<[number, number][]>([]);
  const [hovered, setHovered] = useState<{ layerId: string; feature: Feature } | null>(null);
  const [selected, setSelected] = useState<{ layerId: string; feature: Feature } | null>(null);

  const preparedLayers = useMemo<PreparedLayer[]>(() => {
    if (typeof window === "undefined" || typeof Path2D === "undefined") return [];
    return layers
      .filter((layer) => layer.visible !== false && layer.data && layer.data.features?.length)
      .sort((a, b) => (a.zIndex ?? 0) - (b.zIndex ?? 0))
      .map((layer) => ({
        layer,
        items: (layer.data?.features || []).map((feature) => ({ feature, path: featureToPath(feature) })).filter(Boolean) as PreparedFeature[],
      }));
  }, [layers]);

  const bounds = useMemo(() => computeBounds(preparedLayers), [preparedLayers]);

  useEffect(() => {
    if (activeTool !== "measure") setMeasurePoints([]);
  }, [activeTool]);

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setSize({ width: entry.contentRect.width, height: entry.contentRect.height || fallbackHeight });
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [fallbackHeight]);

  useEffect(() => {
    if (!bounds) return;
    const fitted = fitTransformToBounds(bounds, size);
    fitTransformRef.current = fitted;
    setTransform(fitted);
  }, [bounds?.hash, size.width, size.height]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, size.width * dpr);
    canvas.height = Math.max(1, size.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.width, size.height);
    ctx.save();
    ctx.translate(transform.tx, transform.ty);
    ctx.scale(transform.scale, transform.scale);

    for (const { layer, items } of preparedLayers) {
      if (!items.length) continue;
      ctx.save();
      if (layer.opacity != null) ctx.globalAlpha = layer.opacity;
      ctx.lineWidth = (layer.lineWidth ?? 1.6) / transform.scale;
      if (layer.dash) ctx.setLineDash(layer.dash);
      ctx.strokeStyle = layer.stroke ?? "rgba(255,255,255,0.7)";
      ctx.fillStyle = layer.fill ?? "rgba(255,255,255,0.05)";

      for (const item of items) {
        const { path } = item;
        const geometryType = item.feature.geometry?.type?.toLowerCase();
        if (geometryType && geometryType.includes("polygon")) {
          if (layer.fill) ctx.fill(path);
        }
        ctx.stroke(path);
      }
      ctx.restore();
    }

    if (hovered) {
      drawHighlight(ctx, preparedLayers, hovered, transform, { stroke: "var(--dynamic-accent)", lineWidth: 3 });
    }

    if (selected) {
      drawHighlight(ctx, preparedLayers, selected, transform, { stroke: "var(--accent)", lineWidth: 4, dash: [6, 6] });
    }

    if (measurePoints.length === 2) {
      drawMeasure(ctx, measurePoints, transform);
    }

    ctx.restore();
  }, [preparedLayers, transform, size, hovered, selected, measurePoints]);

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    let dragging = false;
    let last = { x: 0, y: 0 };

    const pointerDown = (event: PointerEvent) => {
      if (activeTool === "measure") {
        handleMeasureClick(event);
        return;
      }
      if (activeTool === "select") {
        handleSelect(event);
      }
      dragging = true;
      last = { x: event.clientX, y: event.clientY };
      canvas.setPointerCapture(event.pointerId);
    };

    const pointerMove = (event: PointerEvent) => {
      if (!canvasRef.current) return;
      if (dragging && activeTool === "pan") {
        setTransform((prev) => ({
          ...prev,
          tx: prev.tx + (event.clientX - last.x),
          ty: prev.ty + (event.clientY - last.y),
        }));
        last = { x: event.clientX, y: event.clientY };
      }
      handleHover(event);
    };

    const pointerUp = (event: PointerEvent) => {
      dragging = false;
      canvas.releasePointerCapture(event.pointerId);
    };

    const wheel = (event: WheelEvent) => {
      if (activeTool === "measure") return;
      event.preventDefault();
      const direction = event.deltaY > 0 ? -1 : 1;
      const factor = 1 + direction * 0.08;
      const rect = canvas.getBoundingClientRect();
      const point = { x: event.clientX - rect.left, y: event.clientY - rect.top };
      const minScale = fitTransformRef.current?.scale ?? defaultTransform.scale;
      setTransform((prev) => zoomAtPoint(prev, point, factor, minScale));
    };

    const handleHover = (event: PointerEvent) => {
      if (!canvasRef.current) return;
      const world = screenToWorld(event, canvasRef.current, transform);
      const hit = hitTest(preparedLayers, world, transform, canvasRef.current);
      setHovered(hit);
      if (onHover) onHover(hit);
    };

    const handleSelect = (event: PointerEvent) => {
      const world = screenToWorld(event, canvasRef.current!, transform);
      const hit = hitTest(preparedLayers, world, transform, canvasRef.current!);
      setSelected(hit);
      if (onSelect) onSelect(hit);
    };

    const handleMeasureClick = (event: PointerEvent) => {
      const world = screenToWorld(event, canvasRef.current!, transform);
      setMeasurePoints((prev) => {
        const next: [number, number][] = [...prev, [world.x, world.y]].slice(-2) as [number, number][];
        if (next.length === 2 && onMeasure) {
          const distance = Math.hypot(next[1][0] - next[0][0], next[1][1] - next[0][1]);
          onMeasure({ distance, points: next });
        }
        return next;
      });
    };

    const handleDoubleClick = (event: MouseEvent) => {
      event.preventDefault();
      setTransform(fitTransformRef.current ?? defaultTransform);
    };

    canvas.addEventListener("pointerdown", pointerDown);
    window.addEventListener("pointermove", pointerMove);
    window.addEventListener("pointerup", pointerUp);
    canvas.addEventListener("wheel", wheel, { passive: false });
    canvas.addEventListener("dblclick", handleDoubleClick);

    return () => {
      canvas.removeEventListener("pointerdown", pointerDown);
      window.removeEventListener("pointermove", pointerMove);
      window.removeEventListener("pointerup", pointerUp);
      canvas.removeEventListener("wheel", wheel);
      canvas.removeEventListener("dblclick", handleDoubleClick);
    };
  }, [activeTool, preparedLayers, transform, onHover, onSelect, onMeasure]);

  const handleReset = () => {
    setTransform(fitTransformRef.current ?? defaultTransform);
  };

  return (
    <div
      ref={containerRef}
      className={`relative w-full ${className}`}
      style={{ height: typeof height === "number" ? `${height}px` : height }}
    >
      <canvas ref={canvasRef} className="h-full w-full rounded-lg bg-black/30" />
      <button
        type="button"
        onClick={handleReset}
        className="absolute right-3 top-3 rounded-md bg-black/50 px-2 py-1 text-[11px] uppercase tracking-[0.2em] text-white/70 transition hover:bg-black/70"
      >
        Fit
      </button>
      {overlay}
    </div>
  );
}

function featureToPath(feature: Feature): Path2D {
  const path = new Path2D();
  const geometry = feature.geometry;
  if (!geometry) return path;
  switch (geometry.type) {
    case "Polygon":
      polygonToPath(path, geometry.coordinates);
      break;
    case "MultiPolygon":
      for (const polygon of geometry.coordinates) {
        polygonToPath(path, polygon);
      }
      break;
    case "LineString":
      lineToPath(path, geometry.coordinates);
      break;
    case "MultiLineString":
      for (const line of geometry.coordinates) {
        lineToPath(path, line);
      }
      break;
    case "Point":
      circleToPath(path, geometry.coordinates, 0.5);
      break;
    case "MultiPoint":
      for (const point of geometry.coordinates) circleToPath(path, point, 0.5);
      break;
  }
  return path;
}

function polygonToPath(path: Path2D, polygon: number[][][]) {
  if (!polygon?.length) return;
  polygon.forEach((ring, index) => {
    ring.forEach(([x, y], i) => {
      if (i === 0) path.moveTo(x, y);
      else path.lineTo(x, y);
    });
    if (index === 0) path.closePath();
  });
}

function lineToPath(path: Path2D, line: number[][]) {
  if (!line?.length) return;
  line.forEach(([x, y], i) => {
    if (i === 0) path.moveTo(x, y);
    else path.lineTo(x, y);
  });
}

function circleToPath(path: Path2D, point: number[], radius: number) {
  const [x, y] = point;
  path.moveTo(x + radius, y);
  path.arc(x, y, radius, 0, Math.PI * 2);
}

function computeBounds(layers: PreparedLayer[]) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const { items } of layers) {
    for (const { feature } of items) {
      const geom = feature.geometry;
      if (!geom) continue;
      traverseCoords(geom.coordinates, ([x, y]) => {
        if (typeof x !== "number" || typeof y !== "number") return;
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      });
    }
  }
  if (!isFinite(minX) || !isFinite(minY) || !isFinite(maxX) || !isFinite(maxY)) return null;
  return { minX, minY, maxX, maxY, hash: `${minX}:${minY}:${maxX}:${maxY}` };
}

function traverseCoords(coords: any, fn: (pt: [number, number]) => void) {
  if (!coords) return;
  if (typeof coords[0] === "number") {
    fn(coords as [number, number]);
  } else if (Array.isArray(coords)) {
    coords.forEach((c) => traverseCoords(c, fn));
  }
}

function fitTransformToBounds(
  bounds: { minX: number; minY: number; maxX: number; maxY: number },
  size: { width: number; height: number }
): Transform {
  const width = bounds.maxX - bounds.minX;
  const height = bounds.maxY - bounds.minY;
  if (width <= 0 || height <= 0) return defaultTransform;
  const padding = 40;
  const scale = Math.min((size.width - padding) / width, (size.height - padding) / height);
  const tx = (size.width - width * scale) / 2 - bounds.minX * scale;
  const ty = (size.height - height * scale) / 2 - bounds.minY * scale;
  return { scale, tx, ty };
}

function screenToWorld(event: PointerEvent, canvas: HTMLCanvasElement, transform: Transform) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  return {
    x: (x - transform.tx) / transform.scale,
    y: (y - transform.ty) / transform.scale,
  };
}

function hitTest(preparedLayers: PreparedLayer[], world: { x: number; y: number }, transform: Transform, canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  for (let i = preparedLayers.length - 1; i >= 0; i -= 1) {
    const { layer, items } = preparedLayers[i];
    if (!items.length) continue;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    for (let j = items.length - 1; j >= 0; j -= 1) {
      const { path, feature } = items[j];
      if (ctx.isPointInStroke(path, world.x, world.y) || ctx.isPointInPath(path, world.x, world.y)) {
        ctx.restore();
        return { layerId: layer.id, feature };
      }
    }
    ctx.restore();
  }
  return null;
}

function zoomAtPoint(
  transform: Transform,
  point: { x: number; y: number },
  factor: number,
  minScale: number
): Transform {
  const lowerBound = Math.max(minScale, 0.05);
  const newScale = Math.min(Math.max(transform.scale * factor, lowerBound), 20);
  const dx = point.x - transform.tx;
  const dy = point.y - transform.ty;
  const scaleRatio = newScale / transform.scale;
  return {
    scale: newScale,
    tx: point.x - dx * scaleRatio,
    ty: point.y - dy * scaleRatio,
  };
}

function drawHighlight(
  ctx: CanvasRenderingContext2D,
  layers: PreparedLayer[],
  target: { layerId: string; feature: Feature },
  transform: Transform,
  options: { stroke: string; lineWidth: number; dash?: number[] }
) {
  const layer = layers.find((l) => l.layer.id === target.layerId);
  if (!layer) return;
  const item = layer.items.find((it) => it.feature === target.feature);
  if (!item) return;
  ctx.save();
  ctx.translate(transform.tx, transform.ty);
  ctx.scale(transform.scale, transform.scale);
  ctx.lineWidth = options.lineWidth / transform.scale;
  ctx.strokeStyle = options.stroke;
  ctx.setLineDash(options.dash ?? []);
  ctx.stroke(item.path);
  ctx.restore();
}

function drawMeasure(ctx: CanvasRenderingContext2D, points: [number, number][], transform: Transform) {
  const [[x1, y1], [x2, y2]] = points;
  ctx.save();
  ctx.translate(transform.tx, transform.ty);
  ctx.scale(transform.scale, transform.scale);
  ctx.strokeStyle = "var(--dynamic-accent)";
  ctx.lineWidth = 2 / transform.scale;
  ctx.setLineDash([4, 6]);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();

  const midX = (x1 + x2) / 2;
  const midY = (y1 + y2) / 2;
  const distance = Math.hypot(x2 - x1, y2 - y1);
  const label = `${distance.toFixed(2)} units`;
  ctx.save();
  ctx.fillStyle = "rgba(0,0,0,0.6)";
  ctx.strokeStyle = "rgba(0,0,0,0.4)";
  ctx.lineWidth = 1;
  const screenX = midX * transform.scale + transform.tx;
  const screenY = midY * transform.scale + transform.ty;
  ctx.font = "12px 'Inter', sans-serif";
  const textWidth = ctx.measureText(label).width;
  ctx.fillRect(screenX - textWidth / 2 - 6, screenY - 22, textWidth + 12, 20);
  ctx.fillStyle = "var(--accent)";
  ctx.fillText(label, screenX - textWidth / 2, screenY - 8);
  ctx.restore();
}


