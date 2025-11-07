import { useEffect, useRef, useState } from "react";
import ScaleOverlay, { ScaleCalibration } from "@/components/ScaleOverlay";
import { Prediction } from "@/lib/api";

const COLOR_PALETTE = [
  "#60a5fa",
  "#f87171",
  "#34d399",
  "#fbbf24",
  "#c084fc",
  "#f472b6",
  "#2dd4bf",
  "#a855f7",
  "#fb7185",
  "#38bdf8",
];

const CLASS_COLORS = new Map<string, string>();

function colorForLabel(label: string): string {
  if (!label) return COLOR_PALETTE[0];
  const key = label.toLowerCase();
  if (CLASS_COLORS.has(key)) return CLASS_COLORS.get(key)!;
  const color = COLOR_PALETTE[CLASS_COLORS.size % COLOR_PALETTE.length];
  CLASS_COLORS.set(key, color);
  return color;
}

function formatLabel(pred: Prediction): string {
  const label = pred.class || "unknown";
  const conf = pred.confidence != null ? (pred.confidence * 100).toFixed(1) : "";
  return conf ? `${label} (${conf}%)` : label;
}

type Props = {
  imageSrc: string;
  imageMeta?: Record<string, any> | null;
  predictions: Prediction[];
  drawLabels: boolean;
  drawShapes: boolean;
  showSensors: boolean;
  threshold?: number;
  scaleToolActive?: boolean;
  calibration?: ScaleCalibration | null;
  onCalibrationComplete?: (calibration: ScaleCalibration) => void;
  onCalibrationCancel?: () => void;
};

export default function PredictionViewer({
  imageSrc,
  imageMeta,
  predictions,
  drawLabels,
  drawShapes,
  showSensors,
  threshold,
  scaleToolActive = false,
  calibration = null,
  onCalibrationComplete,
  onCalibrationCancel,
}: Props) {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 });

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    if (img.complete && img.naturalWidth) {
      setNaturalSize({ width: img.naturalWidth, height: img.naturalHeight });
      return;
    }
    const handle = () => {
      setNaturalSize({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.addEventListener("load", handle);
    return () => img.removeEventListener("load", handle);
  }, [imageSrc]);

  const metaWidth = imageMeta && typeof imageMeta.width === "number" ? imageMeta.width : Number(imageMeta?.width || imageMeta?.Width || 0);
  const metaHeight = imageMeta && typeof imageMeta.height === "number" ? imageMeta.height : Number(imageMeta?.height || imageMeta?.Height || 0);
  const baseWidth = naturalSize.width || metaWidth || 1000;
  const baseHeight = naturalSize.height || metaHeight || 1000;

  const thresholdValue = typeof threshold === "number" ? threshold : null;
  const safePredictions = (predictions ?? [])
    .filter((pred) => {
      if (showSensors) return true;
      const label = pred.class?.toLowerCase?.() || "";
      return !label.includes("sensor");
    })
    .filter((pred) => {
      if (thresholdValue == null) return true;
      const confidenceValue = Number(pred.confidence);
      if (!Number.isFinite(confidenceValue)) return true;
      return confidenceValue >= thresholdValue;
    });

  return (
    <div className="relative w-full">
      <img ref={imgRef} src={imageSrc} alt="Analyse" className="w-full rounded-xl border border-white/10 object-contain" />
      {calibration?.points && calibration.points[0] && calibration.points[1] && (
        <svg
          className="pointer-events-none absolute inset-0 h-full w-full"
          viewBox={`0 0 ${baseWidth || 1000} ${baseHeight || 1000}`}
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            <marker id="scale-calibrated" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto-start-reverse">
              <path d="M0,0 L6,3 L0,6 z" fill="#22d3ee" />
            </marker>
          </defs>
          <line
            x1={calibration.points[0][0]}
            y1={calibration.points[0][1]}
            x2={calibration.points[1][0]}
            y2={calibration.points[1][1]}
            stroke="#22d3ee"
            strokeWidth={2}
            strokeDasharray="4 6"
            markerStart="url(#scale-calibrated)"
            markerEnd="url(#scale-calibrated)"
          />
        </svg>
      )}
      {scaleToolActive && (
        <ScaleOverlay
          width={baseWidth || 1000}
          height={baseHeight || 1000}
          imageSrc={imageSrc}
          onComplete={(value) => onCalibrationComplete?.(value)}
          onCancel={() => onCalibrationCancel?.()}
        />
      )}
      {drawShapes || drawLabels || showSensors ? (
        <svg
          className="pointer-events-none absolute inset-0 h-full w-full"
          viewBox={`0 0 ${baseWidth || 1000} ${baseHeight || 1000}`}
          preserveAspectRatio="xMidYMid meet"
        >
          {safePredictions.map((pred, idx) => {
            const color = colorForLabel(pred.class);
            const label = formatLabel(pred);
            const centerX = pred.x ?? 0;
            const centerY = pred.y ?? 0;
            const width = pred.width ?? 0;
            const height = pred.height ?? 0;
            const topLeftX = centerX - width / 2;
            const topLeftY = centerY - height / 2;
            const points = Array.isArray(pred.points) ? pred.points : null;

            return (
              <g key={pred.id || `${pred.class}-${idx}`}>
                {drawShapes && points && points.length > 2 && (
                  <polygon points={points.map((pt) => pt.join(",")).join(" ")} fill={`${color}33`} stroke={color} strokeWidth={2} />
                )}
                {drawShapes && (!points || points.length <= 2) && width > 0 && height > 0 && (
                  <rect
                    x={topLeftX}
                    y={topLeftY}
                    width={width}
                    height={height}
                    fill={`${color}33`}
                    stroke={color}
                    strokeWidth={2}
                    rx={6}
                    ry={6}
                  />
                )}
                {drawLabels && (
                  <text
                    x={topLeftX}
                    y={topLeftY - 6}
                    fill={color}
                    stroke="#0f172a"
                    strokeWidth={0.6}
                    fontSize={Math.max(16, Math.min(24, width / 4 || 18))}
                  >
                    {label}
                  </text>
                )}
                {showSensors && (
                  <circle cx={centerX} cy={centerY} r={Math.max(6, Math.min(18, width / 6 || 12))} fill={color} opacity={0.9} />
                )}
              </g>
            );
          })}
        </svg>
      ) : null}
    </div>
  );
}
