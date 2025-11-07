import { useState } from "react";
import GlassPanel from "@/components/GlassPanel";
import DepthButton from "@/components/DepthButton";

type Values = {
  rf_confidence: number;
  rf_overlap: number;
  per_class_thresholds: Record<string, number>;
};

export default function ThresholdSliders({ initial, onSubmit }: { initial?: Partial<Values>; onSubmit: (v: Values) => void }) {
  const [confidence, setConfidence] = useState(initial?.rf_confidence ?? 0.01);
  const [overlap, setOverlap] = useState(initial?.rf_overlap ?? 0.3);
  const [perClass, setPerClass] = useState<Record<string, number>>(
    initial?.per_class_thresholds ?? {
      internal_wall: 0.01,
      external_wall: 0.01,
      door: 0.01,
      window: 0.01,
      stair: 0.01,
    }
  );

  return (
    <GlassPanel className="p-4 space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <div>
          <label className="text-sm opacity-75" htmlFor="threshold-rf-confidence">
            RF Confidence
          </label>
          <input
            id="threshold-rf-confidence"
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
        <div>
          <label className="text-sm opacity-75" htmlFor="threshold-rf-overlap">
            RF Overlap
          </label>
          <input
            id="threshold-rf-overlap"
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={overlap}
            onChange={(e) => setOverlap(parseFloat(e.target.value))}
            className="w-full"
          />
          <div className="text-sm mt-1 opacity-80">{overlap.toFixed(2)}</div>
        </div>
      </div>
      <div className="grid gap-3 md:grid-cols-3">
        {Object.entries(perClass).map(([k, v]) => {
          const sliderId = `threshold-per-class-${k}`;
          return (
            <div key={k} className="space-y-1">
              <label className="text-xs opacity-70" htmlFor={sliderId}>
                {k}
              </label>
              <input
                id={sliderId}
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={v}
                onChange={(e) => setPerClass({ ...perClass, [k]: parseFloat(e.target.value) })}
                className="w-full"
              />
              <div className="text-xs opacity-70">{v.toFixed(2)}</div>
            </div>
          );
        })}
      </div>
      <div className="flex justify-end">
        <DepthButton onClick={() => onSubmit({ rf_confidence: confidence, rf_overlap: overlap, per_class_thresholds: perClass })}>Reprocess</DepthButton>
      </div>
    </GlassPanel>
  );
}


