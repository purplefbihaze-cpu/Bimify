import { useEffect } from "react";
import { animate, motion, useTransform, useMotionValue } from "framer-motion";
import { motionCurves, motionDurations } from "@/lib/motion/map";

export default function ProgressOrb({ value }: { value: number }) {
  const mv = useMotionValue(0);
  const dash = useTransform(mv, [0, 100], [0, 283]);
  const clamped = Math.max(0, Math.min(100, value));

  useEffect(() => {
    const controls = animate(mv, clamped, {
      duration: motionDurations.slow,
      ease: motionCurves.outExpo,
    });
    return controls.stop;
  }, [clamped, mv]);

  return (
    <div className="relative inline-block">
      <svg width="120" height="120" viewBox="0 0 120 120" className="bloom">
        <defs>
          <linearGradient id="pg" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="var(--dynamic-accent)" />
            <stop offset="100%" stopColor="var(--accent-soft)" />
          </linearGradient>
        </defs>
        <circle cx="60" cy="60" r="50" stroke="rgba(255,255,255,0.08)" strokeWidth="10" fill="none" />
        <motion.circle
          cx="60"
          cy="60"
          r="45"
          stroke="url(#pg)"
          strokeWidth="8"
          strokeLinecap="round"
          fill="none"
          strokeDasharray="283"
          style={{ strokeDashoffset: dash as any }}
          transform="rotate(-90 60 60)"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center text-xl font-semibold">{Math.round(value)}%</div>
    </div>
  );
}


