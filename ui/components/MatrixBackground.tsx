import { useEffect, useRef } from "react";
import { MATRIX_DEFAULTS, MATRIX_LIMITS } from "@/lib/hooks/useSettings";

type MatrixBackgroundProps = {
  speed?: number;
  density?: number;
  opacity?: number;
  color?: string | null;
  enabled?: boolean;
};

const glyphs = "アイウエカサタナ0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const clampNumber = (value: number | undefined, min: number, max: number, fallback: number) => {
  if (value == null || Number.isNaN(value)) return fallback;
  return Math.min(max, Math.max(min, value));
};

export default function MatrixBackground({
  speed = 1,
  density = 1,
  opacity = 0.35,
  color,
  enabled = true,
}: MatrixBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrame = 0;
    const dpr = Math.max(1, Math.floor(window.devicePixelRatio || 1));
    let width = 0;
    let height = 0;
    let columns: number[] = [];

    const safeSpeed = clampNumber(
      typeof speed === "number" ? speed : MATRIX_DEFAULTS.speed,
      MATRIX_LIMITS.speed.min,
      MATRIX_LIMITS.speed.max,
      MATRIX_DEFAULTS.speed,
    );
    const safeDensity = clampNumber(
      typeof density === "number" ? density : MATRIX_DEFAULTS.density,
      MATRIX_LIMITS.density.min,
      MATRIX_LIMITS.density.max,
      MATRIX_DEFAULTS.density,
    );
    const safeOpacity = clampNumber(
      typeof opacity === "number" ? opacity : MATRIX_DEFAULTS.opacity,
      MATRIX_LIMITS.opacity.min,
      MATRIX_LIMITS.opacity.max,
      MATRIX_DEFAULTS.opacity,
    );

    const computeColor = () => {
      if (color && color.trim().length > 0) return color;
      const computed = getComputedStyle(document.documentElement).getPropertyValue("--matrix-color");
      return computed.trim() || "#1f6f4c";
    };

    const effectiveColor = computeColor();

    const minGlyphSize = 12;

    const resize = () => {
      const { innerWidth, innerHeight } = window;
      width = Math.floor(innerWidth * dpr);
      height = Math.floor(innerHeight * dpr);
      canvas.width = width;
      canvas.height = height;
      canvas.style.width = `${innerWidth}px`;
      canvas.style.height = `${innerHeight}px`;

      const glyphSize = Math.max(minGlyphSize, 16 * safeDensity);
      const columnCount = Math.max(1, Math.floor(innerWidth / glyphSize));
      columns = Array.from({ length: columnCount }, () => -Math.random() * 40);
    };

    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    let reducedMotion = prefersReducedMotion.matches;

    const clearCanvas = () => {
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, width, height);
      ctx.restore();
    };

    const stopLoop = () => {
      if (!animationFrame) return;
      window.cancelAnimationFrame(animationFrame);
      animationFrame = 0;
    };

    const step = () => {
      if (!enabled || reducedMotion) {
        stopLoop();
        clearCanvas();
        return;
      }

      const glyphSize = Math.max(minGlyphSize, 16 * safeDensity) * dpr;
      const fallSpeed = Math.max(0.05, 5 * safeSpeed);

      ctx.fillStyle = `rgba(0, 0, 0, ${Math.min(0.6, Math.max(0.08, safeOpacity * 0.4))})`;
      ctx.fillRect(0, 0, width, height);

      ctx.font = `${glyphSize}px "Source Code Pro", "Fira Code", monospace`;
      ctx.textBaseline = "top";
      ctx.fillStyle = effectiveColor;
      ctx.globalAlpha = Math.min(1, Math.max(0.05, safeOpacity));

      for (let index = 0; index < columns.length; index += 1) {
        const x = index * glyphSize;
        const columnY = columns[index] * glyphSize;
        const char = glyphs[Math.floor(Math.random() * glyphs.length)];
        ctx.fillText(char, x, columnY);
        columns[index] += fallSpeed;
        if (columnY > height + glyphSize * 4 || Math.random() < 0.01) {
          columns[index] = -Math.random() * 20;
        }
      }

      ctx.globalAlpha = 1;
      animationFrame = window.requestAnimationFrame(step);
    };

    const startLoop = () => {
      if (animationFrame) return;
      animationFrame = window.requestAnimationFrame(step);
    };

    const onReducedMotionChange = (event: MediaQueryListEvent) => {
      reducedMotion = event.matches;
      if (reducedMotion) {
        stopLoop();
        clearCanvas();
      } else if (enabled && !document.hidden) {
        startLoop();
      }
    };

    const handleVisibility = () => {
      if (document.hidden) {
        stopLoop();
      } else if (!reducedMotion && enabled) {
        startLoop();
      }
    };

    resize();
    const prefersListenerSupported = typeof prefersReducedMotion.addEventListener === "function";
    if (prefersListenerSupported) {
      prefersReducedMotion.addEventListener("change", onReducedMotionChange);
    } else if (typeof prefersReducedMotion.addListener === "function") {
      prefersReducedMotion.addListener(onReducedMotionChange);
    }
    window.addEventListener("resize", resize);
    document.addEventListener("visibilitychange", handleVisibility);
    if (enabled && !reducedMotion && !document.hidden) {
      startLoop();
    } else {
      clearCanvas();
    }

    return () => {
      stopLoop();
      window.removeEventListener("resize", resize);
      document.removeEventListener("visibilitychange", handleVisibility);
      if (prefersListenerSupported) {
        prefersReducedMotion.removeEventListener("change", onReducedMotionChange);
      } else if (typeof prefersReducedMotion.removeListener === "function") {
        prefersReducedMotion.removeListener(onReducedMotionChange);
      }
    };
  }, [color, density, enabled, opacity, speed]);

  return (
    <div className="fixed inset-0 -z-10 pointer-events-none select-none">
      <canvas ref={canvasRef} className="h-full w-full" />
      <div
        className="absolute inset-0"
        style={{
          background:
            "linear-gradient(180deg, rgba(0, 0, 0, 0.18) 0%, rgba(0, 0, 0, 0.12) 30%, rgba(0, 0, 0, 0.24) 70%, rgba(0, 0, 0, 0.4) 100%)",
        }}
      />
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(1200px 600px at 20% -10%, rgba(31, 111, 76, 0.35), transparent 65%), radial-gradient(1200px 600px at 80% 110%, rgba(212, 183, 111, 0.22), transparent 60%)",
          opacity: 0.5,
        }}
      />
    </div>
  );
}

