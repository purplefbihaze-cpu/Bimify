import { useEffect, useState } from "react";

function averageColorFromImage(image: HTMLImageElement): string | null {
  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;
  if (!width || !height) return null;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return null;
  const sampleSize = Math.min(320, Math.max(width, height));
  canvas.width = sampleSize;
  canvas.height = sampleSize;
  ctx.drawImage(image, 0, 0, sampleSize, sampleSize);
  try {
    const data = ctx.getImageData(0, 0, sampleSize, sampleSize).data;
    let r = 0;
    let g = 0;
    let b = 0;
    const step = 4 * 8; // sample every eighth pixel to keep perf smooth
    let count = 0;
    for (let i = 0; i < data.length; i += step) {
      const alpha = data[i + 3];
      if (alpha < 64) continue; // skip mostly transparent pixels
      r += data[i];
      g += data[i + 1];
      b += data[i + 2];
      count += 1;
    }
    if (!count) return null;
    r = Math.round(r / count);
    g = Math.round(g / count);
    b = Math.round(b / count);
    return `rgb(${r}, ${g}, ${b})`;
  } catch (err) {
    return null;
  }
}

export function useDynamicHue(imageUrl: string | null, options?: { strength?: number }) {
  const [color, setColor] = useState<string | null>(null);
  const strength = options?.strength ?? 0.65;

  useEffect(() => {
    if (typeof document === "undefined") return;
    const root = document.documentElement;
    if (!imageUrl) {
      setColor(null);
      root.style.setProperty("--dynamic-accent", "var(--accent)");
      return;
    }
    let active = true;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = imageUrl;
    const onLoad = () => {
      if (!active) return;
      const avg = averageColorFromImage(img);
      if (avg) {
        setColor(avg);
        const computed = getComputedStyle(root);
        const paletteAccent = computed.getPropertyValue("--accent").trim();
        root.style.setProperty(
          "--dynamic-accent",
          `color-mix(in srgb, ${avg} ${Math.round(strength * 100)}%, ${paletteAccent} ${Math.round((1 - strength) * 100)}%)`
        );
      } else {
        root.style.setProperty("--dynamic-accent", paletteAccentFallback(root));
      }
    };
    const onError = () => {
      if (!active) return;
      root.style.setProperty("--dynamic-accent", paletteAccentFallback(root));
    };
    img.addEventListener("load", onLoad);
    img.addEventListener("error", onError);
    return () => {
      active = false;
      img.removeEventListener("load", onLoad);
      img.removeEventListener("error", onError);
      root.style.setProperty("--dynamic-accent", "var(--accent)");
    };
  }, [imageUrl, strength]);

  return color;
}

function paletteAccentFallback(root: HTMLElement) {
  return getComputedStyle(root).getPropertyValue("--accent").trim() || "#7cb7ff";
}














