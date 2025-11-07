type HapticKind = "tap" | "success" | "error" | "heavy";

const patterns: Record<HapticKind, number | number[]> = {
  tap: [12],
  success: [18, 8, 18],
  error: [28, 16, 28, 16, 28],
  heavy: [40],
};

let cachedSupports: boolean | null = null;

function supportsVibrate(): boolean {
  if (cachedSupports != null) return cachedSupports;
  if (typeof navigator === "undefined") return false;
  cachedSupports = typeof navigator.vibrate === "function";
  return cachedSupports;
}

function prefersReducedMotion(): boolean {
  if (typeof window === "undefined") return true;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export function getHapticsPreference(): boolean {
  if (typeof window !== "undefined") {
    const flag = window.localStorage.getItem("bimify-enable-haptics");
    if (flag !== null) return flag === "true";
  }
  const env = process.env.NEXT_PUBLIC_ENABLE_HAPTICS;
  if (env != null) return env !== "false";
  return true;
}

function isEnabled() {
  if (!supportsVibrate()) return false;
  if (prefersReducedMotion()) return false;
  return getHapticsPreference();
}

export function setHapticsEnabled(enabled: boolean) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem("bimify-enable-haptics", String(enabled));
}

export function triggerHaptic(kind: HapticKind = "tap") {
  if (!isEnabled()) return;
  try {
    navigator.vibrate?.(patterns[kind]);
  } catch (err) {
    // ignore silently
  }
}

export function canUseHaptics() {
  return supportsVibrate() && !prefersReducedMotion();
}


