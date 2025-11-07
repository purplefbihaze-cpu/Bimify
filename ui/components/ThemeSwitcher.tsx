import { useEffect, useMemo, useState } from "react";
import { Palette, Sun, Moon, Contrast, Vibrate } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import DepthButton from "@/components/DepthButton";
import { useTheme } from "@/lib/hooks/useTheme";
import { canUseHaptics, getHapticsPreference, setHapticsEnabled, triggerHaptic } from "@/lib/haptics";

const paletteOrder = ["base", "warm", "cool"] as const;

export default function ThemeSwitcher() {
  const { palette, setPalette, mode, setMode, highContrast, toggleHighContrast } = useTheme();
  const [open, setOpen] = useState(false);
  const [haptics, setHaptics] = useState(getHapticsPreference());
  const [hapticsAvailable, setHapticsAvailable] = useState(false);

  useEffect(() => {
    setHapticsAvailable(canUseHaptics());
  }, []);

  const paletteIndex = useMemo(() => paletteOrder.indexOf(palette), [palette]);

  const cyclePalette = () => {
    const next = paletteOrder[(paletteIndex + 1) % paletteOrder.length];
    setPalette(next);
    triggerHaptic("tap");
  };

  const cycleMode = () => {
    const next = mode === "system" ? "light" : mode === "light" ? "dark" : "system";
    setMode(next);
    triggerHaptic("tap");
  };

  const toggleHaptics = () => {
    const next = !haptics;
    setHaptics(next);
    setHapticsEnabled(next);
    if (next) triggerHaptic("tap");
  };

  return (
    <div className="relative">
      <DepthButton onClick={() => setOpen((v) => !v)} aria-haspopup="menu" aria-expanded={open}>
        <Palette className="h-4 w-4" />
      </DepthButton>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
            className="absolute right-0 mt-3 w-56 glass rounded-xl p-3 shadow-glass z-40"
            role="menu"
          >
            <div className="flex items-center justify-between text-xs uppercase tracking-[0.2em] opacity-60 pb-2">Theme</div>
            <div className="space-y-2">
              <button
                className="focus-ring w-full rounded-lg border border-white/10 px-3 py-2 text-left text-sm hover:bg-white/5 transition"
                onClick={cyclePalette}
              >
                Palette: <span className="capitalize">{palette}</span>
              </button>
              <button
                className="focus-ring w-full rounded-lg border border-white/10 px-3 py-2 text-left text-sm hover:bg-white/5 transition flex items-center justify-between"
                onClick={cycleMode}
              >
                <span>Mode: {mode}</span>
                {mode === "light" ? <Sun className="h-4 w-4" /> : mode === "dark" ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
              </button>
              <button
                className="focus-ring w-full rounded-lg border border-white/10 px-3 py-2 text-left text-sm hover:bg-white/5 transition flex items-center justify-between"
                onClick={() => {
                  toggleHighContrast();
                  triggerHaptic("tap");
                }}
              >
                <span>High Contrast</span>
                <Contrast className={`h-4 w-4 ${highContrast ? "text-accent-400" : "opacity-60"}`} />
              </button>
              <button
                className="focus-ring w-full rounded-lg border border-white/10 px-3 py-2 text-left text-sm hover:bg-white/5 transition flex items-center justify-between disabled:opacity-40"
                onClick={toggleHaptics}
                disabled={!hapticsAvailable}
              >
                <span>Haptics</span>
                <Vibrate className={`h-4 w-4 ${haptics && hapticsAvailable ? "text-accent-400" : "opacity-60"}`} />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}


