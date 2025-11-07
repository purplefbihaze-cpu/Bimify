import { ReactNode, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Menu, Github } from "lucide-react";
import DepthButton from "@/components/DepthButton";
import ThemeSwitcher from "@/components/ThemeSwitcher";
import CornerLogo from "@/components/CornerLogo";
import { drawerTransition } from "@/lib/motion/map";
import MatrixBackground from "@/components/MatrixBackground";
import { useMatrixBackgroundConfig } from "@/lib/hooks/useSettings";

export default function AppShell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const navigationId = "primary-navigation";
  const navContentRef = useRef<HTMLDivElement | null>(null);
  const { config: matrixConfig } = useMatrixBackgroundConfig();

  useEffect(() => {
    const node = navContentRef.current;
    if (!node) return;
    if (open) {
      node.removeAttribute("inert");
    } else {
      node.setAttribute("inert", "");
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
      }
    };
    const handlePointerDown = (event: PointerEvent) => {
      const nav = navContentRef.current;
      const toggle = document.getElementById("appshell-menu-toggle");
      const target = event.target as Node;
      if (!nav) return;
      if (nav.contains(target)) return;
      if (toggle && toggle.contains(target)) return;
      setOpen(false);
    };
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("pointerdown", handlePointerDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("pointerdown", handlePointerDown);
    };
  }, [open]);

  return (
    <div className="relative min-h-full overflow-hidden">
      <MatrixBackground
        enabled={matrixConfig.enabled}
        speed={matrixConfig.speed}
        density={matrixConfig.density}
        opacity={matrixConfig.opacity}
        color={matrixConfig.color}
      />
      <header className="sticky top-0 z-40">
        <div className="mx-auto max-w-7xl px-4 py-3">
          <div className="glass relative rounded-xl px-4 py-3 flex items-center justify-between shadow-glass overflow-hidden">
            <div className="flex basis-0 flex-1 items-center gap-3">
              <DepthButton
                id="appshell-menu-toggle"
                onClick={() => setOpen((v) => !v)}
                aria-label={open ? "Menü schließen" : "Menü öffnen"}
                aria-controls={navigationId}
                aria-expanded={open}
              >
                <Menu className="h-5 w-5" />
              </DepthButton>
            </div>
            <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 flex items-center justify-center">
              <motion.div
                initial={{ opacity: 0, scale: 0.92 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.6, ease: "easeOut" }}
                className="relative"
              >
                <div className="pointer-events-none absolute -inset-8 rounded-full bg-emerald-400/10 blur-3xl" />
                <motion.img
                  src="/bimmatrixlogotransparent.png"
                  alt="BIM Matrix"
                  className="relative z-10 h-24 w-auto max-w-[360px] object-contain drop-shadow-[0_14px_40px_rgba(34,211,238,0.35)]"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.8, ease: "easeOut", delay: 0.15 }}
                />
              </motion.div>
            </div>
            <div className="flex basis-0 flex-1 items-center justify-end gap-3">
              <ThemeSwitcher />
              <a
                href="https://github.com/ddrob/Bimify"
                className="text-sm opacity-90 flex items-center gap-2 focus-ring rounded-lg px-2 py-1 text-[var(--accent-strong)] transition-colors hover:text-[var(--accent)]"
                target="_blank"
                rel="noreferrer"
              >
                <Github className="h-4 w-4" />
                <span>Docs</span>
              </a>
            </div>
          </div>
        </div>
        <motion.div
          aria-hidden={!open}
          initial={false}
          animate={{ height: open ? 220 : 0, opacity: open ? 1 : 0 }}
          transition={drawerTransition}
          className="overflow-hidden"
          style={{ pointerEvents: open ? "auto" : "none" }}
        >
          <div className="mx-auto max-w-7xl px-4">
            <div ref={navContentRef} id={navigationId} className="glass rounded-xl px-6 py-6 shadow-depth" role="navigation">
              <nav className="grid grid-cols-2 gap-4" aria-label="Hauptnavigation">
                <Link className="focus-ring rounded-lg border border-white/5 px-4 py-3 hover:bg-white/5 transition-colors" href="/">
                  Upload
                </Link>
                <Link className="focus-ring rounded-lg border border-white/5 px-4 py-3 hover:bg-white/5 transition-colors" href="/settings">
                  Einstellungen
                </Link>
              </nav>
            </div>
          </div>
        </motion.div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-6">{children}</main>
      <footer className="mx-auto max-w-7xl px-4 py-10 opacity-70 text-sm">Built for cinematic clarity.</footer>
      <CornerLogo />
    </div>
  );
}


