import "@/styles/globals.css";
import type { AppProps } from "next/app";
import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import AppShell from "@/layouts/AppShell";
import SignatureLogo from "@/components/SignatureLogo";
import { ThemeProvider } from "@/lib/hooks/useTheme";
import { pageVariants } from "@/lib/motion/map";

export default function MyApp({ Component, pageProps, router }: AppProps) {
  const [showIntro, setShowIntro] = useState(true);
  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    let timer: number | null = null;

    const startIntro = () => {
      if (timer != null) window.clearTimeout(timer);
      setShowIntro(true);
      timer = window.setTimeout(() => {
        setShowIntro(false);
        timer = null;
      }, 1600);
    };

    const stopIntro = () => {
      if (timer != null) window.clearTimeout(timer);
      timer = null;
      setShowIntro(false);
    };

    if (media.matches) {
      stopIntro();
    } else {
      startIntro();
    }

    const handleChange = (event: MediaQueryListEvent) => {
      if (event.matches) {
        stopIntro();
      } else {
        startIntro();
      }
    };

    media.addEventListener("change", handleChange);
    return () => {
      if (timer != null) window.clearTimeout(timer);
      media.removeEventListener("change", handleChange);
    };
  }, []);

  return (
    <ThemeProvider>
      <AppShell>
        <AnimatePresence>
          {showIntro && (
            <motion.div
              initial={{ opacity: 1 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.5 }}
              className="fixed inset-0 z-50 flex items-center justify-center backdrop-blur-sm"
              style={{
                backgroundColor: "var(--bg)",
                background: "color-mix(in srgb, var(--bg) 86%, transparent)",
              }}
            >
              <SignatureLogo />
            </motion.div>
          )}
        </AnimatePresence>
        <AnimatePresence mode="wait" initial={false}>
          <motion.div key={router.asPath} variants={pageVariants} initial="initial" animate="animate" exit="exit">
            <Component {...pageProps} />
          </motion.div>
        </AnimatePresence>
      </AppShell>
    </ThemeProvider>
  );
}


