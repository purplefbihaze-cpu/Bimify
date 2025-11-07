import { Variants } from "framer-motion";

export const motionCurves = {
  outExpo: [0.16, 1, 0.3, 1] as [number, number, number, number],
  inOutQuint: [0.83, 0, 0.17, 1] as [number, number, number, number],
  gentle: [0.33, 1, 0.68, 1] as [number, number, number, number],
};

export const motionDurations = {
  instant: 0.12,
  fast: 0.2,
  base: 0.36,
  slow: 0.6,
  page: 0.72,
};

export const pageVariants: Variants = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0, transition: { duration: motionDurations.page, ease: motionCurves.outExpo } },
  exit: { opacity: 0, y: -12, transition: { duration: motionDurations.base, ease: motionCurves.inOutQuint } },
};

export const sectionVariants: Variants = {
  hidden: { opacity: 0, y: 20 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: motionDurations.base, ease: motionCurves.outExpo },
  },
};

export const panelVariants: Variants = {
  hidden: { opacity: 0, y: 8, scale: 0.98 },
  show: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: { duration: motionDurations.base, ease: motionCurves.outExpo },
  },
};

export const buttonWhileHover = { scale: 1.02, y: -1 };
export const buttonWhileTap = { scale: 0.97, y: 0 };

export const drawerTransition = {
  type: "spring" as const,
  stiffness: 160,
  damping: 18,
};














