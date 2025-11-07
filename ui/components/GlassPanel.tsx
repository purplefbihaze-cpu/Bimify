import { ReactNode } from "react";
import { motion } from "framer-motion";
import { panelVariants } from "@/lib/motion/map";

export default function GlassPanel({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <motion.div variants={panelVariants} initial="hidden" animate="show" className={`glass shadow-glass ${className} rounded-xl`}>
      {children}
    </motion.div>
  );
}


