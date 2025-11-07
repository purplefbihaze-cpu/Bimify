'use client';

import { motion } from "framer-motion";

export default function CornerLogo() {
  return (
    <motion.div
      className="pointer-events-none fixed bottom-6 left-6 z-40"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6, delay: 0.3, ease: "easeOut" }}
    >
      <div className="rounded-xl border border-white/10 bg-black/60 p-2 shadow-[0_12px_32px_-18px_rgba(34,211,238,0.6)] backdrop-blur-md">
        <img src="/bimmatrixlogocorner.png" alt="BIM Matrix Corner Logo" className="h-12 w-12 object-contain" />
      </div>
    </motion.div>
  );
}
