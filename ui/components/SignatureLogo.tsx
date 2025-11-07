import { motion } from "framer-motion";

export default function SignatureLogo() {
  return (
    <div className="flex items-center justify-center">
      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: "spring", stiffness: 120, damping: 14 }}
        className="relative"
      >
        <motion.div
          className="absolute -inset-14 rounded-full blur-3xl"
          style={{ background: "radial-gradient(closest-side, rgba(31,111,76,0.45), transparent)" }}
          initial={{ opacity: 0.2 }}
          animate={{ opacity: 0.35 }}
          transition={{ duration: 1.2 }}
        />
        <img
          src="/2d-3d-logo.png"
          alt="BIM-Matrix"
          className="relative z-10 bloom h-96 w-auto max-w-[800px] max-h-[80vh]"
        />
      </motion.div>
    </div>
  );
}


