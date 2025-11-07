import { motion, HTMLMotionProps } from "framer-motion";
import { clsx } from "clsx";
import { buttonWhileHover, buttonWhileTap } from "@/lib/motion/map";

type Props = HTMLMotionProps<"button"> & { active?: boolean };

export default function DepthButton({ className, active, children, type = "button", ...rest }: Props) {
  const disabled = rest.disabled;
  return (
    <motion.button
      type={type}
      whileTap={disabled ? undefined : buttonWhileTap}
      whileHover={disabled ? undefined : buttonWhileHover}
      className={clsx(
        "focus-ring inline-flex items-center justify-center rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm shadow-soft transition-colors",
        active && !disabled && "bg-white/10",
        disabled && "cursor-not-allowed opacity-50",
        className
      )}
      {...rest}
    >
      {children}
    </motion.button>
  );
}


