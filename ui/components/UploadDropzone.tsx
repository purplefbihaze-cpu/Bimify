import { useCallback, useRef, useState } from "react";
import { motion } from "framer-motion";
import DepthButton from "@/components/DepthButton";
import { panelVariants } from "@/lib/motion/map";

const ACCEPT_EXT = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"];

type Props = {
  onSelect: (file: File) => void;
  disabled?: boolean;
};

export default function UploadDropzone({ onSelect, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [hover, setHover] = useState(false);

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setHover(false);
      if (disabled) return;
      const file = Array.from(e.dataTransfer.files).find((f) => {
        const ext = f.name ? f.name.toLowerCase().match(/\.[^.]+$/)?.[0] : "";
        return (f.type && f.type.startsWith("image/")) || (ext ? ACCEPT_EXT.includes(ext) : false);
      });
      if (file) onSelect(file);
    },
    [disabled, onSelect]
  );

  const pick = () => inputRef.current?.click();

  return (
    <motion.div
      variants={panelVariants}
      initial="hidden"
      animate="show"
      onDragOver={(e) => {
        e.preventDefault();
        setHover(true);
      }}
      onDragLeave={() => setHover(false)}
      onDrop={onDrop}
      className={`glass rounded-xl shadow-glass p-8 text-center transition border ${hover ? "border-accent-400/60" : "border-white/10"} ${disabled ? "opacity-60" : ""}`}
    >
      <div className="space-y-3">
        <div className="text-lg">Drag & drop Bild hierher</div>
        <div className="text-sm opacity-75">oder</div>
        <DepthButton onClick={pick} disabled={disabled}>
          Bild auswählen
        </DepthButton>
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          accept={ACCEPT_EXT.join(',') + ",image/*"}
          onChange={(e) => {
            if (disabled) return;
            const file = Array.from(e.target.files || []).find((f) => {
              const ext = f.name ? f.name.toLowerCase().match(/\.[^.]+$/)?.[0] : "";
              return (f.type && f.type.startsWith("image/")) || (ext ? ACCEPT_EXT.includes(ext) : false);
            });
            if (file) onSelect(file);
          }}
        />
      </div>
    </motion.div>
  );
}


