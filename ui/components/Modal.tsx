import { PropsWithChildren, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

type ModalProps = {
  open: boolean;
  onClose: () => void;
};

export default function Modal({ open, onClose, children }: PropsWithChildren<ModalProps>) {
  const [isMounted, setIsMounted] = useState(false);
  const [isVisible, setIsVisible] = useState(open);
  const closeTimeout = useRef<number | null>(null);

  useEffect(() => {
    setIsMounted(true);
    return () => {
      setIsMounted(false);
      if (closeTimeout.current) {
        window.clearTimeout(closeTimeout.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!isMounted) return;

    if (open) {
      if (closeTimeout.current) {
        window.clearTimeout(closeTimeout.current);
      }
      setIsVisible(true);
      document.body.style.overflow = "hidden";
    } else if (isVisible) {
      closeTimeout.current = window.setTimeout(() => {
        setIsVisible(false);
        document.body.style.overflow = "";
      }, 220);
    }

    return () => {
      if (!open && !isVisible) {
        document.body.style.overflow = "";
      }
    };
  }, [open, isVisible, isMounted]);

  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open, onClose]);

  if (!isMounted || (!isVisible && !open)) {
    return null;
  }

  return createPortal(
    <div
      className={`fixed inset-0 z-[200] flex items-center justify-center bg-slate-950/70 backdrop-blur-[6px] transition-opacity duration-200 ${
        open ? "opacity-100" : "opacity-0 pointer-events-none"
      }`}
      onClick={onClose}
      aria-modal="true"
      role="dialog"
    >
      <div
        className={`w-[min(90vw,960px)] max-h-[92vh] overflow-hidden rounded-3xl border border-white/10 bg-slate-900/80 shadow-2xl backdrop-blur-xl transition-all duration-200 ${
          open ? "translate-y-0 scale-100 opacity-100" : "translate-y-6 scale-95 opacity-0"
        }`}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="max-h-[92vh] overflow-y-auto">{children}</div>
      </div>
    </div>,
    document.body,
  );
}

