'use client';

import { useEffect, useRef, useState } from "react";

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";

type Props = {
  ifcUrl?: string | null;
  className?: string;
  height?: number | string;
};

type ViewerStatus = "idle" | "loading" | "ready" | "error";

const DEFAULT_HEIGHT = "clamp(24rem, 65vh, 36rem)";
const WASM_PATH = "/wasm/web-ifc/";

export default function IfcJsViewerClient({ ifcUrl, className = "", height = DEFAULT_HEIGHT }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const modelRef = useRef<THREE.Object3D | null>(null);
  const resizeObserverRef = useRef<ResizeObserver | null>(null);

  const [status, setStatus] = useState<ViewerStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight, false);
    renderer.setClearColor(0x0b101b, 1);
    rendererRef.current = renderer;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b101b);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(55, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(10, 15, 18);
    cameraRef.current = camera;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0);
    controlsRef.current = controls;

    const ambient = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambient);
    const directional = new THREE.DirectionalLight(0xffffff, 0.8);
    directional.position.set(10, 20, 10);
    scene.add(directional);

    let animationFrame = 0;
    const tick = () => {
      controls.update();
      renderer.render(scene, camera);
      animationFrame = requestAnimationFrame(tick);
    };
    animationFrame = requestAnimationFrame(tick);

    const handleResize = () => {
      const width = container.clientWidth;
      const heightValue = container.clientHeight;
      renderer.setSize(width, heightValue, false);
      camera.aspect = width / Math.max(heightValue, 1);
      camera.updateProjectionMatrix();
    };

    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);
    resizeObserverRef.current = resizeObserver;

    return () => {
      cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      controls.dispose();
      renderer.dispose();
      scene.clear();
      if (modelRef.current) {
        disposeObject(modelRef.current);
        modelRef.current = null;
      }
      rendererRef.current = null;
      sceneRef.current = null;
      cameraRef.current = null;
      controlsRef.current = null;
      resizeObserverRef.current = null;
    };
  }, []);

  useEffect(() => {
    const url = typeof ifcUrl === "string" ? ifcUrl.trim() : "";
    if (!url) {
      setStatus("idle");
      setError(null);
      if (modelRef.current && sceneRef.current) {
        sceneRef.current.remove(modelRef.current);
        disposeObject(modelRef.current);
        modelRef.current = null;
      }
      return;
    }

    const scene = sceneRef.current;
    if (!scene) return;

    let loader: any = null;

    setStatus("loading");
    setError(null);

    const loadModel = async () => {
      try {
        if (modelRef.current) {
          scene.remove(modelRef.current);
          disposeObject(modelRef.current);
          modelRef.current = null;
        }

        const { IFCLoader } = await import("web-ifc-three/IFCLoader");
        loader = new IFCLoader();
        loader.ifcManager.setWasmPath(WASM_PATH);

        const model = await loader.loadAsync(url);
        model.castShadow = true;
        model.receiveShadow = true;
        scene.add(model);
        modelRef.current = model;

        centerModel(model, cameraRef.current, controlsRef.current);

        setStatus("ready");
      } catch (err: any) {
        console.error("[ifc-viewer] IFC konnte nicht geladen werden", err);
        setError(err?.message ?? "IFC konnte nicht geladen werden.");
        setStatus("error");
      }
    };

    loadModel();

    return () => {
      try {
        loader?.ifcManager?.dispose?.();
      } catch {}
    };
  }, [ifcUrl]);

  const displayHeight = typeof height === "number" ? `${height}px` : height;

  return (
    <div ref={containerRef} className={`relative w-full overflow-hidden rounded-xl border border-white/10 bg-slate-950/80 ${className}`} style={{ height: displayHeight }}>
      <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />
      {status === "loading" && (
        <Overlay message="IFC wird geladen…" />
      )}
      {status === "error" && (
        <Overlay message={error ?? "IFC konnte nicht geladen werden."} tone="error" />
      )}
    </div>
  );
}

function Overlay({ message, tone = "info" }: { message: string; tone?: "info" | "error" }) {
  return (
    <div
      className="absolute inset-0 flex items-center justify-center px-4 text-center text-sm"
      style={{
        background: "linear-gradient(180deg, rgba(15,23,42,0.85) 0%, rgba(10,16,26,0.92) 60%, rgba(8,11,18,0.9) 100%)",
        color: tone === "error" ? "#fca5a5" : "#e2e8f0",
      }}
    >
      {message}
    </div>
  );
}

function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    if ((child as THREE.Mesh).isMesh) {
      const mesh = child as THREE.Mesh;
      if (mesh.geometry) mesh.geometry.dispose();
      if (Array.isArray(mesh.material)) {
        mesh.material.forEach((material) => material?.dispose?.());
      } else if (mesh.material) {
        (mesh.material as THREE.Material).dispose();
      }
    }
  });
}

function centerModel(
  object: THREE.Object3D,
  camera: THREE.PerspectiveCamera | null,
  controls: OrbitControls | null,
) {
  const box = new THREE.Box3().setFromObject(object);
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);

  object.position.sub(center);

  const maxDim = Math.max(size.x, size.y, size.z);
  if (camera) {
    const distance = maxDim * 1.5;
    camera.position.set(center.x + distance, center.y + distance, center.z + distance);
    camera.lookAt(0, 0, 0);
  }
  if (controls) {
    controls.target.set(0, 0, 0);
    controls.update();
  }
}

