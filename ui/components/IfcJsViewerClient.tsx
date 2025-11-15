'use client';

import { useEffect, useRef, useState } from "react";

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import type { IFCLoader } from "web-ifc-three/IFCLoader";
import {
  IFCCOVERING,
  IFCDOOR,
  IFCPRODUCT,
  IFCROOF,
  IFCSLAB,
  IFCWINDOW,
  IFCWALL,
  IFCWALLSTANDARDCASE,
} from "web-ifc";

type CategoryKey =
  | "windows"
  | "doors"
  | "floors"
  | "ceilings"
  | "interiorWalls"
  | "exteriorWalls"
  | "roofs";

const CATEGORY_KEYS: CategoryKey[] = [
  "windows",
  "doors",
  "floors",
  "ceilings",
  "interiorWalls",
  "exteriorWalls",
  "roofs",
];

const createCategoryState = <T,>(value: T): Record<CategoryKey, T> => ({
  windows: value,
  doors: value,
  floors: value,
  ceilings: value,
  interiorWalls: value,
  exteriorWalls: value,
  roofs: value,
});

const createEmptyCategoryIds = (): Record<CategoryKey, number[]> => ({
  windows: [],
  doors: [],
  floors: [],
  ceilings: [],
  interiorWalls: [],
  exteriorWalls: [],
  roofs: [],
});

const WindowIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
    <rect x="4" y="4" width="16" height="16" rx="2.2" stroke="currentColor" strokeWidth="1.8" />
    <line x1="12" y1="4" x2="12" y2="20" stroke="currentColor" strokeWidth="1.4" />
    <line x1="4" y1="12" x2="20" y2="12" stroke="currentColor" strokeWidth="1.4" />
  </svg>
);

const DoorIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
    <path
      d="M8.5 4h7v15.5H8.5zM11 12a.9.9 0 0 1 .9-.9h.2a.9.9 0 1 1 0 1.8h-.2A.9.9 0 0 1 11 12Z"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const FloorIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
    <line x1="5" y1="7" x2="19" y2="7" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    <line x1="5" y1="12" x2="19" y2="12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    <line x1="5" y1="17" x2="19" y2="17" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
  </svg>
);

const CeilingIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
    <path d="M5 7h14" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    <path d="M7 12h10" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeDasharray="3 2" />
    <path d="M9 17h6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
  </svg>
);

const InteriorWallIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
    <rect x="6.5" y="4.5" width="3.8" height="15" rx="0.9" stroke="currentColor" strokeWidth="1.8" />
    <rect x="13.7" y="4.5" width="3.8" height="15" rx="0.9" stroke="currentColor" strokeWidth="1.8" />
  </svg>
);

const ExteriorWallIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
    <rect x="4.5" y="4.5" width="4.2" height="15" rx="0.9" stroke="currentColor" strokeWidth="1.8" />
    <rect x="9.8" y="4.5" width="4.4" height="15" rx="0.9" stroke="currentColor" strokeWidth="1.2" strokeDasharray="3 2" />
    <rect x="14.8" y="4.5" width="4.7" height="15" rx="0.9" stroke="currentColor" strokeWidth="1.8" />
  </svg>
);

const RoofIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
    <path
      d="M4 11.5L12 5l8 6.5"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path d="M6.5 11.8V18h11v-6.2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
  </svg>
);

const CATEGORY_META: Record<CategoryKey, { label: string; short: string; color: string; Icon: () => JSX.Element }> = {
  windows: { label: "Fenster", short: "Fen", color: "#38bdf8", Icon: WindowIcon },
  doors: { label: "Türen", short: "Tür", color: "#f472b6", Icon: DoorIcon },
  floors: { label: "Böden", short: "Bod", color: "#facc15", Icon: FloorIcon },
  ceilings: { label: "Decken", short: "Deck", color: "#a855f7", Icon: CeilingIcon },
  interiorWalls: { label: "Innenwände", short: "In", color: "#34d399", Icon: InteriorWallIcon },
  exteriorWalls: { label: "Außenwände", short: "Au", color: "#f87171", Icon: ExteriorWallIcon },
  roofs: { label: "Dächer", short: "Dach", color: "#fb923c", Icon: RoofIcon },
};

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
  const ifcLoaderRef = useRef<IFCLoader | null>(null);
  const subsetsRef = useRef<Record<CategoryKey, THREE.Object3D | null>>(createCategoryState<THREE.Object3D | null>(null));
  const subsetMaterialsRef = useRef<Record<CategoryKey, THREE.Material | null>>(createCategoryState<THREE.Material | null>(null));
  const othersSubsetRef = useRef<THREE.Object3D | null>(null);
  const othersMaterialRef = useRef<THREE.Material | null>(null);

  const [status, setStatus] = useState<ViewerStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [visibility, setVisibility] = useState<Record<CategoryKey, boolean>>(createCategoryState(true));
  const [availableCategories, setAvailableCategories] = useState<Record<CategoryKey, boolean>>(createCategoryState(false));

  const clearSubsets = () => {
    const scene = sceneRef.current;
    CATEGORY_KEYS.forEach((key) => {
      const subset = subsetsRef.current[key];
      if (subset && scene) {
        scene.remove(subset);
      }
      if (subset) {
        disposeObject(subset);
      }
      subsetsRef.current[key] = null;
      const material = subsetMaterialsRef.current[key];
      material?.dispose?.();
      subsetMaterialsRef.current[key] = null;
    });
    if (othersSubsetRef.current && scene) {
      scene.remove(othersSubsetRef.current);
    }
    if (othersSubsetRef.current) {
      disposeObject(othersSubsetRef.current);
      othersSubsetRef.current = null;
    }
    if (othersMaterialRef.current) {
      othersMaterialRef.current.dispose();
      othersMaterialRef.current = null;
    }
    setAvailableCategories(createCategoryState(false));
  };

  const toggleCategory = (key: CategoryKey) => {
    if (!availableCategories[key]) return;
    setVisibility((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const sanitizeIds = (values: number[] | undefined): number[] =>
    Array.from(new Set((values ?? []).filter((id) => typeof id === "number" && id > 0)));

  const collectCategoryIds = async (manager: IFCLoader["ifcManager"], modelID: number) => {
    const ids = createEmptyCategoryIds();

    if (
      !manager ||
      typeof manager.getAllItemsOfType !== "function" ||
      !manager.ifcAPI ||
      typeof (manager.ifcAPI as any)?.GetLineIDsWithType !== "function"
    ) {
      console.warn("[ifc-viewer] IFC manager is not ready; skipping category extraction");
      return { categories: ids, others: [] };
    }

    const safeGetAllItems = async (type: number) => {
      try {
        const values = await manager.getAllItemsOfType(modelID, type, false);
        return sanitizeIds(values);
      } catch (error) {
        console.warn(`[ifc-viewer] getAllItemsOfType failed for type ${type}`, error);
        return [] as number[];
      }
    };

    const safeGetProperties = async (elementID: number) => {
      if (typeof manager.getItemProperties !== "function") {
        return null;
      }
      try {
        return await manager.getItemProperties(modelID, elementID, true, true);
      } catch (error) {
        console.warn(`[ifc-viewer] getItemProperties failed for id ${elementID}`, error);
        return null;
      }
    };

    ids.windows = await safeGetAllItems(IFCWINDOW);
    ids.doors = await safeGetAllItems(IFCDOOR);
    ids.roofs = await safeGetAllItems(IFCROOF);

    const slabCandidates = await safeGetAllItems(IFCSLAB);
    const floorBuffer: number[] = [];
    for (const id of slabCandidates) {
      try {
        const props = await safeGetProperties(id);
        const rawType = (props?.PredefinedType?.value ?? props?.PredefinedType ?? "") as string;
        const typeValue = String(rawType).toUpperCase();
        if (!typeValue || typeValue.includes("FLOOR") || typeValue.includes("BASESLAB")) {
          floorBuffer.push(id);
        }
      } catch {
        floorBuffer.push(id);
      }
    }
    ids.floors = sanitizeIds(floorBuffer.length ? floorBuffer : slabCandidates);

    const coveringCandidates = await safeGetAllItems(IFCCOVERING);
    const ceilingBuffer: number[] = [];
    for (const id of coveringCandidates) {
      try {
        const props = await safeGetProperties(id);
        const rawType = (props?.PredefinedType?.value ?? props?.PredefinedType ?? "") as string;
        const typeValue = String(rawType).toUpperCase();
        if (typeValue.includes("CEILING") || typeValue.includes("ROOF")) {
          ceilingBuffer.push(id);
        }
      } catch {
        // ignore and fallback later
      }
    }
    ids.ceilings = sanitizeIds(ceilingBuffer.length ? ceilingBuffer : coveringCandidates);

    const wallBase = new Set<number>([
      ...await safeGetAllItems(IFCWALL),
      ...await safeGetAllItems(IFCWALLSTANDARDCASE),
    ]);
    const interiorWallSet = new Set<number>();
    const exteriorWallSet = new Set<number>();
    const unknownWalls: number[] = [];

    for (const id of wallBase) {
      try {
        const props = await safeGetProperties(id);
        const raw = props?.IsExternal;
        let isExternal: boolean | null = null;
        if (typeof raw === "boolean") {
          isExternal = raw;
        } else if (raw && typeof raw === "object" && "value" in raw) {
          const value = (raw as any).value;
          if (typeof value === "boolean") {
            isExternal = value;
          } else if (typeof value === "number") {
            isExternal = Boolean(value);
          }
        }
        if (isExternal === true) {
          exteriorWallSet.add(id);
        } else if (isExternal === false) {
          interiorWallSet.add(id);
        } else {
          unknownWalls.push(id);
        }
      } catch {
        unknownWalls.push(id);
      }
    }

    if (unknownWalls.length) {
      if (interiorWallSet.size >= exteriorWallSet.size) {
        unknownWalls.forEach((id) => interiorWallSet.add(id));
      } else {
        unknownWalls.forEach((id) => exteriorWallSet.add(id));
      }
    }
    if (!interiorWallSet.size && !exteriorWallSet.size && wallBase.size) {
      wallBase.forEach((id) => {
        interiorWallSet.add(id);
        exteriorWallSet.add(id);
      });
    }
    ids.interiorWalls = Array.from(interiorWallSet);
    ids.exteriorWalls = Array.from(exteriorWallSet);

    const seen = new Set<number>();
    CATEGORY_KEYS.forEach((key) => {
      ids[key] = sanitizeIds(ids[key]);
      ids[key].forEach((id) => seen.add(id));
    });

    const allProducts = sanitizeIds(await manager.getAllItemsOfType(modelID, IFCPRODUCT, false));
    const others = allProducts.filter((id) => !seen.has(id));

    return { categories: ids, others };
  };

  const prepareCategorySubsets = async (model: any) => {
    const loader = ifcLoaderRef.current;
    const scene = sceneRef.current;
    if (!loader || !scene || !model) return;

    clearSubsets();
    const { categories, others } = await collectCategoryIds(loader.ifcManager, model.modelID);
    if (!sceneRef.current) return;

    const availability = createCategoryState(false);

    CATEGORY_KEYS.forEach((key) => {
      const ids = categories[key];
      if (!ids.length) {
        subsetsRef.current[key] = null;
        subsetMaterialsRef.current[key]?.dispose?.();
        subsetMaterialsRef.current[key] = null;
        return;
      }

      availability[key] = true;
      subsetMaterialsRef.current[key]?.dispose?.();
      const material = new THREE.MeshPhongMaterial({
        color: new THREE.Color(CATEGORY_META[key].color),
        transparent: true,
        opacity: 0.92,
        shininess: 65,
        depthWrite: false,
      });
      subsetMaterialsRef.current[key] = material;
      const subset = loader.ifcManager.createSubset({
        modelID: model.modelID,
        ids,
        removePrevious: true,
        material,
        scene,
        customID: `subset-${key}`,
      });
      subset.visible = visibility[key];
      subset.renderOrder = 2;
      subsetsRef.current[key] = subset;
    });

    setAvailableCategories(availability);

    if (others.length) {
      othersMaterialRef.current?.dispose?.();
      const material = new THREE.MeshPhongMaterial({
        color: new THREE.Color("#64748b"),
        transparent: true,
        opacity: 0.35,
        depthWrite: false,
      });
      othersMaterialRef.current = material;
      const subset = loader.ifcManager.createSubset({
        modelID: model.modelID,
        ids: others,
        removePrevious: true,
        material,
        scene,
        customID: "subset-others",
      });
      subset.visible = true;
      subset.renderOrder = 1;
      othersSubsetRef.current = subset;
    }

    model.visible = false;
  };

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
      clearSubsets();
      ifcLoaderRef.current = null;
      setVisibility(createCategoryState(true));
      if (modelRef.current && sceneRef.current) {
        sceneRef.current.remove(modelRef.current);
        disposeObject(modelRef.current);
        modelRef.current = null;
      }
      return;
    }

    const scene = sceneRef.current;
    if (!scene) return;

    let loader: IFCLoader | null = null;

    setStatus("loading");
    setError(null);

    const loadModel = async () => {
      try {
        if (modelRef.current) {
          scene.remove(modelRef.current);
          disposeObject(modelRef.current);
          modelRef.current = null;
        }
        clearSubsets();

        const { IFCLoader } = await import("web-ifc-three/IFCLoader");
        loader = new IFCLoader();
        loader.ifcManager.setWasmPath(WASM_PATH, true);
        ifcLoaderRef.current = loader;

        const model = await loader.loadAsync(url);
        model.castShadow = true;
        model.receiveShadow = true;
        scene.add(model);
        modelRef.current = model;

        centerModel(model, cameraRef.current, controlsRef.current);
        await prepareCategorySubsets(model);

        setStatus("ready");
      } catch (err: any) {
        console.error("[ifc-viewer] IFC konnte nicht geladen werden", err);
        setError(err?.message ?? "IFC konnte nicht geladen werden.");
        setStatus("error");
        clearSubsets();
        ifcLoaderRef.current = null;
      }
    };

    loadModel();

    return () => {
      clearSubsets();
      ifcLoaderRef.current = null;
      try {
        loader?.ifcManager?.dispose?.();
      } catch {}
    };
  }, [ifcUrl]);

  useEffect(() => {
    CATEGORY_KEYS.forEach((key) => {
      const subset = subsetsRef.current[key];
      if (subset) {
        subset.visible = visibility[key];
      }
    });
  }, [visibility]);

  const displayHeight = typeof height === "number" ? `${height}px` : height;

  return (
    <div ref={containerRef} className={`relative w-full overflow-hidden rounded-xl border border-white/10 bg-slate-950/80 ${className}`} style={{ height: displayHeight }}>
      <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />
      <div className="pointer-events-none absolute top-3 right-3 flex flex-col items-end gap-2">
        <span className="pointer-events-none text-[10px] uppercase tracking-[0.4em] text-white/45">Bauteile</span>
        <div className="pointer-events-auto rounded-2xl bg-slate-950/85 p-2 shadow-[0_18px_38px_-22px_rgba(59,130,246,0.7)] backdrop-blur-md">
          <div className="grid grid-cols-2 gap-2">
            {CATEGORY_KEYS.map((key) => {
              const { Icon, label, short, color } = CATEGORY_META[key];
              const active = visibility[key];
              const available = availableCategories[key];
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => toggleCategory(key)}
                  disabled={!available}
                  className={`group relative flex h-16 w-16 flex-col items-center justify-center gap-2 rounded-2xl border text-[9px] font-semibold uppercase tracking-[0.4em] transition-all duration-200 ${
                    !available
                      ? "cursor-not-allowed border-white/10 bg-slate-900/30 text-slate-500"
                      : active
                      ? "border-transparent bg-cyan-500/15 shadow-[0_16px_34px_-18px_rgba(14,165,233,0.75)]"
                      : "border-white/20 bg-slate-900/70 hover:border-cyan-300/60 hover:bg-slate-800/70"
                  }`}
                  style={{ color: active && available ? color : "#cbd5f5" }}
                  title={label}
                >
                  <span
                    className={`transition-transform duration-300 ${
                      active && available ? "scale-110 drop-shadow-[0_0_12px_rgba(56,189,248,0.55)]" : "scale-95 opacity-70"
                    }`}
                  >
                    <Icon />
                  </span>
                  <span className="text-[8px] uppercase tracking-[0.35em] text-white/70 group-hover:text-white">
                    {short}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      </div>
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

