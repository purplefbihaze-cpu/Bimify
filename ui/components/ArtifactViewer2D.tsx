import { useEffect, useMemo, useState } from "react";
import CanvasGeoViewer, { CanvasLayer } from "@/components/CanvasGeoViewer";

type FeatureCollection = { type: "FeatureCollection"; features: any[] };

type Props = {
  wallsUrl?: string;
  axesUrl?: string;
  spacesUrl?: string;
  enabled: { walls: boolean; walls_axes: boolean; spaces: boolean };
  pageIndex?: number | null;
};

export default function ArtifactViewer2D({ wallsUrl, axesUrl, spacesUrl, enabled, pageIndex }: Props) {
  const [rawWalls, setRawWalls] = useState<FeatureCollection | null>(null);
  const [rawAxes, setRawAxes] = useState<FeatureCollection | null>(null);
  const [spaces, setSpaces] = useState<FeatureCollection | null>(null);
  const [hover, setHover] = useState<{ layerId: string; feature: any } | null>(null);

  useEffect(() => {
    (async () => {
      if (wallsUrl) setRawWalls(await fetchGeoJson(wallsUrl));
      if (axesUrl) setRawAxes(await fetchGeoJson(axesUrl));
      if (spacesUrl) setSpaces(await fetchGeoJson(spacesUrl));
    })();
  }, [wallsUrl, axesUrl, spacesUrl]);

  const walls = useMemo(() => filterByPage(rawWalls, pageIndex), [rawWalls, pageIndex]);
  const axes = useMemo(() => filterByPage(rawAxes, pageIndex), [rawAxes, pageIndex]);

  const layers = useMemo<CanvasLayer[]>(() => {
    const items: CanvasLayer[] = [];
    if (spaces && enabled.spaces) {
      items.push({
        id: "spaces",
        label: "Spaces",
        data: spaces,
        stroke: "rgba(111, 206, 255, 0.7)",
        fill: "rgba(111, 206, 255, 0.1)",
        lineWidth: 1.4,
        zIndex: 1,
      });
    }
    if (walls && enabled.walls) {
      items.push({
        id: "walls",
        label: "Walls",
        data: walls,
        stroke: "rgba(255,255,255,0.82)",
        lineWidth: 2,
        zIndex: 2,
      });
    }
    if (axes && enabled.walls_axes) {
      items.push({
        id: "walls_axes",
        label: "Wall Axes",
        data: axes,
        stroke: "rgba(159,209,255,0.85)",
        lineWidth: 2.2,
        dash: [8, 8],
        zIndex: 3,
      });
    }
    return items;
  }, [spaces, walls, axes, enabled]);

  return (
    <CanvasGeoViewer
      layers={layers}
      activeTool="pan"
      onHover={setHover}
      overlay={
        hover ? (
          <div className="absolute right-4 top-4 glass rounded-lg px-4 py-3 text-xs shadow-soft">
            <div className="uppercase tracking-[0.2em] text-[10px] opacity-70">{hover.layerId}</div>
            <pre className="mt-2 max-h-32 w-52 overflow-auto text-[11px] text-accent-300">
              {JSON.stringify(hover.feature.properties ?? {}, null, 2)}
            </pre>
          </div>
        ) : null
      }
    />
  );
}

async function fetchGeoJson(url: string): Promise<FeatureCollection | null> {
  const resp = await fetch(url);
  if (!resp.ok) return null;
  return (await resp.json()) as FeatureCollection;
}

function filterByPage(fc: FeatureCollection | null, pageIndex: number | null | undefined): FeatureCollection | null {
  if (!fc || pageIndex == null) return fc;
  const filtered = fc.features.filter((feature) => {
    const props = feature.properties || {};
    const value = props.page_index ?? props.page;
    if (value == null) return true;
    return Number(value) === Number(pageIndex);
  });
  return { ...fc, features: filtered };
}


