export type ZoneDefinition = {
  name: string;
  points: Array<[number, number]>;
};

export type LineDefinition = {
  name: string;
  start: [number, number];
  end: [number, number];
};

export type AnalyzeOptions = {
  confidence?: number;
  overlap?: number;
  per_class_thresholds?: Record<string, number> | null;
  zones?: ZoneDefinition[] | null;
  lines?: LineDefinition[] | null;
};

export type Prediction = {
  id?: string | null;
  class: string;
  confidence: number;
  x?: number | null;
  y?: number | null;
  width?: number | null;
  height?: number | null;
  points?: Array<[number, number]> | null;
  raw: Record<string, any>;
};

export type CalibrationPayload = {
  px_per_mm: number;
  pixel_distance: number;
  real_distance_mm: number;
  point_a: [number, number];
  point_b: [number, number];
  unit: "mm" | "m";
};

export type ZoneCount = {
  name: string;
  total: number;
  per_class: Record<string, number>;
};

export type LineCount = {
  name: string;
  counts: Record<string, number>;
  per_class: Record<string, Record<string, number>>;
};

export type AnalyzeResponse = {
  model_id: string;
  confidence: number;
  overlap: number;
  total: number;
  per_class: Record<string, number>;
  predictions: Prediction[];
  zones?: ZoneCount[] | null;
  lines?: LineCount[] | null;
  annotated_image?: string | null;
  image?: Record<string, any> | null;
  raw: Record<string, any>;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const DEFAULT_TIMEOUT_MS = Number(process.env.NEXT_PUBLIC_API_TIMEOUT_MS || 180000);

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort("timeout"), DEFAULT_TIMEOUT_MS);
  try {
    const resp = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        ...(init?.headers || {}),
      },
      signal: controller.signal,
    });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`${resp.status} ${resp.statusText}: ${text}`);
    }
    return (await resp.json()) as T;
  } catch (err: any) {
    if (err?.name === "AbortError" || err === "timeout") {
      throw new Error("Zeitüberschreitung beim Server – bitte erneut versuchen.");
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function analyzeImage(file: File, options: AnalyzeOptions): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("file", file);
  if (options && Object.keys(options).length > 0) form.append("options", JSON.stringify(options));
  const resp = await fetch(`${API_BASE}/v1/analyze`, { method: "POST", body: form });
  if (!resp.ok) throw new Error(await resp.text());
  return (await resp.json()) as AnalyzeResponse;
}

export async function health(): Promise<{ status: string }> {
  return await http(`/healthz`);
}


export type SettingsPayload = {
  roboflow_api_key?: string | null;
  has_roboflow_api_key?: boolean | null;
  roboflow_project?: string | null;
  roboflow_version?: number | null;
  matrix_enabled?: boolean | null;
  matrix_speed?: number | null;
  matrix_density?: number | null;
  matrix_opacity?: number | null;
  matrix_color?: string | null;
};

export async function getSettings(): Promise<SettingsPayload> {
  return await http<SettingsPayload>(`/v1/settings`);
}

export async function saveSettings(payload: SettingsPayload): Promise<SettingsPayload> {
  return await http<SettingsPayload>(`/v1/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}


export type ExportIFCRequestPayload = {
  predictions: Prediction[];
  image?: Record<string, any> | null;
  storey_height_mm: number;
  door_height_mm: number;
  window_height_mm?: number | null;
  window_head_elevation_mm?: number;
  px_per_mm?: number | null;
  project_name?: string | null;
  storey_name?: string | null;
  calibration?: CalibrationPayload | null;
};

export type ExportIFCResponse = {
  ifc_url: string;
  improved_ifc_url?: string | null;
  improved_ifc_stats_url?: string | null;
  improved_wexbim_url?: string | null;
  file_name: string;
  storey_height_mm: number;
  door_height_mm: number;
  window_height_mm?: number | null;
  window_head_elevation_mm: number;
  px_per_mm?: number | null;
  warnings?: string[] | null;
};

export async function exportIfc(payload: ExportIFCRequestPayload): Promise<ExportIFCResponse> {
  return await http<ExportIFCResponse>(`/v1/export-ifc`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export type ExportIFCJobResponse = {
  job_id: string;
};

export async function exportIfcAsync(payload: ExportIFCRequestPayload): Promise<ExportIFCJobResponse> {
  return await http<ExportIFCJobResponse>(`/v1/export-ifc/async`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export type JobStatus = "queued" | "running" | "succeeded" | "failed";

export type JobStatusResponse = {
  id: string;
  status: JobStatus;
  progress: number;
  result?: ExportIFCResponse | null;
  error?: string | null;
};

export async function getJob(jobId: string): Promise<JobStatusResponse> {
  return await http<JobStatusResponse>(`/v1/jobs/${jobId}`);
}



export type IfcTopViewRequestPayload = {
  file_name?: string | null;
  ifc_url?: string | null;
  section_elevation_mm?: number | null;
};

export type IfcTopViewResponsePayload = {
  topview_url: string;
  file_name: string;
};

export async function getIfcTopView(payload: IfcTopViewRequestPayload): Promise<IfcTopViewResponsePayload> {
  return await http<IfcTopViewResponsePayload>(`/v1/ifc/topview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export type IfcRepairRequestPayload = {
  file_name?: string | null;
  ifc_url?: string | null;
  level?: number;
};

export type IfcRepairResponsePayload = {
  file_name: string;
  ifc_url: string;
  level: number;
  topview_url?: string | null;
  warnings?: string[] | null;
};

export async function repairIfc(payload: IfcRepairRequestPayload): Promise<IfcRepairResponsePayload> {
  return await http<IfcRepairResponsePayload>(`/v1/ifc/repair`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ level: 1, ...payload }),
  });
}

export type ExportPDFOptionsPayload = {
  mode?: "wall-fill" | "centerline";
  smooth_tolerance_mm?: number;
  snap_tolerance_mm?: number;
  orthogonal_tolerance_deg?: number;
  include_background?: boolean;
};

export type ExportPDFRequestPayload = {
  predictions: Prediction[];
  image?: Record<string, any> | null;
  px_per_mm?: number | null;
  options?: ExportPDFOptionsPayload | null;
};

export type ExportPDFResponse = {
  pdf_url: string;
  file_name: string;
  warnings?: string[] | null;
};

export async function exportVectorPdf(payload: ExportPDFRequestPayload): Promise<ExportPDFResponse> {
  return await http<ExportPDFResponse>(`/v1/export-pdf`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}


export type HottCADBaseRequestPayload = {
  ifc_url?: string | null;
  job_id?: string | null;
  tolerance_mm?: number;
};

export type HottCADCheck = {
  id: string;
  title: string;
  status: "ok" | "warn" | "fail";
  details: string[];
  affected: Record<string, string[]>;
};

export type HottCADMetrics = {
  wall_count: number;
  interior_walls: number;
  exterior_walls: number;
  walls_with_rectangular_footprint: number;
  walls_with_constant_thickness: number;
  openings_with_relations: number;
  spaces: number;
  floors: number;
  roofs: number;
  connects_relations: number;
  space_boundaries: number;
  material_layer_usages: number;
  avg_wall_thickness_mm?: number | null;
};

export type HottCADFileInfo = {
  schema: string;
  path?: string | null;
  sizeBytes?: number | null;
  isPlainIFC?: boolean | null;
};

export type HottCADValidationResponse = {
  schema: string;
  file_info: HottCADFileInfo;
  checks: HottCADCheck[];
  metrics: HottCADMetrics;
  score: number;
  highlightSets: HottCADHighlight[];
};

export async function validateHottCAD(payload: HottCADBaseRequestPayload): Promise<HottCADValidationResponse> {
  return await http<HottCADValidationResponse>(`/v1/hottcad/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export type HottCADConnection = {
  walls: string[];
  distanceMm: number;
  contactType: "touch" | "gap" | "overlap";
  notes: string[];
};

export type HottCADSpaceBoundary = {
  walls: string[];
  spaces: string[];
  note?: string | null;
};

export type HottCADMaterialSuggestion = {
  wall: string;
  thicknessMm?: number | null;
  note?: string | null;
};

export type HottCADSimulationProposed = {
  connects: HottCADConnection[];
  spaceBoundaries: HottCADSpaceBoundary[];
  materials: HottCADMaterialSuggestion[];
};

export type HottCADCompleteness = {
  roomsClosed: boolean;
  gapCount: number;
  spaces: number;
  walls: number;
};

export type HottCADHighlight = {
  id: string;
  label: string;
  guids: string[];
  productIds: number[];
};

export type HottCADSimulationResponse = {
  proposed: HottCADSimulationProposed;
  completeness: HottCADCompleteness;
  highlightSets: HottCADHighlight[];
};

export async function simulateHottCAD(payload: HottCADBaseRequestPayload): Promise<HottCADSimulationResponse> {
  return await http<HottCADSimulationResponse>(`/v1/hottcad/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

