# IFC Export V2

## Übersicht

IFC Export V2 ist eine verbesserte Implementierung des IFC-Exports mit Post-Processing-Pipeline zur Bereinigung von Pixel-Rauschen aus Roboflow-Vorhersagen.

## Pipeline

1. **Post-Processing**: Bereinigung von Pixel-Rauschen
   - Polygon-Simplification (Douglas-Peucker)
   - Snap-to-Grid
   - Rechtwinkligkeits-Enforcement
   - Kontext-basierte Korrekturen

2. **Viewer-Generierung**: HTML-Viewer für bereinigte Predictions

3. **Normalisierung**: Konvertierung zu NormalizedDet-Format

4. **Raum-Polygonisierung**: Generierung von Räumen aus Wänden

5. **IFC-Export**: Erstellung der IFC-Datei mit:
   - Wänden, Türen, Fenstern
   - Räumen
   - Slabs (Böden)
   - Validation Reports
   - TopView GeoJSON
   - Template-Vergleich

## API

### Endpoint

```
POST /export-ifc/v2/async
```

### Request

```typescript
{
  predictions: Prediction[];
  storey_height_mm: number;
  door_height_mm: number;
  window_height_mm?: number;
  window_head_elevation_mm?: number;
  floor_thickness_mm?: number;
  px_per_mm?: number;
  project_name?: string;
  storey_name?: string;
  calibration?: CalibrationPayload;
  flip_y?: boolean;
  image_height_px?: number;
}
```

### Response

```typescript
{
  ifc_url: string;
  file_name: string;
  viewer_url?: string;  // HTML-Viewer für bereinigte Predictions
  topview_url?: string;
  validation_report_url?: string;
  comparison_report_url?: string;
  storey_height_mm: number;
  door_height_mm: number;
  window_height_mm?: number;
  window_head_elevation_mm: number;
  px_per_mm?: number;
  warnings?: string[];
}
```

## Unterschiede zu V1

- **Post-Processing-Pipeline**: Automatische Bereinigung von Pixel-Rauschen
- **Viewer-Integration**: Visualisierung der bereinigten Predictions vor Export
- **Verbesserte Geometrie**: Snap-to-Grid und Rechtwinkligkeits-Enforcement
- **Bessere Fehlerbehandlung**: Spezifischere Fehlermeldungen

## Verwendung

```typescript
const response = await exportIfcV2Async({
  predictions: geometryResult.predictions,
  storey_height_mm: 3000,
  door_height_mm: 2100,
  window_height_mm: 1000,
  px_per_mm: 1.0,
});
```

## Dateien

- `services/api/ifc_exporter_v2.py` - Hauptexport-Funktion
- `core/ifc/build_ifc43_model_v2.py` - IFC-Modell-Builder
- `core/ml/postprocess_v2.py` - Post-Processing-Pipeline
- `core/ifc/viewer_v2.py` - Viewer-Generator










