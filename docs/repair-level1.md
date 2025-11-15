# Repair Level 1 – Vorschau und Commit

Diese Stufe erstellt aus RF-TopView-Geometrien geglättete, achsengeschnappte Wände mit konsistenter Dicke und ausgerichteten Öffnungen. Zusätzlich werden Bildkanten (optional) für einen finalen Fit herangezogen.

## API

- Vorschau:
  - `POST /v1/ifc/repair/preview`
  - Request: `{ file_name | ifc_url | job_id, image_url?, level }`
  - Response: `{ preview_id, overlay_url, heatmap_url?, proposed_topview_url?, metrics }`
- Commit:
  - `POST /v1/ifc/repair/commit`
  - Request: `{ preview_id }` (oder gleiche Quellen wie Vorschau)
  - Response: `{ file_name, ifc_url, level, topview_url?, warnings? }`

## Einstellungen

`config/default.yaml` (Beispiele):

```yaml
geometry:
  repair_level1:
    posTol_mm: 20
    angleTol_deg: 5
    simplifyTol_mm: 20
    areaThreshold_m2: 0.05
    branchMin_mm: 150
    minOverlap_mm: 50
    canny_low: 50
    canny_high: 150
    hough_threshold: 80
    hough_min_line_length_mm: 200
    hough_max_line_gap_mm: 20
```

## Akzeptanzkriterien

- Median IoU (Wände): ≥ 0.90 (TopView-Vergleich)
- Öffnungen bündig ausgerichtet (Tiefe = Wanddicke bei Level 1)
- HottCAD/IFC-Validierung: keine CRITICAL-Fehler

## Artefakte

- Vorschau-Overlay: `data/previews/repair_preview_<id>.geojson`
- Reparierte IFC: `data/exports/<file>_repair_L1.ifc`
- TopView: `data/exports/<file>_repair_L1_topview.geojson`
- Validation: `data/exports/<file>_repair_L1_validation.json`


