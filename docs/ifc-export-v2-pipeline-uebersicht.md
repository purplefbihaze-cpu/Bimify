# IFC Export V2 Pipeline - Detaillierte Übersicht (Aktualisiert)

## Pipeline-Ablauf (End-to-End mit Verbesserungen)

### 0. Initialisierung
- **PipelineConfig**: Zentrale Konfiguration mit Pydantic-Validierung
  - Post-Processing Parameter (Toleranzen, Grid-Sizes)
  - Validierungs-Schwellenwerte
  - Performance-Parameter (Parallelisierung, Caching)
  - Feature Flags (enable_snap_to_grid, enable_right_angle, etc.)
- **PipelineMetrics**: Metriken-Sammlung initialisiert
  - Performance-Tracking
  - Qualitäts-Metriken
  - Warnungen/Fehler-Sammlung

### 1. Eingabe-Verarbeitung
- **Input**: Roboflow Predictions (JSON mit Polygon-Punkten)
- **Parameter-Extraktion**:
  - Kalibrierung (px_per_mm, flip_y, image_height_px)
  - Projektname, Geschossname
  - Höhen (Geschoss, Tür, Fenster, Boden)
  - Pixel-zu-Millimeter-Konvertierung
- **Konvertierung**: Predictions zu dict-Format für Post-Processing

### 1.5. Eingabe-Validierung (NEU)
**Zweck**: Frühe Erkennung defekter Predictions, verhindert Pipeline-Fehler

- **Geometrie-Checks**:
  - Self-intersecting Polygone (Shapely `is_valid`)
  - Leere/ungültige Polygone
  - Polygon-Reparatur-Versuche
- **Mindestpunktzahl**:
  - Wände: ≥ 4 Punkte (konfigurierbar)
  - Türen: ≥ 3 Punkte (konfigurierbar)
  - Fenster: ≥ 3 Punkte (konfigurierbar)
- **Dimensionalitäts-Prüfung**:
  - Max Wandlänge: 100m (konfigurierbar)
  - Min Wandlänge: 0.1m (konfigurierbar)
  - Warnung bei sehr dünnen Wänden
- **Confidence Score Filter**:
  - Konfigurierbarer Threshold (Standard: 0.0 = kein Filter)
  - Filtert Predictions mit zu niedrigem Confidence
- **Statistiken**: Tracking von gefilterten Predictions nach Kategorie
- **Metriken**: Input-Validierung wird in PipelineMetrics erfasst

### 2. Post-Processing Pipeline (Rauschen-Bereinigung)

#### Schritt 2.1: Polygon-Simplification
- **Douglas-Peucker Algorithmus**
- Toleranz für Wände: 2.0 mm (aus Config)
- Toleranz für Türen/Fenster: 0.5 mm (aus Config)
- Entfernt unnötige Polygon-Punkte
- Topologie-Erhaltung
- Automatische Reparatur bei ungültigen Geometrien

#### Schritt 2.2: Snap-to-Grid (NEU - Adaptiv)
- **Multi-Level-Grid** basierend auf Objektklasse:
  - Wände: 50mm (aus Config: `grid_size_walls`)
  - Türen: 10mm (aus Config: `grid_size_doors`)
  - Fenster: 10mm (aus Config: `grid_size_windows`)
- **Feature Flag**: `enable_snap_to_grid` (Standard: True)
- **Adaptive Auswahl**: Automatisch basierend auf Objektklasse
- Alle Punkte werden auf Raster ausgerichtet
- Reduziert Ungenauigkeiten

#### Schritt 2.3: Rechtwinkligkeits-Enforcement
- **90°-Winkel-Erzwingung**
- Toleranz: 5 Grad (aus Config: `angle_tolerance`)
- **Feature Flag**: `enable_right_angle` (Standard: True)
- Für Wände, Türen, Fenster
- Macht Geometrien rechtwinklig

#### Schritt 2.4: Kontext-basierte Korrekturen
- **Tür-zu-Wand-Anpassung**:
  - Türen werden an nächste Wand gesnappt
  - Maximale Distanz: 100 mm (aus Config: `door_wall_max_distance`)
  - Automatische Ausrichtung
- **Fenster-zu-Wand-Clipping**:
  - Fenster werden auf Wand-Grenzen beschnitten
  - Verhindert Überlappungen
- **Wand-Gap-Closing**:
  - Schließt kleine Lücken zwischen Wänden
  - Maximale Gap-Größe: 10 mm (aus Config: `max_gap_close`)

#### Schritt 2.5: Parallelisierung (NEU - Automatisch)
- **Automatische Erkennung**: >100 Predictions (aus Config: `parallel_processing_threshold`)
- **Feature Flag**: `enable_parallel_processing` (Standard: True)
- **Gruppierung**: Nach Objektklasse (Wände, Türen, Fenster) für bessere Cache-Lokalität
- **Vorbereitet**: Für ProcessPoolExecutor (benötigt Refaktorierung für Pickling)

### 3. Viewer-Generierung (NEU - In-Memory)
- **HTML-Viewer** für bereinigte Predictions
- **In-Memory-Generierung**: Viewer wird im Speicher erstellt
- **Schreib-Strategie**: Nur bei erfolgreichem IFC-Export auf Platte
- Visualisierung vor IFC-Export
- Zeigt Shapes und Labels
- Datei: `{uuid}_viewer.html` (nur bei Erfolg)

### 4. Normalisierung
- **Konvertierung zu NormalizedDet-Format**
- Pixel-zu-Millimeter-Umrechnung
- Y-Achsen-Spiegelung (optional)
- Klassifikation: WALL, DOOR, WINDOW, STAIR
- Geometrie-Validierung
- **Metriken**: Normalisierungs-Zeit wird erfasst

### 5. Geometrie-Rekonstruktion

#### Schritt 5.1: Wandachsen-Erzeugung (NEU - Optimiert)
- **Skeletonization** (Mittellinien-Extraktion)
- **Downsampling**: Bei großen Masken (>1000px, aus Config: `skeletonization_max_dimension`)
  - Target DPI: 100 (aus Config: `skeletonization_target_dpi`)
  - Koordinaten werden wieder hochskaliert
- **Caching-Vorbereitung**: Hash-Funktion für identische Polygone
- **Feature Flag**: `enable_skeletonization` (Standard: True)
- **Cache-Flag**: `enable_skeletonization_cache` (Standard: True)
- Wanddicke-Berechnung
- Einheitliche Wanddicke pro Wand
- **Artefakt**: `walls_axes.geojson`

#### Schritt 5.2: Raum-Polygonisierung
- **Generierung von Räumen aus Wänden**
- Wall Union → Difference
- Automatische Raum-Erkennung
- **Artefakt**: `spaces.geojson`

#### Schritt 5.3: Geometrie-Validierung (NEU)
**Zweck**: Qualitätsprüfung vor IFC-Export

- **Wandachsen-Parallelität**:
  - Prüft ob Achsen parallel zu Ursprungspolygonen sind
  - Toleranz: 10 Grad
  - Warnung bei >30 Grad Abweichung
- **Raum-Polygon-Geschlossenheit**:
  - Prüft ob alle Räume geschlossen sind (außer 1A)
  - Automatische Reparatur-Versuche
  - Fehler bei nicht reparierbaren Räumen
- **Überlappende Räume**:
  - Topologie-Check für Raum-Überlappungen
  - Warnung bei >1% Überlappung
- **Mindestfläche**:
  - Standard: 1m² (aus Config: `min_room_area_m2`)
  - Warnung bei Räumen unter Mindestfläche
- **Wandachsen-Qualität**:
  - Skeletonization-Quality-Score (IoU-basiert)
  - Durchschnittliche Qualität wird berechnet
- **Feature Flag**: `enable_geometry_validation` (Standard: True)
- **Metriken**: Alle Validierungs-Ergebnisse werden erfasst

### 6. IFC-Export

#### Schritt 6.1: IFC-Modell-Erstellung
- **IFC 4.3 Schema**
- Projekt-Struktur:
  - IfcProject
  - IfcSite
  - IfcBuilding
  - IfcBuildingStorey
- Einheiten-Setup (Millimeter)
- Geometrischer Kontext

#### Schritt 6.2: Wände
- **IfcWallStandardCase**
- Extrudierte Geometrie aus Polygonen
- Wanddicke aus Achsen-Berechnung
- Innen/Außen-Klassifikation
- Property Sets (Pset_BuildingElementCommon)

#### Schritt 6.3: Türen
- **IfcDoor**
- Öffnungen in Wänden (IfcOpeningElement)
- Höhe: door_height_mm
- Automatische Wand-Zuordnung
- Property Sets

#### Schritt 6.4: Fenster
- **IfcWindow**
- Öffnungen in Wänden
- Höhe: window_height_mm
- Sturzhöhe: window_head_elevation_mm
- Automatische Wand-Zuordnung
- Property Sets

#### Schritt 6.5: Räume
- **IfcSpace**
- Extrudierte Geometrie aus Raum-Polygonen
- Höhe: storey_height_mm
- Property Sets (Pset_SpaceCommon)
- Flächen- und Volumen-Berechnung

#### Schritt 6.6: Böden (Slabs)
- **IfcSlab**
- Erzeugung aus Außenwänden
- Boden-Dicke: floor_thickness_mm
- Automatische Umrandung

### 7. Validierung & Reports

#### Schritt 7.1: IFC-Parse-Validierung
- **ifcopenshell.open()** Test
- Prüft ob Datei lesbar ist
- Fehlerbehandlung bei Parse-Fehlern

#### Schritt 7.2: Validation Report
- **Geometrie-Validierung**:
  - Wand-Validierung
  - Achsen-Validierung
  - Raum-Validierung
- Auto-Repair-Mechanismen
- **Artefakt**: `{filename}_validation.json`
- **Metriken**: Validierungs-Zeit wird erfasst

#### Schritt 7.3: TopView GeoJSON (NEU - Nur bei Erfolg)
- **2D-Ansicht** aus IFC-Modell
- Schnitt auf Geschoss-Ebene
- GeoJSON-Format
- **Schreib-Strategie**: Nur nach erfolgreichem IFC-Export
- **Artefakt**: `{filename}_topview.geojson`

#### Schritt 7.4: Template-Vergleich
- **Vergleich mit Beispiel-IFC**
- Property Sets-Vergleich
- Fehlende/Extra Property Sets
- **Artefakt**: `{filename}_comparison.json`

### 8. Metriken & Monitoring (NEU)
- **PipelineMetrics**: Vollständige Metriken-Sammlung
  - Input-Statistiken (valid/invalid, gefiltert nach Kategorie)
  - Post-Processing-Statistiken (korrigiert, gefixt)
  - Rekonstruktions-Statistiken (Wände, Türen, Fenster, Räume)
  - Wandachsen-Statistiken (parallel, Qualität)
  - Performance-Metriken (Zeit pro Schritt)
  - Warnungen/Fehler (nach Kategorie)
- **Quality Score**: Berechnung basierend auf:
  - Input-Validierung Pass-Rate
  - Raum-Geschlossenheits-Rate
  - Wandachsen-Parallelität
  - Durchschnittliche Skeletonization-Qualität
  - Fehler-Penalty
- **Summary**: Key-Metriken für Logging
- **Vorbereitung**: Für Prometheus/Grafana Integration

### 9. Ausgabe
- **IFC-Datei**: `{uuid}.ifc`
- **Viewer-URL**: `/files/{filename}_viewer.html` (nur bei Erfolg)
- **TopView-URL**: `/files/{filename}_topview.geojson` (nur bei Erfolg)
- **Validation-URL**: `/files/{filename}_validation.json`
- **Comparison-URL**: `/files/{filename}_comparison.json`
- **Warnings**: Liste von Warnungen (optional)
- **Metriken**: In-Memory gesammelt (kann später exportiert werden)

## Technische Details

### Konfiguration (PipelineConfig)
- **Post-Processing**:
  - `simplify_tolerance_walls`: 2.0 mm
  - `simplify_tolerance_doors_windows`: 0.5 mm
  - `grid_size_walls`: 50 mm
  - `grid_size_doors`: 10 mm
  - `grid_size_windows`: 10 mm
  - `angle_tolerance`: 5 Grad
  - `max_gap_close`: 10 mm
  - `door_wall_max_distance`: 100 mm
- **Validierung**:
  - `min_confidence_score`: 0.0 (kein Filter)
  - `min_points_wall`: 4
  - `min_points_door`: 3
  - `min_points_window`: 3
  - `max_wall_length_mm`: 100000.0 (100m)
  - `min_wall_length_mm`: 100.0 (0.1m)
  - `min_room_area_m2`: 1.0
- **Performance**:
  - `parallel_processing_threshold`: 100 Predictions
  - `skeletonization_max_dimension`: 1000 px
  - `skeletonization_target_dpi`: 100.0
  - `enable_skeletonization_cache`: True
  - `skeletonization_cache_size`: 128
- **Feature Flags**:
  - `enable_snap_to_grid`: True
  - `enable_right_angle`: True
  - `enable_skeletonization`: True
  - `enable_parallel_processing`: True
  - `enable_adaptive_grid`: True
  - `enable_input_validation`: True
  - `enable_geometry_validation`: True

### Dateien
- `services/api/ifc_exporter_v2.py` - Hauptexport-Funktion (mit allen Verbesserungen)
- `core/ml/pipeline_config.py` - Zentrale Konfiguration (NEU)
- `core/ml/postprocess_v2.py` - Post-Processing-Pipeline (mit adaptivem Grid)
- `core/ml/postprocess_floorplan.py` - Skeletonization (mit Downsampling)
- `core/ifc/build_ifc43_model_v2.py` - IFC-Modell-Builder
- `core/ifc/viewer_v2.py` - Viewer-Generator (In-Memory)
- `core/validate/input_validation.py` - Eingabe-Validierung (NEU)
- `core/validate/geometry_validation.py` - Geometrie-Validierung (NEU)
- `core/metrics/pipeline_metrics.py` - Metriken-Sammlung (NEU)

### Unterschiede zu V1
- ✅ Post-Processing-Pipeline (Rauschen-Bereinigung)
- ✅ Viewer-Integration (Visualisierung vor Export)
- ✅ Verbesserte Geometrie (Snap-to-Grid, Rechtwinkligkeit)
- ✅ Bessere Fehlerbehandlung
- ✅ Template-Vergleich
- ✅ **Zentrale Konfiguration** mit Pydantic-Validierung
- ✅ **Eingabe-Validierung** nach Schritt 1
- ✅ **Adaptives Snap-to-Grid** (Multi-Level)
- ✅ **Geometrie-Validierung** vor IFC-Export
- ✅ **Skeletonization-Optimierung** (Downsampling)
- ✅ **File I/O Optimierung** (In-Memory, nur bei Erfolg)
- ✅ **Automatische Parallelisierung** (>100 Predictions)
- ✅ **Metriken-Sammlung** (Performance, Qualität, Validierung)

## Stichworte

**Initialisierung**: Config, Metriken, Feature Flags

**Eingabe**: Predictions, Kalibrierung, Parameter, **Validierung**

**Post-Processing**: Simplification, **Adaptives Snap-to-Grid**, Rechtwinkligkeit, Kontext-Korrektur, **Parallelisierung**

**Normalisierung**: Pixel-zu-mm, Klassifikation, Validierung

**Rekonstruktion**: Wandachsen (**Optimiert**), Räume, Skeletonization, **Geometrie-Validierung**

**IFC-Export**: Wände, Türen, Fenster, Räume, Böden, Property Sets

**Validierung**: Parse-Test, Validation Report, TopView (**Nur bei Erfolg**), Template-Vergleich

**Metriken**: Performance, Qualität, Validierung, Quality Score

**Ausgabe**: IFC-Datei, Viewer (**Nur bei Erfolg**), Reports, GeoJSON, **Metriken**

## Qualitäts-Sicherung

### Validierungs-Punkte
1. **Eingabe-Validierung** (Schritt 1.5): Filtert defekte Predictions früh
2. **Geometrie-Validierung** (Schritt 5.3): Prüft Rekonstruktions-Qualität
3. **IFC-Parse-Validierung** (Schritt 7.1): Prüft IFC-Datei-Korrektheit

### Metriken & Monitoring
- **Quality Score**: 0.0-1.0 basierend auf mehreren Faktoren
- **Performance-Tracking**: Zeit pro Pipeline-Schritt
- **Warnungen/Fehler**: Kategorisiert und gezählt
- **Vorbereitung**: Für Prometheus/Grafana Integration

### Feature Flags
- Alle experimentellen Features können einzeln aktiviert/deaktiviert werden
- Ermöglicht A/B-Testing und schrittweise Aktivierung
- Standard: Alle Features aktiviert
