# HottCAD-Kompatibilitätscheck & Auto-Healing

Dieses Dokument beschreibt den neuen Validierungs- und Simulations-Workflow, der sicherstellt,
dass exportierte IFC4x3-Modelle den Import-Anforderungen von HottCAD entsprechen und gleichzeitig
potentielle Topologie-Fehler automatisch identifiziert.

## Überblick

- **Backend**: Neue Endpunkte `/v1/hottcad/validate` und `/v1/hottcad/simulate` analysieren IFC-Dateien mit
  `ifcopenshell` und erzeugen Health Score, Checklisten sowie Auto-Healing-Vorschläge.
- **Frontend**: Der bestehende WexBIM-Viewer besitzt nun ein HottCAD-Panel mit Statuslampen, Score-Anzeige,
  Simulationstriggern und Highlight-Steuerung im 3D-Modell.
- **Simulation**: Liefert hypothetische `IfcRelConnectsElements`, `IfcRelSpaceBoundary`-Kandidaten und
  Materiallayer-Empfehlungen einschließlich GUID- und Product-ID-Referenzen für gezieltes Highlighting.

## HottCAD-Validierung (Backend)

### Request Schema

```json
POST /v1/hottcad/validate
{
  "ifc_url": "/absolute/or/relative/path.ifc" | null,
  "job_id": "optional-job-uuid" | null,
  "tolerance_mm": 0.5
}
```

- `ifc_url`: Beliebiger Pfad (absolut, relativ oder `/files/xyz.ifc`).
- `job_id`: Alternative Quelle; es wird automatisch `primary`/`improved` aus den Job-Metadaten geladen.
- `tolerance_mm`: Kontakt-/Gap-Toleranz für Wandanschlüsse.

### Response (Auszug)

- `score`: Health Score 0–100 (Gewichtung: Geometrie 30, Topologie 30, Relationen 30, Format 10).
- `checks`: Liste mit Status (`ok|warn|fail`), Details und betroffenen GUIDs.
- `metrics`: Sammelwerte (Wände, Öffnungen, Space Boundaries, MaterialLayerSetUsage usw.).
- `file_info`: Pfad, Größe, Schema, Plain-IFC-Indikator.
- `highlightSets`: Gruppierte GUID/Product-ID-Listen für fehlerhafte Elemente (Validierung & Simulation).

## Auto-Healing-Simulation

`POST /v1/hottcad/simulate` verwendet dieselbe Payload wie `validate` und liefert zusätzliche Felder:

- `proposed.connects`: Wandpaare innerhalb der Toleranz mit Kontakt-Typ (`touch|gap`).
- `proposed.spaceBoundaries`: Ableitbare Raumgrenzen basierend auf Wand-Raum-Zuordnungen.
- `proposed.materials`: Material-Layer-Empfehlungen inklusive ermittelter Wandstärken.
- `highlightSets`: GUID- und Product-ID-Listen für den Viewer (`walls-all`, `walls-gaps`, …).
- `completeness`: Kurzindikatoren (Anzahl Gaps, Wände, Räume, `roomsClosed`).

Intern werden GUIDs → Product IDs per `ifcopenshell`-Lookup gemappt, damit der Viewer gezielt
`State.HIGHLIGHTED` setzen kann.

## UI-Integration (Next.js)

- `ui/pages/wexbim-viewer.tsx` bietet ein linkes Control-Panel mit:
  - Toleranz-Schalter (die xBIM-optimierte IFC wird automatisch verwendet)
  - Health Score-Anzeige & Checkliste der Backend-Ergebnisse
  - Simulationstrigger und Highlight-Liste
- Highlights steuern den `@xbim/viewer` direkt (Reset + `State.HIGHLIGHTED`) anhand übermittelter Product-IDs.
- Query-Parameter:
  - `url`: WexBIM-Datei
  - `ifc`: IFC-Datei (optional, bevorzugt verbesserte IFC)
  - `jobId`: Alternativer Job-Lookup
- `ui/components/WexbimViewer` generiert die URL inkl. optionaler IFC- und Job-Parameter.

## Tests

### Backend (pytest)

```bash
pytest tests/test_hottcad.py tests/test_hottcad_api.py
```

- `test_hottcad.py`: Erstellt synthetische IFC4x3-Dateien (2 Wände + Space) und prüft Validator-/Simulator-Rückgaben.
- `test_hottcad_api.py`: Integrationstest der neuen FastAPI-Endpunkte mit lokalem Sample-IFC.

### Frontend (Vitest)

```bash
cd ui
npm install
npm test
```

- `ui/tests/wexbim-viewer.test.tsx`: Verifiziert die Übergabe von `url`, `ifc` und `jobId` Query-Parametern an das iFrame.

## Bedienhinweise (UI)

1. WexBIM-Viewer mit `url`, optional `ifc`/`jobId` aufrufen (z. B. `/wexbim-viewer?url=/files/model.wexbim&ifc=/files/model_preproc.ifc`).
2. Im Panel Toleranz einstellen und „Validierung starten“ ausführen.
3. Health Score & Checkliste prüfen – rote Items vor Export beheben.
4. „Simulation ausführen“ liefert potenzielle Reconnects/Gaps – Highlights markieren betroffene Wände.
5. Bei grünen Häkchen in kritischen Checks (Geometrie, Relationen, Topologie) gilt das Modell als HottCAD-konform.

## Umsetzungshinweise

- Fehlende `IfcRelConnectsElements`, `IfcRelSpaceBoundary` oder `IfcMaterialLayerSetUsage` werden markiert.
- Materialschichtendaten können aus `Bimify_WallParams.WidthMm` übernommen werden, um `IfcMaterialLayerSetUsage` zu erzeugen.
- Der Workflow ist tolerant gegenüber relativen und absoluten Pfaden; HTTP-URLs werden temporär heruntergeladen.
- Für Jobs werden relative Pfade (`ifc.primary`, `ifc.improved`, `ifc.wexbim`) automatisch in `data/jobs/<id>/...` aufgelöst.

Mit diesen Bausteinen lässt sich der Export um eine automatisierte HottCAD-Qualitätssicherung inklusive visueller
Rückmeldung erweitern. Weitere Schritte (z. B. persistente Auto-Reconnects) können auf Basis der Simulationsergebnisse
implementiert werden.


