# Geometrie-Vorverarbeitung mit xBIM

Dieses Dokument beschreibt den verpflichtenden Zwischenschritt zur Verbesserung von IFC-Geometrien mithilfe des xBIM-Toolkits. Alle Exporte und HottCAD-Checks basieren auf der xBIM-optimierten IFC.

## Überblick

1. Der Python-Worker erzeugt wie bisher ein IFC (`model.ifc`).
2. Anschließend wird zwingend das .NET-Tool `XbimPreprocess` aufgerufen.
3. Das Tool berechnet den xBIM-3D-Kontext (`CreateContext`) für alle `IfcProduct`-Elemente, schreibt eine zweite IFC (`model_preproc.ifc`) und hält alle Originalgeometrien unverändert.
4. Zusätzlich werden ein WexBIM und optional JSON-Statistiken erzeugt.
5. UI und API verwenden das WexBIM beziehungsweise die daraus abgeleiteten Daten – das optimierte IFC dient als konsistenter Zwilling.

## Build & Bereitstellung des Tools

Das Projekt liegt unter `tools/XbimPreprocess` (Target: .NET 6).

> Hinweis: Single-File-Publish ist mit der xBIM Geometry Engine nicht kompatibel, da deren Assembly-Resolver `Assembly.CodeBase` nutzt. Das Tool **muss** als Multi-File-Paket ausgeliefert werden.

```bash
# Beispiel: Windows x64, self-contained
dotnet publish tools/XbimPreprocess/XbimPreprocess.csproj \
  -c Release \
  -r win-x64 \
  --self-contained true \
  /p:PublishSingleFile=false \
  /p:PublishTrimmed=false \
  -o tools/XbimPreprocess/publish/win-x64

# Ergebnis:
# tools/XbimPreprocess/publish/win-x64/XbimPreprocess.exe
# plus Xbim.Geometry.Engine64.dll etc. im gleichen Ordner
```

Für Linux analog (z. B. `-r linux-x64`).

Die ausgelieferte Binärdatei sollte in einen für den Worker erreichbaren Pfad kopiert werden (z. B. `bin/XbimPreprocess/XbimPreprocess`).

## Konfiguration

Globale Voreinstellungen befinden sich in `config/default.yaml`:

```yaml
geometry:
  preprocess:
    enabled: true
    command: null      # Pfad oder Array; null => via ENV konfigurieren
    args: []
    env: {}
    emit_stats: true
    emit_wexbim: true
    capture_logs: true
    timeout_seconds: 600
```

Aktivierung & Pfad können wahlweise über YAML oder Umgebungsvariablen erfolgen:

| Einstellung | YAML-Key | ENV-Fallback |
|-------------|----------|--------------|
| Aktivieren  | `enabled` | – |
| Kommando    | `command` | `XBIM_PREPROCESS_COMMAND` |
| Zusatz-Args | `args`    | `XBIM_PREPROCESS_ARGS` |
| Timeout (s) | `timeout_seconds` | `XBIM_PREPROCESS_TIMEOUT` |
| Stats       | `emit_stats` | `XBIM_PREPROCESS_STATS` (`0/1`) |
| WexBIM      | `emit_wexbim` | `XBIM_PREPROCESS_WEXBIM` |

Wenn kein ausführbares Kommando gefunden wird, schlägt der Export mit Fehler ab. Eine erfolgreiche HottCAD-Validierung setzt die optimierte IFC voraus.

## Artefakte & Metadaten

Bei erfolgreicher Vorverarbeitung werden folgende Dateien neben `model.ifc` abgelegt:

| Datei | Beschreibung |
|-------|--------------|
| `model_preproc.ifc` | IFC-Kopie nach dem xBIM-Schritt (Geometrie identisch zum Input, dient als begleitende Referenz) |
| `model_preproc_stats.json` | Statistik (Produkte, Dreiecke, Dateigrößen etc.) |
| `model_preproc.wexbim` (optional) | Binäres xBIM-Format für externe Viewer |

Der Worker spiegelt diese Artefakte (sofern S3 konfiguriert ist) und erweitert `job.meta.ifc` um Pfade (`primary`, `improved`, `stats`, `wexbim`).

Die REST-API (`POST /v1/export-ifc`) liefert zusätzliche Felder:

- `improved_ifc_url`
- `improved_ifc_stats_url`
- `improved_wexbim_url`

Die Frontend-Seite zeigt einen Umschalter „Verbesserte Geometrie (xBIM)“ und stellt alle Dateien zum Download bereit.

## Statistiken

`model_preproc_stats.json` enthält u. a.:

- Anzahl verarbeiteter Produkte (`productsVisited`, `productsUpdated`)
- Aggregierte Dreiecks- und Vertex-Zahlen
- Dateigrößen vor/nach dem Schritt (`inputBytes`, `outputBytes`)
- Flags (z. B. `tessellationSkipped`, `wexbimRequested`, `wexbimWritten`)
- Ausführungszeitpunkt, Tool-/Stats-Version

Diese Informationen dienen der QA, um Geometrieänderungen nachvollziehen zu können.

## Fehlerfälle & Fallback

- Fehlt das Kommando, wird der Schritt übersprungen (Log + API-Warnung).
- Fehlschläge beim Tool werden protokolliert und als Warning an UI/API weitergegeben; das ursprüngliche IFC bleibt nutzbar.
- Timeout & Exitcode werden in `job.meta.ifc_preprocess` bzw. im API-Response-Warning sichtbar.

## Weiterentwicklung

- Erweiterung des Stats-Files um zusätzliche Kennzahlen (z. B. Produktliste) bei Bedarf.
- Integration weiterer Ausgabeformate (glTF, USD).
- Automatisierte Paketierung der xBIM-Binary über CI.


