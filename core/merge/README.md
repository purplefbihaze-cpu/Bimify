# RoboFloor 3-Model Merge Pipeline

This module merges outputs from three RoboFloor models into a canonical JSON format and exports to IFC4.

## Overview

- **Model 1**: Geometry/topology with reliable snapping and L-corners (but jagged lines)
- **Model 2**: Room segmentation
- **Model 3**: Clean wall geometry + best doors/windows (but poor snapping)

The pipeline combines the strengths of all three models:
- Uses Model 1's topology and snapping
- Adopts Model 3's clean geometry
- Integrates Model 2's room definitions

## Usage

```bash
python -m core.merge.cli \
  --model1 "examples/model 1 json.txt" \
  --model2 "examples/model 2 json.txt" \
  --model3 "examples/model 3 json.txt" \
  --output-dir examples/out \
  --template-ifc "examples/example ifc.ifc" \
  --px-to-meter 0.001
```

## Outputs

- `merged_plan.json` - Canonical merged JSON
- `merged_plan.preview.html` - Interactive viewer
- `merged_plan.qc.json` - Quality validation report
- `merged_plan.ifc` - IFC4 export (if enabled)
- `ifc_conformance_report.json` - Conformance check against template (if template provided)

## Components

- `schema.py` - Canonical JSON schema (openings include `sillHeight`, `headHeight`, `overallHeight`, `depth` for IFC export)
- `parsers.py` - Parsers for all 3 model formats
- `wall_graph.py` - Wall centerline graph with snapping
- `wall_cleaner.py` - Clean wall line extraction from Model 3
- `merger.py` - Main merging logic
- `alignment.py` - Co-registration checking
- `validators.py` - Quality validators
- `viewer.py` - HTML viewer generator
- `ifc_exporter.py` - IFC4 export
- `template_ingest.py` - Template IFC profile extraction
- `conformance.py` - IFC conformance checker
- `cli.py` - Command-line interface

## Configuration

Default parameters:
- `px-to-meter`: 0.001 (1mm per pixel)
- `snap-tolerance`: 5.0 pixels
- `storey-height`: 3.0 meters
- `door-height`: 2.1 meters
- `window-height`: 1.0 meters

