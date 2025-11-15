# Bimify – Roboflow-assisted Floorplan to IFC 4.3 Pipeline

This repository hosts the core services and libraries for converting 2D floorplans (PDF/PNG) into validated IFC 4.3 models with full BIM compliance.

## Project Structure

```
Bimify/
├── services/              # Backend services
│   ├── api/              # FastAPI service for uploads, job management and exports
│   └── worker/           # Celery worker for Roboflow inference, post-processing, IFC export
├── core/                 # Shared domain logic
│   ├── ifc/              # IFC model generation with full BIM compliance
│   ├── reconstruct/      # Geometry reconstruction (walls, openings, spaces)
│   ├── validate/         # Validation and quality checks
│   ├── ml/               # ML integration (Roboflow client, post-processing)
│   ├── preprocess/       # Image preprocessing and raster pipeline
│   └── vector/           # Vector geometry and IFC topview generation
├── config/               # Static configuration templates
├── tests/                # Unit and integration tests
├── tools/                # Additional tools
│   ├── XbimPreprocess/   # C# tool for geometry optimization (optional)
│   └── publish-xbim.ps1  # PowerShell script to build XbimPreprocess
├── ui/                   # Next.js frontend application
├── data/                 # Runtime data (uploads, exports, jobs, previews)
├── logs/                 # Application logs
├── docker/               # Docker configuration files
└── docs/                 # Documentation
```

**Note**: `Lib/` and `Scripts/` directories are Python virtual environment artifacts (Windows) and should be ignored. They are already in `.gitignore`.

## BIM Features

The system generates BIM-ready IFC files with:

- **Complete Element Classification**: All elements (walls, doors, windows, floors, ceilings) are properly classified with IfcWallType, IfcDoorType, IfcWindowType, and correct PredefinedType attributes. External/internal wall classification with heuristic fallback when detection is uncertain.
- **Wall Gap Closure**: Automatic gap closure between walls (50-500mm tolerance) with two-pass processing, T-junction detection, and post-processing repair of remaining gaps. Ensures no gaps > 50mm in final model.
- **Opening Connections**: Robust IfcRelVoidsElement and IfcRelFillsElement relationships with fallback logic to nearest wall, retry mechanisms, and post-processing to ensure all openings are properly connected to walls.
- **Material Assignment**: Complete material layer sets for all elements (walls, floors, ceilings, doors, windows) with proper thermal and acoustic properties, U-values, and fire ratings.
- **Geometry Quality**: Automatic repair of invalid polygons before IFC export, validation of space/wall geometries, and filtering of invalid/empty geometries.
- **Space Boundaries**: Proper IfcRelSpaceBoundary relationships for energy analysis and space management.
- **Property Sets**: Comprehensive PropertySets (Pset_WallCommon, Pset_SlabCommon, Pset_CoveringCommon) with thermal, acoustic, and fire rating properties.

## Getting Started (Development)

```bash
poetry install
poetry run uvicorn services.api.main:app --reload

# in a separate shell
poetry run celery -A services.worker.app worker --loglevel=INFO
```

Docker and CI workflows are provided in the repository for reproducible deployments.

## Futuristic Frontend Companion

The `ui/` directory hosts a cinematic Next.js 14 application that surfaces the pipeline with a glassmorphism aesthetic, micro-interactions, and adaptive theming.

### Highlights

- **Dynamic Themes & Accessibility** – Toggle warm/cool palettes, dark/light/system modes, and high-contrast from the in-app palette switcher. Motion curves respect `prefers-reduced-motion`, and optional haptic cues are gated behind both device support and user preference.
- **Multi-page Artifact Explorer** – Upload multi-page PDFs and browse each page via thumbnail carousel. The 2D canvas viewer filters walls/axes per selected page, while the 3D viewer (Three.js + web-ifc) streams IFC previews.
- **Live Metrics** – Processing timeline metrics are charted with Recharts, giving quick insight into inference/post-processing durations.
- **S3 Mirroring** – When AWS credentials are present, artifacts, thumbnails, and IFC exports are mirrored to S3 and exposed via presigned URLs for lightweight downloads.

### Running the UI

```bash
cd ui
npm install
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
```

### AWS Integration

To activate S3 mirroring and presigned URLs, provide credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) alongside `AWS_REGION` and make sure `storage.bucket` is set in the config. The worker uploads all generated artifacts to `s3://{bucket}/artifacts/{job_id}/...` while keeping local fallbacks for development.


