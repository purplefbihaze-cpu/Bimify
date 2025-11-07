# Bimify – Roboflow-assisted Floorplan to IFC 4.3 Pipeline

This repository hosts the core services and libraries for converting 2D floorplans (PDF/PNG) into validated IFC 4.3 models.

## Project Structure

- `services/api/` – FastAPI service for uploads, job management and exports
- `services/worker/` – Celery worker responsible for heavy lifting (Roboflow inference, post-processing, IFC export)
- `core/` – shared domain logic (ingestion, preprocessing, ML integration, reconstruction, IFC export)
- `config/` – static configuration templates
- `tests/` – unit and integration tests

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


