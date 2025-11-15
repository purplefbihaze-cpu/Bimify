"""HottCAD validation and simulation endpoints."""

from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, HTTPException

from core.ifc.hottcad_simulate import simulate_hottcad
from core.ifc.hottcad_validator import validate_hottcad
from services.api.utils import _resolve_ifc_source
from services.api.schemas import (
    HottCADCheckOut,
    HottCADCompletenessOut,
    HottCADConnectionOut,
    HottCADFileInfo,
    HottCADMaterialSuggestionOut,
    HottCADMetricsOut,
    HottCADSimulateRequest,
    HottCADSimulationProposedOut,
    HottCADSimulationResponse,
    HottCADValidateRequest,
    HottCADValidationResponse,
    HottCADHighlightOut,
    HottCADSpaceBoundaryOut,
)


router = APIRouter(prefix="/v1/hottcad", tags=["hottcad"])


@router.post("/validate", response_model=HottCADValidationResponse)
async def hottcad_validate(payload: HottCADValidateRequest) -> HottCADValidationResponse:
    """Validate IFC file against HottCAD requirements."""
    path, cleanup = _resolve_ifc_source(payload.ifc_url, payload.job_id)
    cleanup_path = path if cleanup else None
    try:
        try:
            result = validate_hottcad(path, tolerance_mm=payload.tolerance_mm)
        except HTTPException:
            raise
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"HottCAD-Validierung fehlgeschlagen: {exc}") from exc

        file_info = HottCADFileInfo(
            schema=result.file_info.get("schema") or result.schema,
            path=result.file_info.get("path"),
            sizeBytes=result.file_info.get("sizeBytes"),
            isPlainIFC=result.file_info.get("isPlainIFC"),
        )
        checks_out = [
            HottCADCheckOut(
                id=check.id,
                title=check.title,
                status=check.status,
                details=list(check.details),
                affected={key: list(values) for key, values in check.affected.items()},
            )
            for check in result.checks
        ]
        metrics_out = HottCADMetricsOut(**vars(result.metrics))
        highlight_out = [
            HottCADHighlightOut(
                id=hs.id,
                label=hs.label,
                guids=list(hs.guids),
                productIds=list(hs.product_ids),
            )
            for hs in result.highlight_sets
        ]

        return HottCADValidationResponse(
            schema=result.schema,
            file_info=file_info,
            checks=checks_out,
            metrics=metrics_out,
            score=result.score,
            highlightSets=highlight_out,
        )
    finally:
        if cleanup_path:
            try:
                cleanup_path.unlink(missing_ok=True)
            except Exception:
                pass


@router.post("/simulate", response_model=HottCADSimulationResponse)
async def hottcad_simulate(payload: HottCADSimulateRequest) -> HottCADSimulationResponse:
    """Simulate HottCAD improvements for IFC file."""
    path, cleanup = _resolve_ifc_source(payload.ifc_url, payload.job_id)
    cleanup_path = path if cleanup else None
    try:
        try:
            result = simulate_hottcad(path, tolerance_mm=payload.tolerance_mm)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"HottCAD-Simulation fehlgeschlagen: {exc}") from exc

        connects = [
            HottCADConnectionOut(
                walls=list(prop.walls),
                distanceMm=prop.distance_mm,
                contactType=prop.contact_type,
                notes=list(prop.notes),
            )
            for prop in result.proposed.get("connects", [])
        ]
        space_boundaries = [
            HottCADSpaceBoundaryOut(
                walls=list(entry.get("walls", [])),
                spaces=list(entry.get("spaces", [])),
                note=entry.get("note"),
            )
            for entry in result.proposed.get("spaceBoundaries", [])
        ]
        materials = [
            HottCADMaterialSuggestionOut(
                wall=item.wall,
                thicknessMm=item.thickness_mm,
                note=item.note,
            )
            for item in result.proposed.get("materials", [])
        ]

        proposed_out = HottCADSimulationProposedOut(
            connects=connects,
            spaceBoundaries=space_boundaries,
            materials=materials,
        )

        completeness_out = HottCADCompletenessOut(**result.completeness)
        highlight_out = [
            HottCADHighlightOut(
                id=hs.id,
                label=hs.label,
                guids=list(hs.guids),
                productIds=list(hs.product_ids),
            )
            for hs in result.highlight_sets
        ]

        return HottCADSimulationResponse(
            proposed=proposed_out,
            completeness=completeness_out,
            highlightSets=highlight_out,
        )
    finally:
        if cleanup_path:
            try:
                cleanup_path.unlink(missing_ok=True)
            except Exception:
                pass

