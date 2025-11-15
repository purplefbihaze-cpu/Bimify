"""
Comprehensive test for IFC V2 Export Pipeline.

Tests the complete pipeline from Roboflow predictions to IFC file generation.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ifcopenshell")
import ifcopenshell
from pathlib import Path

from services.api.ifc_exporter_v2 import run_ifc_export_v2
from services.api.schemas import ExportIFCV2Request, GeometryFidelityLevel
from core.ml.roboflow_client import RoboflowPrediction
from tests.test_complete_ifc_generation import create_sample_roboflow_predictions


def convert_rfpred_to_roboflow_prediction(rfpred) -> RoboflowPrediction:
    """Convert RFPred to RoboflowPrediction for V2 export."""
    points = [(float(x), float(y)) for x, y in (rfpred.polygon or [])] if rfpred.polygon else []
    return RoboflowPrediction(
        klass=rfpred.klass,
        confidence=rfpred.confidence,
        points=points,
        x=rfpred.bbox[0] + rfpred.bbox[2] / 2.0 if rfpred.bbox else None,
        y=rfpred.bbox[1] + rfpred.bbox[3] / 2.0 if rfpred.bbox else None,
        width=rfpred.bbox[2] if rfpred.bbox else None,
        height=rfpred.bbox[3] if rfpred.bbox else None,
    )


@pytest.mark.asyncio
async def test_ifc_v2_export_complete(tmp_path: Path) -> None:
    """Test complete IFC V2 export from Roboflow predictions."""
    # Create sample predictions
    rf_preds = create_sample_roboflow_predictions()
    
    # Convert to RoboflowPrediction format for V2 export
    roboflow_predictions = [convert_rfpred_to_roboflow_prediction(rfpred) for rfpred in rf_preds]
    
    # Create export request
    export_request = ExportIFCV2Request(
        predictions=roboflow_predictions,
        storey_height_mm=3000.0,
        door_height_mm=2100.0,
        window_height_mm=1000.0,
        window_head_elevation_mm=2000.0,
        floor_thickness_mm=200.0,
        px_per_mm=1.0,
        project_name="Test Project V2",
        storey_name="EG",
        geometry_fidelity=GeometryFidelityLevel.HIGH,
        preserve_exact_geometry=True,
    )
    
    # Run export
    export_dir = tmp_path / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    result = await run_ifc_export_v2(export_request, export_root=export_dir)
    
    # Validate response
    assert result.ifc_url is not None, "IFC URL should be present"
    assert result.file_name is not None, "File name should be present"
    
    # Check IFC file exists
    ifc_path = export_dir / result.file_name
    assert ifc_path.exists(), f"IFC file should be created at {ifc_path}"
    
    # Validate IFC file can be opened
    model = ifcopenshell.open(str(ifc_path))
    assert model is not None, "IFC file should be parseable"
    
    # Check for required entities
    project = model.by_type("IfcProject")
    assert len(project) > 0, "Should have IfcProject"
    
    site = model.by_type("IfcSite")
    assert len(site) > 0, "Should have IfcSite"
    
    building = model.by_type("IfcBuilding")
    assert len(building) > 0, "Should have IfcBuilding"
    
    storey = model.by_type("IfcBuildingStorey")
    assert len(storey) > 0, "Should have IfcBuildingStorey"
    
    # Check for building elements
    walls = model.by_type("IfcWallStandardCase")
    assert len(walls) > 0, f"Should have walls, found {len(walls)}"
    
    doors = model.by_type("IfcDoor")
    assert len(doors) > 0, f"Should have doors, found {len(doors)}"
    
    windows = model.by_type("IfcWindow")
    assert len(windows) > 0, f"Should have windows, found {len(windows)}"
    
    spaces = model.by_type("IfcSpace")
    assert len(spaces) > 0, f"Should have spaces, found {len(spaces)}"
    
    # Check for slabs/floors
    slabs = model.by_type("IfcSlab")
    assert len(slabs) > 0, f"Should have slabs, found {len(slabs)}"
    
    # Check opening relationships
    void_rels = model.by_type("IfcRelVoidsElement")
    assert len(void_rels) > 0, f"Should have void relationships, found {len(void_rels)}"
    
    # Check that walls have representations
    for wall in walls:
        assert wall.Representation is not None, f"Wall {wall.Name} should have representation"
    
    # Check that doors have representations
    for door in doors:
        assert door.Representation is not None, f"Door {door.Name} should have representation"
    
    # Check that windows have representations
    for window in windows:
        assert window.Representation is not None, f"Window {window.Name} should have representation"
    
    # Check that spaces have representations
    for space in spaces:
        assert space.Representation is not None, f"Space {space.Name} should have representation"
    
    model.close()
    
    print(f"✓ IFC V2 Export successful: {result.file_name}")
    print(f"  - Walls: {len(walls)}")
    print(f"  - Doors: {len(doors)}")
    print(f"  - Windows: {len(windows)}")
    print(f"  - Spaces: {len(spaces)}")
    print(f"  - Slabs: {len(slabs)}")

