"""API endpoint for merging 3 RoboFloor models."""

from pathlib import Path
from typing import Dict, Any
import json
import tempfile
import logging

from core.merge.merger import merge_models
from core.merge.validators import validate_plan, generate_qa_report
from core.merge.viewer import generate_viewer_html

logger = logging.getLogger(__name__)


def run_merge(
    model1_json: Dict[str, Any],
    model2_json: Dict[str, Any],
    model3_json: Dict[str, Any],
    px_to_meter: float = 0.001,
    snap_tolerance_px: float = 5.0,
    output_dir: Path | None = None,
) -> Dict[str, Any]:
    """Merge three model JSONs and return merged plan.
    
    Args:
        model1_json: Model 1 JSON (geometry/topology)
        model2_json: Model 2 JSON (rooms)
        model3_json: Model 3 JSON (clean geometry)
        px_to_meter: Pixel to meter scale
        snap_tolerance_px: Snap tolerance in pixels
        output_dir: Optional output directory for files
    
    Returns:
        Dict with merged plan JSON, QA report, and file paths
    """
    # Create temporary files for the models
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        model1_path = tmp_path / "model1.json"
        model2_path = tmp_path / "model2.json"
        model3_path = tmp_path / "model3.json"
        
        model1_path.write_text(json.dumps(model1_json), encoding='utf-8')
        model2_path.write_text(json.dumps(model2_json), encoding='utf-8')
        model3_path.write_text(json.dumps(model3_json), encoding='utf-8')
        
        # Merge models
        plan = merge_models(
            model1_path,
            model2_path,
            model3_path,
            px_to_meter=px_to_meter,
            snap_tolerance_px=snap_tolerance_px,
        )
        
        # Validate
        validation = validate_plan(plan)
        qa_report = generate_qa_report(plan, validation)
        
        # Generate viewer HTML if output_dir provided
        viewer_path = None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            viewer_path = output_dir / "merged_plan.preview.html"
            generate_viewer_html(plan, viewer_path)
        
        return {
            "plan": plan.model_dump(),
            "qa_report": qa_report,
            "viewer_path": str(viewer_path) if viewer_path else None,
        }

