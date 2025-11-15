"""CLI for merge pipeline."""

import argparse
import json
import logging
from pathlib import Path

from .merger import merge_models
from .viewer import generate_viewer_html
from .ifc_exporter import export_to_ifc
from .schema import CanonicalPlan
from .validators import validate_plan, generate_qa_report
from .conformance import check_ifc_conformance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Merge RoboFloor 3-model outputs to canonical JSON and IFC")
    parser.add_argument("--model1", type=Path, required=True, help="Model 1 JSON (geometry/topology)")
    parser.add_argument("--model2", type=Path, required=True, help="Model 2 JSON (rooms)")
    parser.add_argument("--model3", type=Path, required=True, help="Model 3 JSON (clean geometry)")
    parser.add_argument("--output-dir", type=Path, default=Path("examples/out"), help="Output directory")
    parser.add_argument("--px-to-meter", type=float, default=0.001, help="Pixel to meter scale (default: 0.001 = 1mm/px)")
    parser.add_argument("--snap-tolerance", type=float, default=5.0, help="Snap tolerance in pixels")
    parser.add_argument("--template-ifc", type=Path, help="Template IFC file to follow")
    parser.add_argument("--storey-height", type=float, default=3.0, help="Storey height in meters")
    parser.add_argument("--door-height", type=float, default=2.1, help="Door height in meters")
    parser.add_argument("--window-height", type=float, default=1.0, help="Window height in meters")
    parser.add_argument("--skip-viewer", action="store_true", help="Skip HTML viewer generation")
    parser.add_argument("--skip-ifc", action="store_true", help="Skip IFC export")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge models
    logger.info("Merging models...")
    plan = merge_models(
        args.model1,
        args.model2,
        args.model3,
        px_to_meter=args.px_to_meter,
        snap_tolerance_px=args.snap_tolerance,
    )
    
    # Validate plan
    logger.info("Validating merged plan...")
    validation = validate_plan(plan)
    qa_report = generate_qa_report(plan, validation)
    
    # Save QA report
    qa_path = args.output_dir / "merged_plan.qc.json"
    qa_path.write_text(json.dumps(qa_report, indent=2), encoding='utf-8')
    logger.info(f"Saved QA report to {qa_path}")
    
    if qa_report["summary"]["total_issues"] > 0:
        logger.warning(f"Found {qa_report['summary']['total_issues']} validation issues")
    
    # Save canonical JSON
    json_path = args.output_dir / "merged_plan.json"
    json_path.write_text(plan.model_dump_json(indent=2), encoding='utf-8')
    logger.info(f"Saved canonical JSON to {json_path}")
    
    # Generate viewer
    if not args.skip_viewer:
        viewer_path = args.output_dir / "merged_plan.preview.html"
        generate_viewer_html(plan, viewer_path)
        logger.info(f"Generated viewer at {viewer_path}")
    
    # Export IFC
    if not args.skip_ifc:
        ifc_path = args.output_dir / "merged_plan.ifc"
        export_to_ifc(
            plan,
            ifc_path,
            template_path=args.template_ifc,
            storey_height_m=args.storey_height,
            door_height_m=args.door_height,
            window_height_m=args.window_height,
        )
        logger.info(f"Exported IFC to {ifc_path}")
        
        # Check conformance if template provided
        if args.template_ifc and args.template_ifc.exists():
            logger.info("Checking IFC conformance against template...")
            conformance_report = check_ifc_conformance(ifc_path, args.template_ifc)
            conformance_path = args.output_dir / "ifc_conformance_report.json"
            conformance_path.write_text(json.dumps(conformance_report, indent=2), encoding='utf-8')
            logger.info(f"Saved conformance report to {conformance_path}")
            
            if conformance_report.get("summary", {}).get("total_missing", 0) > 0:
                logger.warning(f"Found {conformance_report['summary']['total_missing']} missing elements compared to template")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

