from __future__ import annotations

from pathlib import Path
from typing import List

import ifcopenshell
import ifcopenshell.api


def export_ifc43(out_path: Path) -> Path:
    # ifcopenshell.api.project.create_file uses `version` as parameter name
    model = ifcopenshell.api.run("project.create_file", version="IFC4")
    project = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcProject", name="Bimify Project")
    ifcopenshell.api.run(
        "unit.assign_unit",
        model,
        length={
            "is_metric": True,
            "raw": "MILLIMETERS",
        },
    )
    context = ifcopenshell.api.run("context.add_context", model, context_type="Model")
    body = ifcopenshell.api.run("context.add_context", model, context_type="Model", context_identifier="Body", target_view="MODEL_VIEW", parent=context)
    site = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSite", name="Site")
    building = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuilding", name="Building")
    storey = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuildingStorey", name="EG")
    try:
        storey.Elevation = 0.0
    except Exception:
        pass
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=project, products=[site])
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=site, products=[building])
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=building, products=[storey])
    # Placeholder: no elements yet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(out_path))
    return out_path


