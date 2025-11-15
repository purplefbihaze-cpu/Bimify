from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import ifcopenshell  # type: ignore
    from ifcopenshell.entity_instance import entity_instance  # type: ignore
except ImportError as exc:  # pragma: no cover - defensive guard
    raise RuntimeError(
        "ifcopenshell is required to use core.ifc.template_analyzer"
    ) from exc


@dataclass
class OwnerHistorySummary:
    application: Dict[str, Any]
    owning_user: Dict[str, Any]
    creation_step: str
    timestamp: int


@dataclass
class ContextSummary:
    identifier: Optional[str]
    type: Optional[str]
    target_view: Optional[str]
    precision: Optional[float]
    world_coordinate_system: Dict[str, Any]
    true_north: Optional[Dict[str, Any]]


@dataclass
class UnitSummary:
    type: str
    name: str
    prefix: Optional[str]
    unit_type: Optional[str]


@dataclass
class PropertySetSummary:
    name: str
    property_count: int
    has_single_values: bool
    has_quantities: bool


@dataclass
class TemplateAnalysis:
    schema: str
    header: Dict[str, Any]
    owner_history: List[OwnerHistorySummary]
    contexts: List[ContextSummary]
    subcontexts: List[ContextSummary]
    units: List[UnitSummary]
    element_counts: Dict[str, int]
    property_sets: List[PropertySetSummary]


def _safe_getattr(entity: entity_instance, attr: str, default: Any = None) -> Any:
    try:
        return getattr(entity, attr, default)
    except Exception:
        return default


def _format_cartesian_point(entity: entity_instance | None) -> Dict[str, Any]:
    if entity is None:
        return {}
    coords = _safe_getattr(entity, "Coordinates")
    if isinstance(coords, Iterable):
        coords_list = [float(c) for c in coords]
    else:
        coords_list = []
    return {"type": entity.is_a() if entity else None, "coordinates": coords_list}


def _format_direction(entity: entity_instance | None) -> Dict[str, Any]:
    if entity is None:
        return {}
    ratios = _safe_getattr(entity, "DirectionRatios")
    if isinstance(ratios, Iterable):
        ratios_list = [float(r) for r in ratios]
    else:
        ratios_list = []
    return {"type": entity.is_a() if entity else None, "ratios": ratios_list}


def _summarize_owner_history(model: ifcopenshell.file) -> List[OwnerHistorySummary]:
    summaries: List[OwnerHistorySummary] = []
    for history in model.by_type("IfcOwnerHistory"):
        application = _safe_getattr(history, "OwningApplication")
        owning_user = _safe_getattr(history, "OwningUser")
        summary = OwnerHistorySummary(
            application={
                "identifier": _safe_getattr(application, "ApplicationIdentifier"),
                "name": _safe_getattr(application, "ApplicationFullName"),
                "version": _safe_getattr(application, "Version"),
                "organization": _safe_getattr(
                    _safe_getattr(application, "ApplicationDeveloper"), "Name"
                ),
            }
            if application
            else {},
            owning_user={
                "person": _safe_getattr(
                    _safe_getattr(owning_user, "ThePerson"), "FamilyName"
                ),
                "organization": _safe_getattr(
                    _safe_getattr(owning_user, "TheOrganization"), "Name"
                ),
            }
            if owning_user
            else {},
            creation_step=str(_safe_getattr(history, "ChangeAction")),
            timestamp=int(_safe_getattr(history, "CreationDate", 0) or 0),
        )
        summaries.append(summary)
    return summaries


def _summarize_context(entity: entity_instance) -> ContextSummary:
    placement = _safe_getattr(entity, "WorldCoordinateSystem")
    return ContextSummary(
        identifier=_safe_getattr(entity, "ContextIdentifier"),
        type=_safe_getattr(entity, "ContextType"),
        target_view=_safe_getattr(entity, "TargetView"),
        precision=_safe_getattr(entity, "Precision"),
        world_coordinate_system={
            "location": _format_cartesian_point(_safe_getattr(placement, "Location")),
            "axis": _format_direction(_safe_getattr(placement, "Axis")),
            "ref_direction": _format_direction(
                _safe_getattr(placement, "RefDirection")
            ),
        },
        true_north=_format_direction(_safe_getattr(entity, "TrueNorth")),
    )


def _summarize_units(model: ifcopenshell.file) -> List[UnitSummary]:
    unit_assignment = model.get_unit_assignment()
    if unit_assignment is None:
        return []
    summaries: List[UnitSummary] = []
    for unit in getattr(unit_assignment, "Units", []) or []:
        if hasattr(unit, "UnitType"):
            unit_type = str(_safe_getattr(unit, "UnitType"))
        else:
            unit_type = None
        if hasattr(unit, "Name"):
            name = str(_safe_getattr(unit, "Name"))
        elif hasattr(unit, "UnitName"):
            name = str(_safe_getattr(unit, "UnitName"))
        else:
            name = unit.is_a()
        summaries.append(
            UnitSummary(
                type=unit.is_a(),
                name=name,
                prefix=str(_safe_getattr(unit, "Prefix")) if hasattr(unit, "Prefix") else None,
                unit_type=unit_type,
            )
        )
    return summaries


def _count_elements(model: ifcopenshell.file) -> Dict[str, int]:
    element_types = [
        "IfcWall", "IfcWallStandardCase",
        "IfcDoor", "IfcWindow",
        "IfcSpace", "IfcSlab",
        "IfcColumn", "IfcBeam",
    ]
    counts: Counter[str] = Counter()
    for element_type in element_types:
        counts[element_type] = len(model.by_type(element_type))
    return dict(counts)


def _summarize_property_sets(model: ifcopenshell.file) -> List[PropertySetSummary]:
    summaries: List[PropertySetSummary] = []
    for pset in model.by_type("IfcPropertySet"):
        properties = list(getattr(pset, "HasProperties", []) or [])
        has_single_values = any(getattr(prop, "is_a", lambda: "")() == "IfcPropertySingleValue" for prop in properties)
        has_quantities = any(getattr(prop, "is_a", lambda: "")().startswith("IfcQuantity") for prop in properties)
        summaries.append(
            PropertySetSummary(
                name=str(_safe_getattr(pset, "Name")),
                property_count=len(properties),
                has_single_values=has_single_values,
                has_quantities=has_quantities,
            )
        )
    return summaries


def analyze_ifc_template(ifc_path: Path) -> TemplateAnalysis:
    if not ifc_path.exists():
        raise FileNotFoundError(ifc_path)

    model = ifcopenshell.open(ifc_path.as_posix())

    header = {
        "file_description": list(model.header.file_description.description),
        "implementation_level": model.header.file_description.implementation_level,
        "file_name": {
            "name": model.header.file_name.name,
            "time_stamp": model.header.file_name.time_stamp,
            "author": list(model.header.file_name.author),
            "organization": list(model.header.file_name.organization),
            "preprocessor_version": model.header.file_name.preprocessor_version,
            "originating_system": model.header.file_name.originating_system,
            "authorization": model.header.file_name.authorization,
        },
        "file_schema": list(model.header.file_schema.schema_identifiers),
    }

    contexts = [_summarize_context(ctx) for ctx in model.by_type("IfcGeometricRepresentationContext")]
    subcontexts = [_summarize_context(ctx) for ctx in model.by_type("IfcGeometricRepresentationSubContext")]

    analysis = TemplateAnalysis(
        schema=str(model.schema),
        header=header,
        owner_history=_summarize_owner_history(model),
        contexts=contexts,
        subcontexts=subcontexts,
        units=_summarize_units(model),
        element_counts=_count_elements(model),
        property_sets=_summarize_property_sets(model),
    )
    return analysis


def _default_serializer(obj: Any) -> Any:  # pragma: no cover - fallback for JSON dumps
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def save_analysis(analysis: TemplateAnalysis, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(analysis), fh, indent=2, default=_default_serializer)


def compare_analysis(reference: TemplateAnalysis, candidate: TemplateAnalysis) -> Dict[str, Any]:
    ref_psets = {ps.name for ps in reference.property_sets}
    cand_psets = {ps.name for ps in candidate.property_sets}

    element_keys = set(reference.element_counts) | set(candidate.element_counts)
    element_diff = {
        key: candidate.element_counts.get(key, 0) - reference.element_counts.get(key, 0)
        for key in sorted(element_keys)
    }

    return {
        "schema_match": reference.schema == candidate.schema,
        "application_match": reference.owner_history[0].application if reference.owner_history else None,
        "candidate_application": candidate.owner_history[0].application if candidate.owner_history else None,
        "element_count_diff": element_diff,
        "missing_property_sets": sorted(ref_psets - cand_psets),
        "extra_property_sets": sorted(cand_psets - ref_psets),
    }


def compare_ifc_to_template(template_path: Path, candidate_path: Path) -> Dict[str, Any]:
    template_analysis = analyze_ifc_template(template_path)
    candidate_analysis = analyze_ifc_template(candidate_path)
    return compare_analysis(template_analysis, candidate_analysis)


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze IFC template file and emit summary JSON")
    parser.add_argument("ifc", type=Path, help="Path to IFC template file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write JSON analysis. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="Optional reference IFC file to compare against the analyzed file.",
    )
    return parser.parse_args(args=args)


def main(argv: Optional[List[str]] = None) -> None:
    namespace = parse_arguments(argv)
    if namespace.compare:
        comparison = compare_ifc_to_template(namespace.compare, namespace.ifc)
        if namespace.output:
            namespace.output.parent.mkdir(parents=True, exist_ok=True)
            namespace.output.write_text(json.dumps(comparison, indent=2, default=_default_serializer), encoding="utf-8")
        else:
            print(json.dumps(comparison, indent=2, default=_default_serializer))
    else:
        analysis = analyze_ifc_template(namespace.ifc)

        if namespace.output:
            save_analysis(analysis, namespace.output)
        else:
            print(json.dumps(asdict(analysis), indent=2, default=_default_serializer))


if __name__ == "__main__":  # pragma: no cover
    main()
