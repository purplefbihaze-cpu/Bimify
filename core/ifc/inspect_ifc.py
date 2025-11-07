from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import ifcopenshell


Coordinate = Tuple[float, float, float]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect an IFC file and report object counts and bounding extents.",
    )
    parser.add_argument("ifc_path", type=Path, help="Path to the IFC file to inspect")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Return the summary as formatted JSON instead of human-readable text.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.ifc_path.exists():
        parser.error(f"File not found: {args.ifc_path}")

    summary = inspect_ifc(args.ifc_path)

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        pretty_print_summary(summary)

    return 0


def inspect_ifc(path: Path) -> dict:
    model = ifcopenshell.open(str(path))
    products_with_geometry = [
        product
        for product in model.by_type("IfcProduct")
        if getattr(product, "Representation", None)
    ]

    counts = {
        "IfcProductWithRepresentation": len(products_with_geometry),
        "IfcWallStandardCase": len(model.by_type("IfcWallStandardCase")),
        "IfcDoor": len(model.by_type("IfcDoor")),
        "IfcWindow": len(model.by_type("IfcWindow")),
        "IfcSpace": len(model.by_type("IfcSpace")),
    }

    bounds = compute_bounds_from_products(products_with_geometry)

    return {
        "path": str(Path(path).resolve()),
        "schema": model.schema,
        "counts": counts,
        "bounds": bounds,
        "has_geometry": bounds is not None,
    }


def compute_bounds_from_products(products: Sequence[Any]) -> dict | None:
    points: List[Coordinate] = []
    for product in products:
        points.extend(extract_cartesian_points(product))

    if not points:
        return None

    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    min_z = min(point[2] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)
    max_z = max(point[2] for point in points)

    return {
        "min": {"x": round(min_x, 4), "y": round(min_y, 4), "z": round(min_z, 4)},
        "max": {"x": round(max_x, 4), "y": round(max_y, 4), "z": round(max_z, 4)},
        "size": {
            "x": round(max_x - min_x, 4),
            "y": round(max_y - min_y, 4),
            "z": round(max_z - min_z, 4),
        },
    }


def extract_cartesian_points(entity: Any) -> List[Coordinate]:
    collected: List[Coordinate] = []
    visited: set[int] = set()

    def visit(value: object) -> None:
        if value is None:
            return
        if isinstance(value, (int, float)):
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
            return
        if hasattr(value, "is_a"):
            entity_id = id(value)
            if entity_id in visited:
                return
            visited.add(entity_id)
            instance = value
            if instance.is_a("IfcCartesianPoint"):
                coords = getattr(instance, "Coordinates", None)
                if coords:
                    collected.append(normalize_point(coords))
                return
            try:
                info = instance.get_info()
            except Exception:
                return
            for key, nested in info.items():
                if key in {"type", "id"}:
                    continue
                visit(nested)

    visit(entity)
    return collected


def normalize_point(coords: Iterable[float]) -> Coordinate:
    values = list(coords)
    x = float(values[0]) if len(values) > 0 else 0.0
    y = float(values[1]) if len(values) > 1 else 0.0
    z = float(values[2]) if len(values) > 2 else 0.0
    return (x, y, z)


def pretty_print_summary(summary: dict) -> None:
    print(f"IFC file: {summary['path']}")
    print(f"Schema:   {summary['schema']}")
    counts = summary.get("counts", {})
    for key, value in counts.items():
        print(f"  {key}: {value}")
    bounds = summary.get("bounds")
    if bounds:
        print("Bounds (min -> max):")
        min_bounds = bounds["min"]
        max_bounds = bounds["max"]
        size = bounds["size"]
        print(f"  min:  x={min_bounds['x']}, y={min_bounds['y']}, z={min_bounds['z']}")
        print(f"  max:  x={max_bounds['x']}, y={max_bounds['y']}, z={max_bounds['z']}")
        print(f"  size: x={size['x']}, y={size['y']}, z={size['z']}")
    else:
        print("Bounds:  (no cartesian points found in product representations)")


if __name__ == "__main__":
    sys.exit(main())

