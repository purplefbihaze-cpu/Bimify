import json
from pathlib import Path

import pytest

from core.merge.merger import merge_models
from core.merge.schema import Opening


@pytest.fixture
def sample_model_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create minimal model JSON files emulating Model1/Model2/Model3 outputs."""
    model1_data = {
        "image": {"width": 1000, "height": 1000},
        "predictions": [
            {
                "class": "internal-walls",
                "detection_id": "wall-1",
                "confidence": 0.95,
                "points": [
                    {"x": 100, "y": 100},
                    {"x": 900, "y": 100},
                ],
            }
        ],
    }
    model2_data = {
        "image": {"width": 1000, "height": 1000},
        "predictions": [],
    }
    model3_data = {
        "image": {"width": 1000, "height": 1000},
        "predictions": [
            {
                "class": "window",
                "detection_id": "open-1",
                "confidence": 0.9,
                "x": 300,
                "y": 80,
                "width": 100,
                "height": 120,
            }
        ],
    }

    m1 = tmp_path / "model1.json"
    m2 = tmp_path / "model2.json"
    m3 = tmp_path / "model3.json"

    m1.write_text(json.dumps(model1_data))
    m2.write_text(json.dumps(model2_data))
    m3.write_text(json.dumps(model3_data))

    return m1, m2, m3


def test_merge_opening_vertical_metadata(sample_model_files: tuple[Path, Path, Path]) -> None:
    model1, model2, model3 = sample_model_files

    plan = merge_models(model1, model2, model3, px_to_meter=0.001)

    assert plan.openings, "Merged plan should contain at least one opening"

    opening: Opening = plan.openings[0]
    assert opening.sillHeight is not None and opening.sillHeight >= 0.0
    assert opening.headHeight is not None and opening.headHeight > opening.sillHeight
    assert opening.overallHeight is not None and opening.overallHeight > 0.0
    assert opening.depth is not None and opening.depth > 0.0
