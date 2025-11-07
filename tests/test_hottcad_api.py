from __future__ import annotations

import pytest

pytest.importorskip("ifcopenshell")
pytest.importorskip("shapely")
pytest.importorskip("loguru")

from httpx import AsyncClient

from services.api.main import app
from tests.utils_ifc import build_sample_ifc


@pytest.mark.asyncio()
async def test_hottcad_validate_endpoint(tmp_path):
    path = tmp_path / "sample.ifc"
    build_sample_ifc(path)

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/v1/hottcad/validate",
            json={"ifc_url": str(path)},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"]["wall_count"] == 2
    assert payload["score"] >= 0
    assert "highlightSets" in payload
    assert isinstance(payload["highlightSets"], list)


@pytest.mark.asyncio()
async def test_hottcad_simulate_endpoint(tmp_path):
    path = tmp_path / "sample.ifc"
    build_sample_ifc(path)

    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/v1/hottcad/simulate",
            json={"ifc_url": str(path)},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["proposed"]["connects"], "Expected connection proposals"
    assert payload["highlightSets"], "Expected highlight sets"
    first = payload["highlightSets"][0]
    assert first["productIds"], "Highlight set should contain product IDs"

