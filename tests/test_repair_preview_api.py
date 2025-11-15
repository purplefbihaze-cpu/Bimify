from __future__ import annotations

import pytest
from httpx import AsyncClient

pytest.importorskip("ifcopenshell")
pytest.importorskip("shapely")

from services.api.main import app
from tests.utils_ifc import build_sample_ifc


@pytest.mark.asyncio()
async def test_repair_preview_and_commit(tmp_path):
    # Build a tiny IFC to preview
    path = tmp_path / "sample.ifc"
    build_sample_ifc(path)

    async with AsyncClient(app=app, base_url="http://test") as client:
        # Preview
        resp_preview = await client.post(
            "/v1/ifc/repair/preview",
            json={"ifc_url": str(path), "level": 1},
        )
        assert resp_preview.status_code == 200, resp_preview.text
        preview = resp_preview.json()
        assert preview["preview_id"]
        assert "overlay_url" in preview
        # Commit
        resp_commit = await client.post(
            "/v1/ifc/repair/commit",
            json={"preview_id": preview["preview_id"], "level": 1},
        )
        assert resp_commit.status_code == 200, resp_commit.text
        payload = resp_commit.json()
        assert payload["file_name"].endswith(".ifc")
        assert payload["level"] == 1
        assert "ifc_url" in payload


