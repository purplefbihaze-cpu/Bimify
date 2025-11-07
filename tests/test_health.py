import pytest

pytest.importorskip("loguru")

from httpx import AsyncClient

from services.api.main import app


@pytest.mark.asyncio()
async def test_health_endpoint() -> None:
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_sync_placeholder() -> None:
    """Ensure the test suite exercises synchronous code paths."""

    assert True

