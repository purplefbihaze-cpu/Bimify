import asyncio
import sys
import types

import pytest


class _InferenceStub:
    def __init__(self, *args, **kwargs):
        self.inference_configuration = types.SimpleNamespace(
            confidence_threshold=0.0,
            iou_threshold=0.0,
        )

    def use_configuration(self, configuration):
        class _Ctx:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _Ctx()

    def infer(self, *args, **kwargs):
        return {}


sys.modules.setdefault("inference_sdk", types.SimpleNamespace(InferenceHTTPClient=_InferenceStub))

from core.ml.roboflow_client import RFOptions, infer_floorplan_with_raw


def test_confidence_passed_through(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "dummy-key")

    captured: dict[str, float | str | Path] = {}

    async def fake_infer(client, image_path, model_id, *, confidence, overlap):
        captured["confidence"] = confidence
        captured["overlap"] = overlap
        captured["model_id"] = model_id
        return {
            "predictions": [
                {"class": "wall", "confidence": 0.05},
                {"class": "door", "confidence": 0.01},
            ],
            "image": {"width": 100, "height": 100},
        }

    monkeypatch.setattr(
        "core.ml.roboflow_client._infer_with_client",
        lambda client, image_path, model_id, *, confidence, overlap: fake_infer(
            client,
            image_path,
            model_id,
            confidence=confidence,
            overlap=overlap,
        ),
    )
    monkeypatch.setattr("core.ml.roboflow_client._get_client", lambda *args, **kwargs: object())

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake")

    opts = RFOptions(project="workspace/project", version=1, confidence=0.37, overlap=0.2)
    preds, raw = asyncio.run(infer_floorplan_with_raw(image_path, opts=opts))

    assert captured["confidence"] == pytest.approx(0.37)
    assert captured["overlap"] == pytest.approx(0.2)
    assert captured["model_id"] == "project/1"
    assert len(preds) == 2
    assert raw["predictions"][0]["confidence"] == 0.05


def test_per_class_thresholds_do_not_filter(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "dummy-key")

    async def fake_infer(*args, **kwargs):
        return {
            "predictions": [
                {"class": "wall", "confidence": 0.05},
                {"class": "wall", "confidence": 0.001},
            ],
        }

    monkeypatch.setattr(
        "core.ml.roboflow_client._infer_with_client",
        lambda *args, **kwargs: fake_infer(*args, **kwargs),
    )
    monkeypatch.setattr("core.ml.roboflow_client._get_client", lambda *args, **kwargs: object())

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake")

    opts = RFOptions(
        project="workspace/project",
        version=1,
        confidence=0.5,
        overlap=0.3,
        per_class={"wall": 0.9},
    )
    preds, raw = asyncio.run(infer_floorplan_with_raw(image_path, opts=opts))

    assert len(preds) == 2
    assert len(raw["predictions"]) == 2

