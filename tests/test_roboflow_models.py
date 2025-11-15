from services.api.utils import _resolve_model


def test_resolve_model_default_geometry():
    cfg = _resolve_model()
    assert cfg.project == "floor-plan-ts7cp"
    assert cfg.version == 31


def test_resolve_model_rooms():
    cfg = _resolve_model("rooms")
    assert cfg.project == "floor-plans-zeb7z"
    assert cfg.version == 8


def test_resolve_model_verification():
    cfg = _resolve_model("verification")
    assert cfg.project == "cubicasa5k-2-qpmsa"
    assert cfg.version == 6

