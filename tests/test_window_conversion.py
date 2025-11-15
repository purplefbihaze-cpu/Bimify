"""Unit tests for window mm/m conversion."""

from core.ifc.build_ifc43_model_v2 import WindowProfile, IFCConstants


def test_window_profile_keeps_mm():
    """Test that WindowProfile keeps values in mm, not meters."""
    profile = WindowProfile(
        width=1000.0,  # 1m in meters (for IFC)
        height=1000.0,
        thickness=200.0,
    )
    
    # Default lining values should be in mm
    assert profile.lining_thickness == IFCConstants.LINING_THICKNESS_MM
    assert profile.lining_depth == IFCConstants.LINING_DEPTH_MM
    assert profile.panel_thickness == IFCConstants.PANEL_THICKNESS_MM
    
    # Values should be in mm (50mm, 70mm, 50mm)
    assert profile.lining_thickness == 50.0
    assert profile.lining_depth == 70.0
    assert profile.panel_thickness == 50.0


def test_window_profile_custom_values():
    """Test WindowProfile with custom lining values in mm."""
    profile = WindowProfile(
        width=1000.0,
        height=1000.0,
        thickness=200.0,
        lining_thickness=60.0,  # 60mm
        lining_depth=80.0,  # 80mm
        panel_thickness=55.0,  # 55mm
    )
    
    # Custom values should be preserved in mm
    assert profile.lining_thickness == 60.0
    assert profile.lining_depth == 80.0
    assert profile.panel_thickness == 55.0

