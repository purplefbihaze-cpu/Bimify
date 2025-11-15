"""IFC Standard Values Configuration

This module contains all standard values for IFC generation.
All values are in meters unless otherwise specified.
"""

STANDARDS = {
    # --- WÄNDE ---
    "WALL_EXTERNAL_THICKNESS": 0.24,  # in Meter
    "WALL_INTERNAL_THICKNESS": 0.115,  # in Meter
    "WALL_MATERIAL_EXTERNAL": "Mauerwerk KS-Wi 12-0,9 24cm",
    "WALL_MATERIAL_INTERNAL": "Mauerwerk KS 11,5cm",
    "WALL_FIRE_RATING": "F90",
    "WALL_LOAD_BEARING": True,  # IfcBoolean
    "WALL_THERMAL_TRANSMITTANCE_EXTERNAL": 0.45,  # U-Wert in W/m²K
    "WALL_THERMAL_TRANSMITTANCE_INTERNAL": 0.77,  # U-Wert in W/m²K
    
    # --- FENSTER ---
    "WINDOW_SILL_HEIGHT": 0.9,  # OKFF + 0,9m = Brüstung
    "WINDOW_OVERALL_HEIGHT": 1.2,  # lichte Höhe der Öffnung
    "WINDOW_HEAD_HEIGHT": 2.1,  # Oberkante Fensteröffnung in Meter
    "WINDOW_FRAME_WIDTH": 0.07,  # in Meter
    "WINDOW_FRAME_DEPTH": 0.09,  # in Meter
    "WINDOW_U_VALUE": 1.4,  # W/m²K
    "WINDOW_GLASS_U_VALUE": 0.7,  # W/m²K
    "WINDOW_GLASS_AREA_RATIO": 0.8,  # 80% Glasfläche
    
    # --- TÜREN ---
    "DOOR_SILL_HEIGHT": 0.0,  # Schwellenhöhe ab OKFF
    "DOOR_WIDTH": 0.9,  # in Meter
    "DOOR_HEIGHT": 2.0,  # in Meter
    "DOOR_HEAD_HEIGHT": 2.0,  # Oberkante Türöffnung in Meter
    "DOOR_FRAME_WIDTH": 0.07,  # in Meter
    "DOOR_FIRE_RATING": "T30",
    "DOOR_HANDICAP_ACCESSIBLE": False,  # IfcBoolean
    "DOOR_OPENING_DEPTH": 1.0,  # Türbreite + Rahmen in Meter
    
    # --- DECKEN ---
    "SLAB_THICKNESS": 0.20,  # in Meter
    "SLAB_MATERIAL": "Stahlbeton C25/30",
    "SLAB_CONCRETE_COVER": 0.03,  # in Meter
    
    # --- BÖDEN ---
    "FLOOR_THICKNESS_STRUCTURE": 0.08,  # Estrich/Zementestrich in Meter
    "FLOOR_THICKNESS_FINISH": 0.005,  # Fliese/Belag in Meter
    
    # --- RÄUME ---
    "SPACE_HEIGHT": 2.5,  # Raumhöhe von OKFF bis Unterkante Decke in Meter
    
    # --- GEBÄUDE ---
    "BUILDING_TYPE": "RESIDENTIAL",
    "BUILDING_ENERGY_STANDARD": "KfW 55",
    
    # --- MATERIAL PROPERTIES ---
    "MATERIAL_MASONRY_DENSITY": 1800.0,  # kg/m³
    "MATERIAL_MASONRY_THERMAL_CONDUCTIVITY": 0.77,  # W/(m·K)
    "MATERIAL_CONCRETE_DENSITY": 2400.0,  # kg/m³
    "MATERIAL_CONCRETE_THERMAL_CONDUCTIVITY": 1.4,  # W/(m·K)
    "MATERIAL_GLASS_DENSITY": 2500.0,  # kg/m³
    "MATERIAL_GLASS_THERMAL_CONDUCTIVITY": 0.8,  # W/(m·K)
}

