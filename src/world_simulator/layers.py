import geopandas as gpd
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any
import numpy as np

@dataclass
class TerrainLayer:
    """Base protocol for all terrain layers."""
    name: str

@dataclass
class RiverWarpLayer(TerrainLayer):
    """
    Warps the underlying coordinate system (X, Y) based on proximity to river vectors.
    
    The engine rasterizes the provided river vectors and calculates a gravity-like
    pull towards them. This distorts the domain that subsequent noise layers will 
    sample from, causing tectonic ridges and erosion patterns to align naturally 
    with river valleys.

    Attributes:
        rivers_gdf: GeoDataFrame containing the river linestrings.
        width_column: Name of the attribute column in rivers_gdf containing width (meters).
                      This scales the MAGNITUDE of the warp (wide rivers pull harder).
        warp_strength: A global multiplier for the warp effect.
                       Shift (pixels) = Width * Strength * Falloff.
        influence_km: The maximum radius of the gravity well. The pull decays to zero
                      at this distance from the river bank.
    """
    rivers_gdf: gpd.GeoDataFrame
    width_column: str
    warp_strength: float
    influence_km: float

    def __post_init__(self):
        if self.width_column not in self.rivers_gdf.columns:
            raise ValueError(f"Column '{self.width_column}' not found in river GeoDataFrame.")

@dataclass
class TectonicLayer(TerrainLayer):
    """
    Generates structural terrain features (ridges, plates, bulk forms) using 
    SIMD-accelerated noise generation.

    This layer supports 'Batched Processing' in the engine to maintain low RAM usage
    regardless of map resolution.

    Attributes:
        seed: Random seed for noise generation.
        frequency: The scale of the features (lower = larger features).
                   Typical values: 0.005 (Continental) to 0.05 (Local).
        amplitude: The vertical height of the features (meters).
        ridge_mode: The specific cellular noise calculation to use.
                    - "Distance": Standard Voronoi (cobblestone/shattered look).
                    - "Distance2Sub": (F2-F1), creates sharp 'Dragon's Back' ridges.
        mask_config: Configuration dictionary for masking this layer.
                     
                     Schema A (Altitude Ramp):
                     {
                        'type': 'altitude', 
                        'min_height': 300.0,   # Start blending in here
                        'fade_height': 600.0   # Reach full strength 600m later (at 900m)
                     }

                     Schema B (Vector Shape):
                     {
                        'type': 'vector_shape', 
                        'geometry': shapely_geometry, 
                        'fade_km': 40.0        # Feather the edges
                     }
    """
    seed: int
    frequency: float
    amplitude: float
    ridge_mode: str
    mask_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        valid_modes = ["Distance", "Distance2Sub"]
        if self.ridge_mode not in valid_modes:
            raise ValueError(f"Invalid ridge_mode '{self.ridge_mode}'. Must be one of {valid_modes}")

@dataclass
class ErosionLayer(TerrainLayer):
    """
    Applies Domain Warped Multifractal noise to add weathering, 
    soil texture, and flow patterns to the terrain.
    """
    seed: int
    frequency: float
    amplitude: float
    fractal_type: str  # e.g., "RigidMulti", "FBM"
    perturb_strength: float
    mask_config: Optional[Dict[str, Any]] = None

@dataclass
class LakeIntegrationLayer(TerrainLayer):
    """
    Integrates vector lakes into the heightmap.
    1. Auto-detects lake elevation from the surrounding terrain shoreline.
    2. Flattens the water surface.
    3. Blends the banks to avoid artificial cliffs.
    """
    lakes_gdf: gpd.GeoDataFrame
    bank_width_km: float
    bed_depth_m: float = 0.0
    priority_mode: str = "Auto"

@dataclass
class SmartCoastalTaper(TerrainLayer):
    """
    Tapers the terrain to sea level using a 'smart' logic that distinguishes 
    between Cliffs (high elevation at coast) and Beaches (low elevation at coast).
    """
    mask_array: np.ndarray  # The fractalized boolean mask
    sea_level: float
    cliff_height_threshold: float
    beach_width_m: float
    cliff_width_m: float
    smoothing_sigma: float