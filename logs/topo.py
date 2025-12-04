from dataclasses import dataclass
import geopandas as gpd
import logging

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import momepy
import noise

import rasterio
from rasterio import features

import networkx as nx
import numpy as np
import pandas as pd

from scipy import ndimage
import shapely
from shapely.geometry import Point, LineString

from tqdm import tqdm
from typing import List, Optional, Union, Dict, Any


# Ensure we have a logger
logger = logging.getLogger(__name__)
from .logger import setup_logger
setup_logger(log_name="wfrp_phys_geo")



def vector_to_mask(vector_gdf: gpd.GeoDataFrame, reference_profile: Dict) -> np.ndarray:
    """
    Converts a Vector GeoDataFrame into a boolean Numpy Array (Mask) 
    that perfectly aligns with the reference raster profile.

    Logic:
    1. Extracts geometries from the GDF.
    2. Uses rasterio.features.rasterize to burn them into a grid.
    3. The grid dimensions and transform are taken from 'reference_profile'.

    Args:
        vector_gdf: Polygons to rasterize (e.g., Sea, Lakes).
        reference_profile: The rasterio profile of the Base DEM.

    Returns:
        np.ndarray: A boolean array (True = Inside Vector, False = Outside).
    """
    if vector_gdf is None or vector_gdf.empty:
        logger.warning("Empty vector layer provided for mask generation. Returning blank mask.")
        return np.zeros((reference_profile['height'], reference_profile['width']), dtype=bool)

    logger.info(f"Rasterizing {len(vector_gdf)} vector features into mask...")
    
    # Extract dimensions and transform from the profile
    height = reference_profile['height']
    width = reference_profile['width']
    transform = reference_profile['transform']
    
    # Rasterize
    # shapes expects list of (geometry, value) tuples
    # We burn '1' where the polygon exists
    shapes = ((geom, 1) for geom in vector_gdf.geometry)
    
    mask = features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,      # Background value
        dtype=rasterio.uint8
    )
    
    # Convert to Boolean
    return mask.astype(bool)




def flatten_by_mask(dem_array: np.ndarray, mask_array: np.ndarray, value: float = 0.0) -> np.ndarray:
    """
    Sets all pixels in the DEM that fall inside the mask to a specific constant value.
    Used for: Flattening the Sea.
    
    Args:
        dem_array: The terrain heightmap.
        mask_array: Boolean mask (True = Flatten this area).
        value: The height to set (default 0.0 for sea).
        
    Returns:
        np.ndarray: The modified DEM.
    """
    logger.info(f"Flattening masked area to elevation {value}...")
    
    # Work on a copy to avoid mutating the original input unexpectedly
    modified_dem = dem_array.copy()
    
    # Vectorized assignment (extremely fast)
    modified_dem[mask_array] = value
    
    return modified_dem



def flatten_lakes_smart(dem_array: np.ndarray, lakes_mask: np.ndarray) -> np.ndarray:
    """
    Flattens lakes to the elevation of their lowest neighbor.
    
    CRITICAL LOGIC:
    We cannot flatten all lakes to the SAME value. A mountain lake is higher than a valley lake.
    
    Algorithm:
    1. Use scipy.ndimage.label to identify distinct lake clusters (islands) in the mask.
    2. Iterate through each unique lake:
       a. Dilate the lake shape by 1 pixel to find the "Shoreline".
       b. Extract DEM values from the Shoreline pixels.
       c. Find the Minimum value of the Shoreline.
       d. Set the Lake pixels to that Minimum value.
       
    Args:
        dem_array: The terrain heightmap.
        lakes_mask: Boolean mask of all lakes.
        
    Returns:
        np.ndarray: The DEM with flattened lakes.
    """
    logger.info("Smart-flattening lakes to local shoreline elevation...")
    
    modified_dem = dem_array.copy()
    
    # 1. Identify distinct lakes
    # 'labeled_array' has a unique int ID for every separate lake blob
    # 'num_features' is the total count
    labeled_array, num_features = ndimage.label(lakes_mask)
    
    logger.info(f"  Found {num_features} distinct lakes.")
    
    # Structure for dilation (3x3 square connectivity)
    # This defines what "neighbor" means
    struct = ndimage.generate_binary_structure(2, 2)
    
    # 2. Iterate through lakes
    # We start at 1 because 0 is the background
    for lake_id in range(1, num_features + 1):
        
        # Create a boolean mask for JUST this specific lake
        # Note: For massive maps with 10k lakes, this can be optimized with bounding boxes,
        # but for typical Empire maps, this is fast enough.
        this_lake_mask = (labeled_array == lake_id)
        
        # Dilate to find the shoreline
        # binary_dilation expands the True area by 1 pixel in all directions
        dilated = ndimage.binary_dilation(this_lake_mask, structure=struct)
        
        # The Shoreline is the Dilated area MINUS the Lake itself
        # This gives us the ring of pixels touching the lake
        shoreline_mask = dilated & ~this_lake_mask
        
        # Edge Case: If lake covers the whole map (no shoreline), skip
        if not np.any(shoreline_mask):
            continue
            
        # Get the terrain heights at the shoreline
        shore_heights = modified_dem[shoreline_mask]
        
        if shore_heights.size > 0:
            # Find the lowest point on the shore (the drain point)
            water_level = np.min(shore_heights)
            
            # Flatten the lake to this level
            modified_dem[this_lake_mask] = water_level
            
    return modified_dem




@dataclass
class NoiseRule:
    """
    Defines a specific layer of fractal noise to be added to the terrain.
    """
    seed: int
    multiplier: float           # Vertical scale (e.g., 800.0)
    scale: float                # Horizontal scale (e.g., 50.0 for mountains, 400.0 for hills)
    min_elevation: float = -99999.0  # Apply only above this altitude in Base DEM
    mask_layer: str = None      # Name of mask in mask_library (e.g., 'north')
    octaves: int = 6
    persistence: float = 0.5
    lacunarity: float = 2.0

def _generate_noise_in_memory(shape: tuple, scale: float, seed: int, 
                              octaves: int, persistence: float, lacunarity: float) -> np.ndarray:
    """
    Private helper to generate 2D Perlin noise directly into a NumPy array.
    Returns values normalized roughly between 0 and 1 (or absolute billowy noise).
    """
    height, width = shape
    arr = np.zeros((height, width), dtype=np.float32)
    
    # We use tqdm here because noise generation is CPU intensive
    for y in tqdm(range(height), desc=f"Generating Noise (Scale {scale})", leave=False):
        for x in range(width):
            # Using 'mountain_mode' logic (abs) by default as it looks best for terrain
            val = noise.pnoise2(
                x / scale,
                y / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
                base=seed
            )
            arr[y][x] = abs(val) # Billowy noise
            
    # # Normalize to 0.0 - 1.0 range based on actual min/max generated
    # # This ensures the 'multiplier' in the rule implies a consistent height addition
    # arr_min, arr_max = arr.min(), arr.max()
    # if arr_max > arr_min:
    #     arr = (arr - arr_min) / (arr_max - arr_min)
        
    return arr

def apply_noise_rules(base_dem: np.ndarray, 
                      rules: List[NoiseRule], 
                      mask_library: Dict[str, np.ndarray] = {}) -> np.ndarray:
    """
    The Master Compositor. Stacks multiple noise layers onto the base DEM.
    
    LOGIC UPDATE:
    - If a rule has BOTH min_elevation AND a mask_layer, they are combined with OR (|).
      (e.g., "Apply if > 500m OR inside North mask").
    - If a rule has only one, it applies that one.
    - If a rule has neither, it applies everywhere.
    
    Args:
        base_dem: The starting topography (NumPy array).
        rules: List of NoiseRule objects defining the layers.
        mask_library: Dictionary mapping names ('north', 'sea') to Numpy Boolean Arrays.
        
    Returns:
        np.ndarray: The final composite terrain.
    """
    logger.info(f"Applying {len(rules)} noise rules to base terrain...")
    
    # 1. Copy Base DEM (Float32 for precision)
    final_dem = base_dem.astype(np.float32).copy()
    height, width = final_dem.shape
    
    for i, rule in enumerate(rules):
        logger.info(f"  Rule {i+1}: Scale={rule.scale}, Mult={rule.multiplier}, Seed={rule.seed}")
        
        # 2. Generate Raw Noise
        raw_noise = _generate_noise_in_memory(
            shape=(height, width),
            scale=rule.scale,
            seed=rule.seed,
            octaves=rule.octaves,
            persistence=rule.persistence,
            lacunarity=rule.lacunarity
        )
        
        # 3. Scale Noise
        scaled_noise = raw_noise * rule.multiplier
        
        # 4. Build Application Mask (The OR Logic)
        
        # Check which conditions are active
        has_elevation = rule.min_elevation > -9999.0
        has_mask = (rule.mask_layer is not None)
        
        # Resolve the mask layer if it exists
        spatial_mask = None
        if has_mask:
            if rule.mask_layer in mask_library:
                spatial_mask = mask_library[rule.mask_layer]
            else:
                logger.warning(f"    Mask '{rule.mask_layer}' not found! Ignoring mask condition.")
                has_mask = False

        # Combine Conditions
        if has_elevation and has_mask:
            # Elevation OR Mask
            elevation_mask = (base_dem >= rule.min_elevation)
            app_mask = elevation_mask | spatial_mask
            logger.info(f"    Filtered by Elevation > {rule.min_elevation} OR Mask '{rule.mask_layer}'")
            
        elif has_elevation:
            # Only Elevation
            app_mask = (base_dem >= rule.min_elevation)
            logger.info(f"    Filtered by Elevation > {rule.min_elevation}")
            
        elif has_mask:
            # Only Mask
            app_mask = spatial_mask
            logger.info(f"    Filtered by Mask '{rule.mask_layer}'")
            
        else:
            # No filters -> Global
            app_mask = np.ones((height, width), dtype=bool)
            logger.info(f"    No filters (Global application)")

        # 5. Apply
        final_dem[app_mask] += scaled_noise[app_mask]
        
    logger.info("Terrain composition complete.")
    return final_dem

