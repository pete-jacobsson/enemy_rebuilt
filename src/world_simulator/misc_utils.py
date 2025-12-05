from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import noise
import pyfastnoisesimd as fns
from tqdm import tqdm
import logging
from typing import Dict, Any
import os
from joblib import Parallel, delayed

from .topo import CoastalTaper

# Ensure we have a logger
logger = logging.getLogger(__name__)
from .logger import setup_logger
setup_logger(log_name="wfrp_phys_geo")

# --- NUMPY 2.0 COMPATIBILITY PATCH ---
# pyfastnoisesimd relies on np.product, which was removed in NumPy 2.0.
# We manually restore it as an alias to np.prod.
if not hasattr(np, 'product'):
    np.product = np.prod
# -------------------------------------

# ==========================================
#        PARALLEL WORKER FUNCTIONS
# (Must be at top level for Joblib pickling)
# ==========================================

def _process_sdf_task(mask_tile: np.ndarray, pad_top: int, pad_left: int, core_height: int, core_width: int) -> np.ndarray:
    """
    Internal worker to calculate SDF on a padded tile and extract the core result.
    """
    # Calculate the EDT on the padded tile
    dist_outside = ndimage.distance_transform_edt(~mask_tile)
    dist_inside = ndimage.distance_transform_edt(mask_tile)
    sdf = dist_outside - dist_inside
    
    # Extract the un-padded core result
    y_start = pad_top
    y_end = pad_top + core_height
    x_start = pad_left
    x_end = pad_left + core_width
    
    return sdf[y_start:y_end, x_start:x_end]


def _process_blur_task(mask_tile: np.ndarray, sigma: float, pad_top: int, pad_left: int, core_height: int, core_width: int) -> np.ndarray:
    """Worker to apply Gaussian Blur to a tile."""
    float_tile = mask_tile.astype(np.float32)
    blurred = gaussian_filter(float_tile, sigma=sigma)
    
    y_start = pad_top
    y_end = pad_top + core_height
    x_start = pad_left
    x_end = pad_left + core_width
    
    return (blurred[y_start:y_end, x_start:x_end] > 0.5)


# ==========================================
#             HELPER FUNCTIONS
# ==========================================
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


def _generate_noise_in_memory(shape: tuple, scale: float, seed: int, 
                              octaves: int, persistence: float, lacunarity: float,
                              fractal_type: str = 'FBM') -> np.ndarray:
    """
    High-Performance Vectorized SIMD Noise Generation.
    
    This function uses CPU vector instructions (AVX2/AVX512) to generate 
    Fractal Brownian Motion (FBM) noise. It is significantly faster than standard 
    loops for large rasters (e.g., 16k x 10k).

    --- FRACTAL DIMENSION & ROUGHNESS ---
    
    The geometric "roughness" of the coastline is determined by the FRACTAL DIMENSION (D).
    In this implementation, D is controlled primarily by the 'persistence' (gain) parameter.

    The relationship for Fractional Brownian Motion (FBM) is approximately:
        Persistence (G) = 2^(-H)
        Fractal Dimension (D) = 2 - H  (for 1D coastlines)

    Practical Translation:
    - Low Persistence (e.g., 0.45): 
        - H is higher (~1.15). 
        - D is lower (~1.1 - 1.2).
        - Result: A "Smooth" fractal. The coastline looks like sweeping sandy bays. 
          Fine details are suppressed; the line is relatively straight at high zoom.
    
    - High Persistence (e.g., 0.65): 
        - H is lower (~0.6).
        - D is higher (~1.3 - 1.4).
        - Result: A "Rough" fractal. The coastline looks like jagged rocks or fjords.
          Fine details are almost as strong as large features; the line remains 
          complex and "wiggly" even at high zoom.

    Args:
        shape: Tuple of (height, width).
        scale: The input coordinate scale (Note: logic is inverted vs 'noise' lib).
               This function handles the conversion: Frequency = 1.0 / scale.
        seed: Random seed.
        octaves: Number of layers of noise. 
                 4 is usually sufficient for masks; 6-8 for terrain.
        persistence: (Gain) Amplitude multiplier per octave (0.0 to 1.0).
        lacunarity: Frequency multiplier per octave (usually 2.0).

    Returns:
        np.ndarray: float32 array with values roughly between -1.0 and 1.0.
    """
    height, width = shape
    
    # 1. Initialize SIMD Noise State
    # numWorkers uses all physical cores by default.
    # We must ensure the library allocates the memory to guarantee 
    # the 64-byte alignment required by AVX512/AVX2.
    simplex = fns.Noise(numWorkers = min(1, os.cpu_count() - 1))
    
    simplex.noiseType = fns.NoiseType.Perlin
    simplex.seed = int(seed)

    # --- FRACTAL TYPE SELECTION ---
    if fractal_type == 'RigidMulti':
        simplex.fractal.type = fns.FractalType.RigidMulti
    elif fractal_type == 'Billow':
        simplex.fractal.type = fns.FractalType.Billow
    else:
        simplex.fractal.type = fns.FractalType.FBM
    # ------------------------------


    
    # CONVERSION NOTE:
    # The 'noise' library uses: noise(x / scale)
    # This library uses: noise(x * frequency)
    # Therefore: frequency = 1.0 / scale
    if scale == 0: scale = 0.0001
    simplex.frequency = 1.0 / scale
    
    simplex.fractal.octaves = int(octaves)
    simplex.fractal.lacunarity = float(lacunarity)
    simplex.fractal.gain = float(persistence)

    # 2. PADDING LOGIC (The Fix)
    # The array size must be divisible by the SIMD vector length.
    # We pad the WIDTH to ensure the total size is valid.
    # 32 is a safe alignment block (covers AVX2/AVX512 float32 requirements)
    align_block = 32
    padded_width = width
    
    remainder = width % align_block
    if remainder != 0:
        padded_width = width + (align_block - remainder)
    
    # 3. Generate Grid
    # genAsGrid generates noise based on coordinate inputs.
    # It returns a 3D array [z, y, x]. Since we are 2D, z=1.
    # The library handles multi-threading internally.
    logger.debug(f"SIMD Noise Gen: {width}x{height} (Padded to {padded_width}) | Scale: {scale}")
    

    # This allocates aligned memory automatically
    result_buffer = simplex.genAsGrid(shape=[1, height, padded_width])
    
    # 4. Reshape and Return
    # We generated extra data to satisfy the CPU; now we slice it off.
    # [Z, Y, X] -> Strip Z -> Slice Y to height (unchanged) -> Slice X to original width
    return result_buffer[0, :height, :width]





# --- SDF GENERATION HELPER ---

def _calculate_parallel_sdf(mask: np.ndarray, tile_size: int, buffer_size: int) -> np.ndarray:
    """
    Orchestrates the parallel calculation of the Signed Distance Field (SDF).
    """
    logger.info("  Starting parallel SDF calculation...")
    
    height, width = mask.shape
    n_jobs = max(1, os.cpu_count() - 1)
    
    tasks = []
    
    for y_start in range(0, height, tile_size):
        for x_start in range(0, width, tile_size):
            
            # 1. Define slice boundaries including the buffer
            y_end = min(y_start + tile_size, height)
            x_end = min(x_start + tile_size, width)
            
            y_pad_start = max(0, y_start - buffer_size)
            x_pad_start = max(0, x_start - buffer_size)
            y_pad_end = min(height, y_end + buffer_size)
            x_pad_end = min(width, x_end + buffer_size)
            
            # 2. Extract the tile data and pass necessary padding information
            tile_mask = mask[y_pad_start:y_pad_end, x_pad_start:x_pad_end]
            
# --- CRITICAL FIX: Ensure task dictionary has the end coordinates ---
            tasks.append({
                'mask_tile': tile_mask,
                'y_start': y_start,
                'x_start': x_start,
                'y_end': min(y_start + tile_size, height),  # <--- Ensure y_end is calculated and present
                'x_end': min(x_start + tile_size, width),    # <--- Ensure x_end is calculated and present
                'core_height': y_end - y_start,
                'core_width': x_end - x_start,
                'pad_top': y_start - y_pad_start,
                'pad_left': x_start - x_pad_start
            })
            
    # 3. Run parallel processing (backend='loky' uses processes for multiprocessing stability)
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_process_sdf_task)(
            task['mask_tile'], 
            task['pad_top'], 
            task['pad_left'], 
            task['core_height'], 
            task['core_width']
        ) for task in tqdm(tasks, desc="SDF Tiling", leave=False)
    )
    
    # 4. Reconstruct the final SDF map
    sdf_map = np.empty(mask.shape, dtype=np.float32)
    
    for result, task in zip(results, tasks):
        sdf_map[task['y_start']:task['y_end'], task['x_start']:task['x_end']] = result
        
    logger.info("  Parallel SDF reconstruction complete.")
    return sdf_map





def _smooth_mask_parallel(mask: np.ndarray, sigma: float, tile_size: int) -> np.ndarray:
    """
    Orchestrates parallel Gaussian smoothing to remove 'stair-step' artifacts 
    from low-res source vectors.
    """
    if sigma <= 0:
        return mask

    logger.info(f"  Pre-smoothing mask (Sigma={sigma})...")
    
    height, width = mask.shape
    n_jobs = max(1, os.cpu_count() - 1)
    
    # Buffer needs to be approx 3x Sigma to catch the blur falloff
    buffer_size = int(max(20, sigma * 4))
    
    tasks = []
    
    for y_start in range(0, height, tile_size):
        for x_start in range(0, width, tile_size):
            y_end = min(y_start + tile_size, height)
            x_end = min(x_start + tile_size, width)
            
            y_pad_start = max(0, y_start - buffer_size)
            x_pad_start = max(0, x_start - buffer_size)
            y_pad_end = min(height, y_end + buffer_size)
            x_pad_end = min(width, x_end + buffer_size)
            
            tasks.append({
                'mask_tile': mask[y_pad_start:y_pad_end, x_pad_start:x_pad_end],
                'y_start': y_start,
                'x_start': x_start,
                'y_end': y_end,
                'x_end': x_end,
                'core_height': y_end - y_start,
                'core_width': x_end - x_start,
                'pad_top': y_start - y_pad_start,
                'pad_left': x_start - x_pad_start
            })
            
    # Run Parallel Blur
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_process_blur_task)(
            task['mask_tile'], 
            sigma,
            task['pad_top'], 
            task['pad_left'], 
            task['core_height'], 
            task['core_width']
        ) for task in tqdm(tasks, desc="Smoothing", leave=False)
    )
    
    # Reconstruct
    smoothed_map = np.zeros_like(mask)
    
    for result, task in zip(results, tasks):
        smoothed_map[task['y_start']:task['y_end'], task['x_start']:task['x_end']] = result
        
    return smoothed_map


def roughen_mask_edges(mask: np.ndarray, 
                       seed: int = 42, 
                       scale: float = 20.0, 
                       magnitude: float = 10.0,
                       persistence: float = 0.5,
                       lacunarity: float = 2.0,
                       tile_size: int = 2048,
                       fractal_type: str = 'FBM',
                       pre_smooth: float = 0.0) -> np.ndarray:
    """
    Applies Domain Warping (Fractal Displacement) to the edges of a boolean mask.
    
    The Method:
    1. Converts Mask -> Signed Distance Field (SDF).
    2. Generates a Noise Field.
    3. Perturbs the SDF by the Noise.
    4. Re-thresholds the SDF to create a new, jagged mask.

    Args:
        mask: The boolean mask to roughen.
        seed: Random seed.
        scale: Frequency of the wobble (Lower = High Freq/Jagged, Higher = Broad bays).
               Note: scale is relative to pixel coordinates.
        magnitude: Strength of the distortion in PIXELS.
        persistence: Fractal Dimension control. 
                     0.5 = Smooth/Average. 
                     0.6-0.7 = High Fractal Dimension (Jagged/Fjords).
        lacunarity: Detail frequency gap.

    Returns:
        np.ndarray: A new boolean mask with fractal edges.
    """
    logger.info(f"Fractalizing edges (Scale={scale}, Mag={magnitude}, Pers={persistence})...")
    
    height, width = mask.shape
    
    # Default buffer size for distance calculation (e.g., 25km at 125m/px)
    BUFFER_SIZE = 200 


    # 0. PRE-SMOOTHING (The Fix for Blocky Pixels)
    # Initialize variable first to ensure scope safety
    working_mask = mask 
    
    if pre_smooth > 0:
        working_mask = _smooth_mask_parallel(mask, pre_smooth, tile_size)

    
    # 1. CALCULATE SDF (Parallelized)
    # The new function handles the tiling and reconstruction
    sdf_map = _calculate_parallel_sdf(working_mask, tile_size, BUFFER_SIZE)
    

    # 2. Generate Perturbation Noise
    noise_field = _generate_noise_in_memory(
        shape=(height, width),
        scale=scale,
        seed=seed,
        octaves=4,        # 4 octaves usually sufficient for edge detail
        persistence=persistence,
        lacunarity=lacunarity,
        fractal_type=fractal_type
    )

    # --- CRITICAL FIX FOR RIGID NOISE ---
    if fractal_type == 'RigidMulti':
        # 1. Invert: Turns "Mountain Ridges" (Jagged) into "Canyons" (Jagged)
        #    This ensures the coastline cuts IN along the sharp lines.
        noise_field = -noise_field 
        
        # 2. Center: Rigid noise is often positive [0, 1] or skewed.
        #    We force it to roughly [-0.5, 0.5] so it distorts inward AND outward.
        #    (Adjust '0.5' if your specific library output range differs)
        noise_field = noise_field - np.mean(noise_field)

    
    # 3. Apply Distortion
    # We add noise to the distance. 
    # If a pixel is 5px away from the edge (SDF=5), and noise is -0.6 * 10mag = -6,
    # New SDF = -1. It flips from Outside to Inside.
    distorted_sdf = sdf_map + (noise_field * magnitude)
    
    # Everything negative is the new Water
    return distorted_sdf < 0


# ==========================================
#             Fractal Blender elements
# ==========================================


@dataclass
class FractalLayer:
    """
    Configuration for a single layer of procedural terrain noise.
    """
    name: str
    seed: int
    frequency_scale: float      # Horizontal size (Lower = Higher Freq/Detail)
    vertical_scale: float       # Vertical amplitude (Meters)
    
    fractal_type: str = "FBM"   # "FBM", "RigidMulti", "Billow"
    octaves: int = 6
    persistence: float = 0.5
    lacunarity: float = 2.0
    
    # Mask Config: None (Global), or a Dict defining the rule
    # Examples:
    #   None -> Apply Everywhere
    #   {'type': 'vector_fade', 'vector_key': 'north', 'fade_km': 200.0}
    #   {'type': 'altitude', 'min_height': 500.0, 'fade_height': 200.0}
    mask_config: Optional[Dict[str, Any]] = None




class FractalBlender:
    """
    Stateful processor that manages the generation and composition of 
    fractal terrain layers.
    
    Separates concerns into:
    1. Noise Generation (SIMD/Physics)
    2. Mask Resolution (Spatial/Contextual)
    3. Composition (Accumulation)
    """
    
    def __init__(self, 
                 base_dem: np.ndarray, 
                 vector_context: Dict[str, gpd.GeoDataFrame], 
                 profile: Dict[str, Any]):
        """
        Args:
            base_dem: The starting elevation model (Canvas).
            vector_context: Dictionary of loaded GeoDataFrames for masking.
            profile: Rasterio profile (transform, crs) for aligning vectors.
        """
        self.profile = profile
        self.vector_context = vector_context
        
        # Working state
        self.shape = base_dem.shape
        self.pixel_size = profile['transform'][0] # Assumes positive x-res
        
        # Initialize accumulator with base DEM (float32 for precision)
        self.current_terrain = base_dem.astype(np.float32).copy()

    def process(self, recipe: List[FractalLayer]) -> np.ndarray:
        """
        Main Entry Point: Iterates through the recipe and evolves the terrain.
        """
        logger.info(f"Starting Fractal Blender ({len(recipe)} layers)...")
        
        for i, layer in enumerate(recipe):
            logger.info(f"  [{i+1}/{len(recipe)}] Processing Layer: {layer.name}")
            self._process_single_layer(layer)
            
        return self.current_terrain

    def _process_single_layer(self, layer: FractalLayer):
        """
        Orchestrates the lifecycle of a single layer to minimize RAM usage.
        """
        # A. Generate Noise
        noise_map = self._generate_layer_noise(layer)
        
        # B. Resolve Mask (Control Signal)
        mask_map = self._resolve_layer_mask(layer)
        
        # C. Composite
        # Math: Terrain += Noise * Amplitude * Mask
        contribution = noise_map * layer.vertical_scale * mask_map
        self.current_terrain += contribution
        
        # D. Garbage Collection
        # Explicit deletion to ensure immediate memory freeing
        del noise_map
        del mask_map
        del contribution

    def _generate_layer_noise(self, layer: FractalLayer) -> np.ndarray:
        """
        Concern: Physics & Patterns.
        Handles SIMD generation and fractal-type specific post-processing.
        """
        height, width = self.shape
        
        raw_noise = _generate_noise_in_memory(
            shape=(height, width),
            scale=layer.frequency_scale,
            seed=layer.seed,
            octaves=layer.octaves,
            persistence=layer.persistence,
            lacunarity=layer.lacunarity,
            fractal_type=layer.fractal_type
        )
        
        # Post-Processing: Fix RigidMulti to look like mountains, not bubbles
        if layer.fractal_type == 'RigidMulti':
            # Invert: Turns "Ridges" into "Canyons" (or spikes if added)
            # Center: Ensure it pushes up AND down
            raw_noise = (raw_noise * -1.0) - np.mean(raw_noise * -1.0)
            
        return raw_noise

    def _resolve_layer_mask(self, layer: FractalLayer) -> np.ndarray:
        """
        Concern: Context & Geography.
        Interprets abstract mask configs into concrete 0.0-1.0 float arrays.
        """
        # Default: Global application (1.0 everywhere)
        if not layer.mask_config:
            return np.ones(self.shape, dtype=np.float32)

        conf = layer.mask_config
        m_type = conf.get('type')
        
        if m_type == 'vector_fade':
            return self._create_vector_fade_mask(conf)
        
        elif m_type == 'altitude':
            return self._create_altitude_mask(conf)
        
        else:
            logger.warning(f"    Unknown mask type '{m_type}'. Applying globally.")
            return np.ones(self.shape, dtype=np.float32)

    def _create_vector_fade_mask(self, conf: Dict[str, Any]) -> np.ndarray:
        """Helper for Vector Fade logic."""
        vec_key = conf['vector_key']
        fade_km = conf.get('fade_km', 100.0)
        invert = conf.get('invert', False)
        
        if vec_key not in self.vector_context:
            logger.warning(f"    Vector '{vec_key}' missing! Returning 0.0 mask.")
            return np.zeros(self.shape, dtype=np.float32)
            
        # 1. Rasterize
        bool_mask = vector_to_mask(self.vector_context[vec_key], self.profile)
        
        # 2. Distance Transform
        # Calculate distance in pixels
        dist_px = (fade_km * 1000) / self.pixel_size
        
        # Distance to the "Core" (True pixels)
        # edt calculates distance to nearest zero, so we invert.
        dist_to_core = ndimage.distance_transform_edt(~bool_mask)
        
        # 3. Normalize (Linear Fade)
        # 1.0 inside, fading to 0.0 at dist_px
        weight = 1.0 - (dist_to_core / dist_px)
        weight = np.clip(weight, 0.0, 1.0)
        
        if invert:
            return 1.0 - weight
            
        return weight.astype(np.float32)

    def _create_altitude_mask(self, conf: Dict[str, Any]) -> np.ndarray:
        """Helper for Altitude Ramp logic."""
        min_h = conf.get('min_height', 0.0)
        fade_h = conf.get('fade_height', 100.0)
        
        # Calculate Ramp based on CURRENT terrain state
        # (Allows stacking mountains on top of previous layers)
        ramp = (self.current_terrain - min_h) / fade_h
        ramp = np.clip(ramp, 0.0, 1.0)
        
        return ramp.astype(np.float32)





