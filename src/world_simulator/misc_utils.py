import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from scipy import ndimage
import noise
import pyfastnoisesimd as fns
from tqdm import tqdm
import logging
from typing import Dict, Any
import os
from joblib import Parallel, delayed

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
                              octaves: int, persistence: float, lacunarity: float) -> np.ndarray:
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
    simplex.fractal.type = fns.FractalType.FBM
    
    simplex.seed = int(seed)
    
    # CONVERSION NOTE:
    # The 'noise' library uses: noise(x / scale)
    # This library uses: noise(x * frequency)
    # Therefore: frequency = 1.0 / scale
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




def roughen_mask_edges(mask: np.ndarray, 
                       seed: int = 42, 
                       scale: float = 20.0, 
                       magnitude: float = 10.0,
                       persistence: float = 0.5,
                       lacunarity: float = 2.0,
                       tile_size: int = 2048) -> np.ndarray:
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
    
    # 1. CALCULATE SDF (Parallelized)
    # The new function handles the tiling and reconstruction
    sdf_map = _calculate_parallel_sdf(mask, tile_size, BUFFER_SIZE)
    

    # 2. Generate Perturbation Noise
    noise_field = _generate_noise_in_memory(
        shape=(height, width),
        scale=scale,
        seed=seed,
        octaves=4,        # 4 octaves usually sufficient for edge detail
        persistence=persistence,
        lacunarity=lacunarity
    )
    
    # 3. Apply Distortion
    # We add noise to the distance. 
    # If a pixel is 5px away from the edge (SDF=5), and noise is -0.6 * 10mag = -6,
    # New SDF = -1. It flips from Outside to Inside.
    distorted_sdf = sdf_map + (noise_field * magnitude)
    
    # Everything negative is the new Water
    return distorted_sdf < 0