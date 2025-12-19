import cupy as cp
import ctypes
import glob

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
import os
import pandas as pd
from pathlib import Path
import rasterio

from scipy import ndimage
from scipy.ndimage import zoom
import shapely
from shapely.geometry import Point, LineString
import sys

from tqdm import tqdm
from typing import List, Optional, Union, Dict, Any, Tuple

from time import time




# Ensure we have a logger
logger = logging.getLogger(__name__)
from .logger import setup_logger
setup_logger(log_name="wfrp_phys_geo")



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





logger = logging.getLogger(__name__)

class CoastalTaper:
    """
    Applies a smooth elevation taper (attenuation) near the coastline.
    
    The goal is to fix the 'Cliffs of Dover' problem where land meets sea at 
    an abrupt 200m cliff. This tool creates a gradient that forces the 
    terrain to 0m at the coast and smoothly ramps up to full height inland.
    
    Optimizations:
    1. Downscaling: Calculates heavy gradients at low resolution.
    2. Float16: Uses half-precision floats for the mask to save ~300MB RAM.
    """
    
    def __init__(self, sea_mask: np.ndarray, taper_distance_px: float):
        """
        Args:
            sea_mask: Boolean mask (True = Sea).
            taper_distance_px: Distance in pixels over which the land fades in.
        """
        self.sea_mask = sea_mask
        self.taper_dist = taper_distance_px
        self.taper_map = None

    def generate_taper_map(self, power: float = 2.0, downsample_factor: float = 0.1) -> np.ndarray:
        """
        Orchestrator: Generates the attenuation map using low-res approximation.
        
        Args:
            power: Curve steepness (1.0 = Linear, 2.0 = Quadratic).
            downsample_factor: Resolution scale (0.1 = 10%).
            
        Returns:
            np.ndarray: float16 map (0.0 at coast, 1.0 inland).
        """
        logger.info(f"Generating Coastal Taper (Dist={self.taper_dist}px, Power={power})...")
        
        # 1. Create low-res working canvas
        small_mask = self._downscale_mask(self.sea_mask, downsample_factor)
        
        # 2. Calculate distance field on the small mask
        # Adjust the target distance to match the new scale
        scaled_dist = self.taper_dist * downsample_factor
        small_gradient = self._calculate_gradient_field(small_mask, scaled_dist, power)
        
        # 3. Upscale back to full resolution
        full_shape = self.sea_mask.shape
        self.taper_map = self._upscale_to_target(small_gradient, full_shape)
        
        return self.taper_map

    def apply(self, dem: np.ndarray) -> np.ndarray:
        """
        Multiplies the DEM by the generated taper map.
        
        FIX: Calculates in float32 to capture the gradient, then rounds 
        and casts back to the original input type (e.g. int16) to save RAM.
        """
        if self.taper_map is None:
            raise ValueError("Run generate_taper_map() first.")
            
        logger.info(f"Applying Coastal Taper (returning {dem.dtype})...")
        
        # 1. Capture original type
        original_dtype = dem.dtype
        
        # 2. Promote to float32 for calculation
        # This prevents the 'int * float' issue where 0.99 becomes 0
        dem_float = dem.astype(np.float32)
        taper_float = self.taper_map.astype(np.float32)
        
        # 3. Multiply
        result = dem_float * taper_float
        
        # 4. Cast back to original type
        # If the input was integer (int16), we round to nearest to avoid 
        # truncation bias (e.g., 99.9 becoming 99 instead of 100).
        if np.issubdtype(original_dtype, np.integer):
            return np.round(result).astype(original_dtype)
        else:
            return result.astype(original_dtype)

    def _downscale_mask(self, mask: np.ndarray, factor: float) -> np.ndarray:
        """Helper: shrinks the boolean mask."""
        if factor >= 1.0:
            return mask
        
        logger.debug(f"  Downscaling mask by factor {factor}...")
        # Order 0 = Nearest Neighbor (keeps it boolean/integer)
        return zoom(mask, factor, order=0)

    def _calculate_gradient_field(self, mask: np.ndarray, max_dist: float, power: float) -> np.ndarray:
        """
        Helper: Calculates normalized distance field on a (potentially small) array.
        Returns float16 to save memory.
        """
        logger.debug("  Calculating Distance Transform...")
        
        # Calculate distance to nearest "False" (Land pixels look for Sea)
        # Invert mask: True=Sea. We want distance FROM Sea (0 at sea).
        # edt calculates distance to background (0). So we want Sea to be 0.
        # Input to edt should be LAND (True) and SEA (False/0).
        # Our self.sea_mask is True=Sea. So we pass ~mask (True=Land).
        dist_field = ndimage.distance_transform_edt(~mask)
        
        # Normalize to 0.0 - 1.0
        # We calculate in float32 for precision, then cast to float16
        gradient = np.clip(dist_field / max_dist, 0.0, 1.0)
        
        # Apply Curve
        if power != 1.0:
            gradient = np.power(gradient, power)
            
        return gradient.astype(np.float16)

    def _upscale_to_target(self, low_res_map: np.ndarray, target_shape: tuple) -> np.ndarray:
        """
        Helper: Resizes the low-res gradient back to full dimensions.
        """
        current_shape = low_res_map.shape
        if current_shape == target_shape:
            return low_res_map
            
        logger.debug(f"  Upscaling result to {target_shape}...")
        
        # Calculate exact zoom factors
        zoom_y = target_shape[0] / current_shape[0]
        zoom_x = target_shape[1] / current_shape[1]
        
        # CRITICAL FIX: Scipy zoom crashes on float16. 
        # We must promote to float32 for the calculation.
        input_array = low_res_map.astype(np.float32)
        
        # Order 1 = Bilinear Interpolation
        upscaled = zoom(input_array, (zoom_y, zoom_x), order=1)
        
        # Safety: Zoom can sometimes be off by +/- 1 pixel due to rounding
        if upscaled.shape != target_shape:
            h, w = target_shape
            # If bigger, crop
            if upscaled.shape[0] >= h and upscaled.shape[1] >= w:
                upscaled = upscaled[:h, :w]
            else:
                # If smaller (rare), pad with edge values
                # This is a robust fallback
                from numpy import pad
                diff_h = h - upscaled.shape[0]
                diff_w = w - upscaled.shape[1]
                upscaled = np.pad(upscaled, ((0, max(0, diff_h)), (0, max(0, diff_w))), mode='edge')

        # Cast back to float16 to save RAM
        return upscaled.astype(np.float16)



###################################################################################################
#########  DEM Topography smoothing with CuPy
###################################################################################################

import cupy as cp
import rasterio
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union

class GPUThermalEroder:
    """
    A GPU-accelerated implementation of thermal erosion (granular diffusion) 
    using Cellular Automata on a raster grid.
    This is a pure-Python/CuPy implementation of thermal erosion that does NOT require 
    the NVRTC compiler. It uses vectorized array operations.

    Mathematical Model:
    -------------------
    The simulation models the terrain as a grid of discrete cells where material 
    moves from a center cell $C$ to its neighbors $N_i$ (North, South, East, West) 
    if the slope exceeds a critical stability threshold (Talus Angle).

    For each neighbor $N_i$, we calculate the height difference:
    $$ Delta h_i = H_C - H_{N_i} $$

    Material flows only if the difference exceeds the Talus Threshold $T$:
    $$ text{Flow}_i = begin{cases} Delta h_i - T & text{if } Delta h_i > T  0 & text{otherwise} end{cases} $$

    The total material removed from the center cell is proportional to the sum of flows, 
    scaled by an erosion rate $R$ (relaxation constant, $0 < R le 0.5$):
    $$ Delta H_C = - R cdot sum text{Flow}_i $$

    This creates a non-linear diffusion process where plateaus remain stable until 
    their edges are "eaten away" by the critical angle constraint, naturally forming 
    conical slopes.

    Attributes:
        talus_threshold (float): The height difference required to trigger material movement.
        erosion_rate (float): The speed of the simulation (stability factor).


    # --- Usage Example ---
    if __name__ == "__main__":
        eroder = GPUThermalEroder(talus_threshold=2.5, erosion_rate=0.1)
        
        # 1. Load
        raw_dem = eroder.load_dem("input_blocky.tif")
        
        # 2. Process
        smoothed_dem = eroder.process(raw_dem, iterations=200)
        
        # 3. Save
        eroder.save_dem(smoothed_dem, "output_smooth_gpu.tif")
        print("Done.")
        
    """

    def __init__(self, talus_threshold: float = 4.0, erosion_rate: float = 0.1):
        # 1. Fix the environment BEFORE doing anything CUDA-related
        self._ensure_nvrtc_loaded()
        
        self.talus = talus_threshold
        self.rate = erosion_rate
        
        # 2. Compile Kernel (This is where it would normally fail)
        self.kernel = self._compile_kernel()
        self.profile = None

    def _ensure_nvrtc_loaded(self):
        """
        Private Helper: Dynamically finds and force-loads the NVRTC library 
        into the process memory to prevent CuPy 'OSError' on conda envs.
        """
        try:
            # We don't want to reload it if it's already there
            # (Checks if a known NVRTC symbol exists in the current process)
            try:
                ctypes.CDLL("libnvrtc.so.12")
                return # Already loaded by system, we are good.
            except OSError:
                pass # Not found, let's go hunting.

            # Dynamic path finding: Look inside the current Conda/Python environment
            # This avoids hardcoding '/home/pete/...'
            base_prefix = sys.prefix
            
            # Pattern to find the library deeply nested in site-packages
            # Works for Python 3.10, 3.11, 3.12 etc.
            search_pattern = os.path.join(
                base_prefix, 
                "lib", "python*", "site-packages", "nvidia", 
                "cuda_nvrtc", "lib", "libnvrtc.so.12"
            )
            
            matches = glob.glob(search_pattern)
            
            if not matches:
                # Fallback: Try looking in the general lib folder (sometimes simpler envs put it there)
                search_pattern = os.path.join(base_prefix, "lib", "libnvrtc.so.12")
                matches = glob.glob(search_pattern)

            if matches:
                target_lib = matches[0]
                # Force load with RTLD_GLOBAL so CuPy can "see" it
                ctypes.CDLL(target_lib, mode=ctypes.RTLD_GLOBAL)
                # print(f"DEBUG: Auto-patched NVRTC from {target_lib}")
            else:
                print("WARNING: Could not auto-locate libnvrtc.so.12. Kernel compilation may fail.")

        except Exception as e:
            print(f"WARNING: NVRTC patch failed: {e}")

    def _compile_kernel(self) -> cp.RawKernel:
        # ... (Same kernel code as before) ...
        kernel_code = r'''
        extern "C" __global__
        void thermal_erode(const float* src, float* dst, 
                           int width, int height, 
                           float talus_threshold, float erosion_rate) {
            
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            int idx = y * width + x;

            if (x >= width || y >= height) return;

            float h = src[idx];
            
            // Boundary Conditions
            if (h <= 0.0f || x == 0 || x == width - 1 || y == 0 || y == height - 1) {
                dst[idx] = h;
                return;
            }

            int n_idxs[4] = {
                (y - 1) * width + x, (y + 1) * width + x,
                y * width + (x - 1), y * width + (x + 1)
            };

            float flow_total = 0.0f;
            for (int i = 0; i < 4; i++) {
                float diff = h - src[n_idxs[i]];
                if (diff > talus_threshold) {
                    flow_total += (diff - talus_threshold);
                }
            }
            dst[idx] = h - (flow_total * erosion_rate);
        }
        '''
        return cp.RawKernel(kernel_code, 'thermal_erode')

    # ... (Rest of the class: load_dem, process, save_dem) ...
    def load_dem(self, filepath: Union[str, Path]) -> np.ndarray:
        with rasterio.open(filepath) as src:
            self.profile = src.profile
            return src.read(1).astype(np.float32)

    def process(self, dem: np.ndarray, iterations: int = 100) -> np.ndarray:
        height, width = dem.shape
        buf_a = cp.asarray(dem)
        buf_b = cp.empty_like(buf_a)
        
        threads = (16, 16)
        blocks_x = (width + threads[0] - 1) // threads[0]
        blocks_y = (height + threads[1] - 1) // threads[1]
        grid = (blocks_x, blocks_y)
        
        # print(f"Processing on GPU: {width}x{height} for {iterations} iters...")
        
        for i in range(iterations):
            src, dst = (buf_a, buf_b) if i % 2 == 0 else (buf_b, buf_a)
            self.kernel(grid, threads, (
                src, dst, width, height, 
                cp.float32(self.talus), cp.float32(self.rate)
            ))

        final_gpu = buf_b if iterations % 2 != 0 else buf_a
        return cp.asnumpy(final_gpu)

    def save_dem(self, dem: np.ndarray, filepath: Union[str, Path]) -> None:
        output_data = np.round(dem).astype(np.int16)
        save_profile = self.profile.copy()
        save_profile.update({'dtype': 'int16', 'nodata': None, 'compress': 'lzw'})
        with rasterio.open(filepath, 'w', **save_profile) as dst:
            dst.write(output_data, 1)










### OLD STUFF ##########
# @dataclass
# class NoiseRule:
#     """
#     Defines a specific layer of fractal noise to be added to the terrain.
#     """
#     seed: int
#     multiplier: float           # Vertical scale (e.g., 800.0)
#     scale: float                # Horizontal scale (e.g., 50.0 for mountains, 400.0 for hills)
#     min_elevation: float = -99999.0  # Apply only above this altitude in Base DEM
#     mask_layer: str = None      # Name of mask in mask_library (e.g., 'north')
#     octaves: int = 6
#     persistence: float = 0.5
#     lacunarity: float = 2.0



# def apply_noise_rules(base_dem: np.ndarray, 
#                       rules: List[NoiseRule], 
#                       mask_library: Dict[str, np.ndarray] = {}) -> np.ndarray:
#     """
#     The Master Compositor. Stacks multiple noise layers onto the base DEM.
    
#     LOGIC UPDATE:
#     - If a rule has BOTH min_elevation AND a mask_layer, they are combined with OR (|).
#       (e.g., "Apply if > 500m OR inside North mask").
#     - If a rule has only one, it applies that one.
#     - If a rule has neither, it applies everywhere.
    
#     Args:
#         base_dem: The starting topography (NumPy array).
#         rules: List of NoiseRule objects defining the layers.
#         mask_library: Dictionary mapping names ('north', 'sea') to Numpy Boolean Arrays.
        
#     Returns:
#         np.ndarray: The final composite terrain.
#     """
#     logger.info(f"Applying {len(rules)} noise rules to base terrain...")
    
#     # 1. Copy Base DEM (Float32 for precision)
#     final_dem = base_dem.astype(np.float32).copy()
#     height, width = final_dem.shape
    
#     for i, rule in enumerate(rules):
#         logger.info(f"  Rule {i+1}: Scale={rule.scale}, Mult={rule.multiplier}, Seed={rule.seed}")
        
#         # 2. Generate Raw Noise
#         raw_noise = _generate_noise_in_memory(
#             shape=(height, width),
#             scale=rule.scale,
#             seed=rule.seed,
#             octaves=rule.octaves,
#             persistence=rule.persistence,
#             lacunarity=rule.lacunarity
#         )
        
#         # 3. Scale Noise
#         scaled_noise = raw_noise * rule.multiplier
        
#         # 4. Build Application Mask (The OR Logic)
        
#         # Check which conditions are active
#         has_elevation = rule.min_elevation > -9999.0
#         has_mask = (rule.mask_layer is not None)
        
#         # Resolve the mask layer if it exists
#         spatial_mask = None
#         if has_mask:
#             if rule.mask_layer in mask_library:
#                 spatial_mask = mask_library[rule.mask_layer]
#             else:
#                 logger.warning(f"    Mask '{rule.mask_layer}' not found! Ignoring mask condition.")
#                 has_mask = False

#         # Combine Conditions
#         if has_elevation and has_mask:
#             # Elevation OR Mask
#             elevation_mask = (base_dem >= rule.min_elevation)
#             app_mask = elevation_mask | spatial_mask
#             logger.info(f"    Filtered by Elevation > {rule.min_elevation} OR Mask '{rule.mask_layer}'")
            
#         elif has_elevation:
#             # Only Elevation
#             app_mask = (base_dem >= rule.min_elevation)
#             logger.info(f"    Filtered by Elevation > {rule.min_elevation}")
            
#         elif has_mask:
#             # Only Mask
#             app_mask = spatial_mask
#             logger.info(f"    Filtered by Mask '{rule.mask_layer}'")
            
#         else:
#             # No filters -> Global
#             app_mask = np.ones((height, width), dtype=bool)
#             logger.info(f"    No filters (Global application)")

#         # 5. Apply
#         final_dem[app_mask] += scaled_noise[app_mask]
        
#     logger.info("Terrain composition complete.")
#     return final_dem

