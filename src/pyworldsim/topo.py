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
from scipy.ndimage import zoom, uniform_filter, label, distance_transform_edt, generate_binary_structure, binary_dilation
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
    
    def __init__(self, taper_distance_px: float):
        """
        Args:
            taper_distance_px: Distance in pixels over which the land fades in.
        """
        self.taper_dist = taper_distance_px
        
        # State populated by load_inputs
        self.sea_mask = None
        self.profile = None
        self.valid_data_mask = None # Tracks where original DEM had data
        self.taper_map = None

    def load_inputs(self, dem_path: str, sea_vector: Union[str, gpd.GeoDataFrame]):
        """
        Loads the DEM to get dimensions/profile and rasterizes the sea vector 
        to create the internal boolean mask.

        Args:
            dem_path: Path to the reference DEM.
            sea_vector: Path to vector file OR loaded GeoDataFrame of the sea.
        
        Returns:
            np.ndarray: The loaded DEM data (float32).
        """
        logger.info(f"Loading inputs from {dem_path}...")
        
        # 1. Load Reference DEM and Profile
        with rasterio.open(dem_path) as src:
            self.profile = src.profile
            dem_data = src.read(1)
            
            # Handle NoData: Create a mask of valid pixels to restore later
            nodata_val = src.nodata
            if nodata_val is not None:
                # Use a small epsilon for float safety, or direct comparison for others
                if np.issubdtype(dem_data.dtype, np.floating):
                    self.valid_data_mask = ~np.isclose(dem_data, nodata_val)
                else:
                    self.valid_data_mask = (dem_data != nodata_val)
            else:
                self.valid_data_mask = np.ones(dem_data.shape, dtype=bool)

        # 2. Handle Vector Input
        if isinstance(sea_vector, (str, Path)):
            logger.info("Loading Sea Vector from disk...")
            sea_gdf = gpd.read_file(sea_vector)
        else:
            sea_gdf = sea_vector

        # 3. Internal Vector-to-Mask (Rasterization)
        logger.info("Rasterizing Sea Vector to match DEM profile...")
        
        if sea_gdf is None or sea_gdf.empty:
            logger.warning("Empty vector provided. Assuming NO SEA.")
            self.sea_mask = np.zeros(dem_data.shape, dtype=bool)
        else:
            # Burn '1' where polygons exist
            shapes = ((geom, 1) for geom in sea_gdf.geometry)
            
            mask_uint8 = features.rasterize(
                shapes=shapes,
                out_shape=(self.profile['height'], self.profile['width']),
                transform=self.profile['transform'],
                fill=0,
                dtype=rasterio.uint8
            )
            self.sea_mask = mask_uint8.astype(bool)

        return dem_data.astype(np.float32)



    
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

    
    def save_result(self, dem: np.ndarray, output_path: str):
        """
        Saves the processed DEM as Int16, handling NoData correctly.
        
        Logic:
        1. Resets off-map pixels (NoData) to -32768.
        2. Rounds valid pixels to nearest integer.
        3. Saves with LZW compression.
        """
        if self.profile is None:
            raise ValueError("No profile context. Run load_inputs() first.")
            
        logger.info(f"Saving result to {output_path} as Int16...")

        # 1. Prepare Output Array (Default to Int16 Min value)
        int16_nodata = -32768
        output_data = np.full(dem.shape, int16_nodata, dtype=np.int16)

        # 2. Populate only VALID pixels
        # We use the valid_data_mask captured during load_inputs
        # to ensure we don't accidentally turn a void (-9999) into a tapered zero.
        if self.valid_data_mask is not None:
            # Round float result to nearest int before casting
            valid_pixels = dem[self.valid_data_mask]
            output_data[self.valid_data_mask] = np.round(valid_pixels).astype(np.int16)
        else:
            output_data = np.round(dem).astype(np.int16)

        # 3. Update Profile
        save_profile = self.profile.copy()
        save_profile.update({
            'dtype': 'int16',
            'nodata': int16_nodata,
            'compress': 'lzw',
            'driver': 'GTiff'
        })

        # 4. Write
        with rasterio.open(output_path, 'w', **save_profile) as dst:
            dst.write(output_data, 1)

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
        kernel_code = r'''
        extern "C" __global__
        void thermal_erode(const float* src, float* dst, 
                           int width, int height, 
                           float talus_threshold, float erosion_rate,
                           float nodata_val) {  // <--- NEW PARAMETER
            
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            int idx = y * width + x;

            if (x >= width || y >= height) return;

            float h = src[idx];
            
            // 1. Ignore NoData pixels entirely (Preserve them)
            // We use a small epsilon for float comparison safety
            if (abs(h - nodata_val) < 1e-6) {
                dst[idx] = h;
                return;
            }
            
            // 2. Boundary Conditions (Map Edges & Sea Level)
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
                float neighbor_h = src[n_idxs[i]];

                // 3. Check Neighbor Validity
                // If neighbor is NoData, treat it like a wall (infinite height) 
                // so material does NOT flow into the void.
                if (abs(neighbor_h - nodata_val) < 1e-6) {
                    continue; 
                }

                float diff = h - neighbor_h;
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
        
        # <--- NEW: Detect NoData value (default to -9999 if not in profile)
        nodata_val = self.profile.get('nodata', -9999.0)
        if nodata_val is None: nodata_val = -9999.0 # Fallback
        
        buf_a = cp.asarray(dem)
        buf_b = cp.empty_like(buf_a)
        
        threads = (16, 16)
        blocks_x = (width + threads[0] - 1) // threads[0]
        blocks_y = (height + threads[1] - 1) // threads[1]
        grid = (blocks_x, blocks_y)
        
        for i in range(iterations):
            src, dst = (buf_a, buf_b) if i % 2 == 0 else (buf_b, buf_a)
            self.kernel(grid, threads, (
                src, dst, width, height, 
                cp.float32(self.talus), 
                cp.float32(self.rate),
                cp.float32(nodata_val) # <--- NEW ARGUMENT
            ))

        final_gpu = buf_b if iterations % 2 != 0 else buf_a
        return cp.asnumpy(final_gpu)

    def save_dem(self, dem: np.ndarray, filepath: Union[str, Path]) -> None:
        # <--- NEW: Handle NoData preservation during cast
        nodata_val = self.profile.get('nodata', -9999.0)
        
        # Create a mask of valid data before rounding
        # (Assuming NoData is distinct enough, e.g. < -1000)
        valid_mask = (dem != nodata_val)
        
        output_data = np.full(dem.shape, -32768, dtype=np.int16) # Initialize with int16 min
        
        # Only round and cast valid pixels
        output_data[valid_mask] = np.round(dem[valid_mask]).astype(np.int16)

        save_profile = self.profile.copy()
        save_profile.update({
            'dtype': 'int16',
            'nodata': -32768, # <--- Set standard Int16 NoData
            'compress': 'lzw'
        })
        with rasterio.open(filepath, 'w', **save_profile) as dst:
            dst.write(output_data, 1)


def destep_terrain(dem_array, nodata_value, radius=64, iterations=3, scale_factor=2):
    """
    Applies a 'Normalized Convolution' iterative box blur to a quantized DEM.
    This ensures smooth slopes without 'smearing' Nodata values into valid terrain.
    
    Parameters:
        dem_array (numpy array): Input elevation data.
        nodata_value (float/int): The value representing void data (e.g. -32768).
        radius (int): Smoothing radius.
        iterations (int): Number of passes.
        scale_factor (int): Multiplier for vertical precision.
    
    Returns:
        numpy array: Smoothed terrain as int16 (with original nodata restored).
    """
    print(f"--- Starting Nodata-Aware De-stepping (R={radius}, I={iterations}) ---")
    
    # 1. Identify Valid Data
    # Create a boolean mask: 1.0 where data exists, 0.0 where it's void
    # handle NaN if present, otherwise check equality
    if np.isnan(nodata_value):
        valid_mask = ~np.isnan(dem_array)
    else:
        valid_mask = (dem_array != nodata_value)
    
    # 2. Prepare Float Arrays
    # Convert data to float and scale
    working_data = dem_array.astype(np.float32) * scale_factor
    
    # CRITICAL: Set Nodata pixels to exactly 0.0 in the data array
    # This prevents the massive negative numbers from influencing the sum.
    working_data[~valid_mask] = 0.0
    
    # Create a float weight mask (0.0 or 1.0)
    weight_mask = valid_mask.astype(np.float32)
    
    window_size = (radius * 2) + 1
    
    # 3. Iterative Normalized Convolution
    for i in range(iterations):
        print(f"  > Blur Pass {i+1}/{iterations}...")
        
        # A. Blur the Data (Accumulates sum of neighbors, treating nodata as 0)
        # We use mode='constant' (cval=0) so image edges behave like Nodata
        blurred_data = uniform_filter(working_data, size=window_size, mode='constant', cval=0.0)
        
        # B. Blur the Mask (Accumulates count of valid neighbors)
        blurred_weights = uniform_filter(weight_mask, size=window_size, mode='constant', cval=0.0)
        
        # C. Normalize (Average = Sum / Count)
        # Avoid division by zero where weights are 0 (pure nodata zones)
        non_zero_weights = blurred_weights > 1e-6
        
        # Update working data:
        # Where we have valid neighbors, calculate the weighted average.
        # Where we have NO valid neighbors, keep as 0.0.
        working_data[non_zero_weights] = blurred_data[non_zero_weights] / blurred_weights[non_zero_weights]
        
        # Note: We do NOT re-apply the mask to 0.0 here yet, allowing valid data 
        # to 'bleed' slightly into the void to smooth the edge, 
        # but we effectively clamp the result later.
        
        # For the next pass, the weight_mask remains the same (original validity)
        # or we could blur the weights too? 
        # Standard approach: Keep weight_mask static to strictly smooth *existing* valid geometry
        # without hallucinating new terrain into the void.
        
    # 4. Restore Nodata & Cast
    print("Restoring Nodata and quantizing...")
    
    # Round to integer
    final_output = np.round(working_data).astype(np.int16)
    
    # Hard reset of original nodata positions
    # (Optional: remove this if you WANT the terrain to expand into the void slightly)
    # Given the user request "bands at intersections", strict masking is safer.
    final_output[~valid_mask] = int(nodata_value)
    
    return final_output


import numpy as np

def apply_zone_elevation_change(dem_array, mask_array, zone_id, elevation_change_m, scale_factor=2, nodata_value=-32768):
    """
    Applies a uniform elevation change to a specific geological zone.
    
    Parameters:
        dem_array (numpy array): The source elevation raster (int16).
        mask_array (numpy array): The geological zone mask (uint8).
        zone_id (int): The specific zone value to target (e.g., 3).
        elevation_change_m (float): Amount to uplift (+) or depress (-) in meters.
        scale_factor (int): Multiplier used in the DEM (e.g. 2 means 1m = 2 units).
        nodata_value (int): Value to ignore/preserve.
        
    Returns:
        numpy array: Modified DEM.
    """
    print(f"--- Applying Zonal Uplift (Zone {zone_id}: {elevation_change_m}m) ---")
    
    # 1. Calculate Shift in Internal Units
    # elevation_change_m can be positive (uplift) or negative (subsidence)
    shift_units = int(elevation_change_m * scale_factor)
    
    if shift_units == 0:
        print("  > No change calculated. Returning original.")
        return dem_array
    
    # 2. Create Working Copy
    # We work on a copy to preserve the input array state if needed elsewhere
    modified_dem = dem_array.copy()
    
    # 3. Create Selection Mask
    # Target pixels that belong to the zone AND are not nodata
    # (Handling nodata check ensures we don't uplift the void)
    target_pixels = (mask_array == zone_id) & (modified_dem != nodata_value)
    
    count = np.sum(target_pixels)
    print(f"  > Targeted {count} pixels.")
    
    # 4. Apply Uplift
    # We use numpy in-place addition for speed
    # We assume int16 is sufficient (max height ~32000m). 
    # If you expect overflow, we could clamp, but for +200m it is safe.
    modified_dem[target_pixels] += shift_units
    
    return modified_dem



import numpy as np
from scipy.ndimage import label, distance_transform_edt, center_of_mass

def get_south_bank_mask(gcm_crop, river_crop, target_zone_id):
    """
    1. ISOLATE SOUTH BANK (VOTING STRATEGY)
    Splits the zone by the river and identifies the Southern component 
    by voting across multiple vertical slices.
    """
    print("  > Step 1: Isolating South Bank (Column Voting)...")
    
    # A. The Cut
    zone_pixels = (gcm_crop == target_zone_id)
    severed_zone = zone_pixels & (river_crop == 0)
    
    # B. Label Blobs
    # 3x3 structure to ensure diagonal connections are caught
    structure = np.ones((3,3), dtype=int)
    labeled_array, num_features = label(severed_zone, structure=structure)
    
    if num_features < 2:
        print(f"    ! CRITICAL WARNING: River did not sever Zone {target_zone_id}. Found only {num_features} blob.")
        # Fallback: Return everything to avoid crash, but warn user
        return zone_pixels

    # C. The Vote: Who is South?
    # We ignore background (0). We tally votes for "Being the Southernmost"
    vote_tally = {i: 0 for i in range(1, num_features + 1)}
    
    # Define columns to sample: 20 slices across the width of the crop
    height, width = labeled_array.shape
    # Filter to columns that actually have data to avoid wasting checks on empty padding
    valid_cols = np.where(np.any(labeled_array > 0, axis=0))[0]
    
    if len(valid_cols) == 0:
        return zone_pixels # Should not happen if features > 0
        
    # Pick 20 equidistant columns from the valid range
    sample_columns = np.linspace(valid_cols[0], valid_cols[-1], num=20, dtype=int)
    
    print(f"    [Analysis] Sampling {len(sample_columns)} columns for North/South check...")
    
    for col in sample_columns:
        # Get the column slice
        col_data = labeled_array[:, col]
        
        # Find unique labels in this column (excluding 0)
        present_labels = np.unique(col_data)
        present_labels = present_labels[present_labels != 0]
        
        # If less than 2 blobs coexist in this column, we can't compare relative height
        if len(present_labels) < 2:
            continue
            
        # For each label present, calculate its average ROW index in this column
        # Higher Row Index = South
        label_positions = {}
        for lbl in present_labels:
            # np.where returns indices where condition is true
            rows = np.where(col_data == lbl)[0]
            avg_row = np.mean(rows)
            label_positions[lbl] = avg_row
            
        # Identify the winner for this column (The one with the Highest Row Index)
        # max(dict, key=dict.get) gets the key with the max value
        winner = max(label_positions, key=label_positions.get)
        
        vote_tally[winner] += 1
        
    # D. Determine Winner
    # Find label with max votes
    south_idx = max(vote_tally, key=vote_tally.get)
    total_votes = sum(vote_tally.values())
    
    print(f"    [Vote Result] Winner: Blob #{south_idx} with {vote_tally[south_idx]} / {total_votes} votes.")
    print(f"    [Tally] {vote_tally}")

    final_mask = (labeled_array == south_idx)

    # --- VISUALIZATION 1: THE CUT & VOTE ---
    plt.figure(figsize=(12, 5))
    
    # Plot 1: The Labeled Blobs
    plt.subplot(1, 3, 1)
    plt.title(f"Blobs (Winner: #{south_idx})")
    plt.imshow(labeled_array, cmap='nipy_spectral', interpolation='nearest')
    # Draw sample lines
    for col in sample_columns:
        plt.axvline(x=col, color='white', alpha=0.3, linewidth=0.5)
        
    # Plot 2: The Selected South Bank
    plt.subplot(1, 3, 2)
    plt.title("Selected South Bank")
    plt.imshow(final_mask, cmap='gray', interpolation='nearest')
    
    # Plot 3: Context
    plt.subplot(1, 3, 3)
    plt.title("Context (River Cut)")
    plt.imshow(river_crop, cmap='Blues', alpha=0.5)
    plt.imshow(zone_pixels, cmap='Reds', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    # ---------------------------------------

    return final_mask





def compute_normalized_ramp(south_bank_mask, river_crop):
    """
    2. CALCULATE RAMP (SHORELINE PROPAGATION)
    Creates a 0.0 to 1.0 gradient.
    Explicitly sets distance to 0 at the shoreline pixels to prevent cliffs.
    """
    print("  > Step 2: Calculating Normalized Incline (Shoreline Propagation)...")
    
    # 1. Define the Sources
    # ---------------------------------------------------------
    # River Mask: Target the River pixels (Value > 0)
    # in previous steps we established river_crop=0 is LAND.
    river_mask = (river_crop != 0)
    
    # Boundary Mask ("The Hills"): 
    # Defined as: Anything that is NOT South Bank AND NOT River.
    # Logic: The zone ends where the South Bank mask ends, provided it's not the river edge.
    hill_mask = (~south_bank_mask) & (~river_mask)
    
    # 2. Identify the Contact Lines
    # ---------------------------------------------------------
    # Shoreline: South Bank pixels that touch the River.
    # Dilate River to find overlap.
    shoreline_pixels = binary_dilation(river_mask) & south_bank_mask
    
    # Hill-line: South Bank pixels that touch the Hills.
    # Dilate Hills to find overlap.
    hill_line_pixels = binary_dilation(hill_mask) & south_bank_mask

    # SAFETY: If hill line is empty (e.g. crop cut off the hills?), 
    # we default to the furthest points in the south bank mask?
    # For now, let's assume the mask is valid.
    
    # 3. Calculate Geodesic Distances
    # ---------------------------------------------------------
    # We want distance FROM the contact lines INTO the South Bank.
    # edt calculates distance to the nearest Zero. 
    # So we invert the contact masks (Contact=0, Field=1).
    
    # Distance from Shoreline
    d_shore = distance_transform_edt(~shoreline_pixels)
    
    # Distance from Hill-line
    d_hill = distance_transform_edt(~hill_line_pixels)

    # 4. Normalize
    # ---------------------------------------------------------
    total_dist = d_shore + d_hill
    
    # Avoid div/0
    total_dist[total_dist == 0] = 1.0 
    
    ramp = d_shore / total_dist
    
    # Mask strict output
    ramp[~south_bank_mask] = 0.0
    
    # --- VISUALIZATION 3: THE RAMP ---
    plt.figure(figsize=(18, 15))
    plt.title("Computed Gradient (Shoreline Propagation)")
    vis_ramp = ramp.copy()
    vis_ramp[vis_ramp == 0] = np.nan
    plt.imshow(vis_ramp, cmap='magma')
    plt.colorbar(label="Incline Factor")
    plt.show()
    # ---------------------------------

    return ramp



def add_ramp_to_dem(dem_crop, ramp, south_bank_mask, max_incline_m, scale_factor=2):
    """
    3. INJECT
    Adds the scaled ramp height to the existing DEM.
    """
    print(f"  > Step 3: Injecting +{max_incline_m}m Incline...")
    
    # Convert meters to units
    max_incline_units = int(max_incline_m * scale_factor)
    
    # Calculate elevation delta
    # ramp is 0.0-1.0 float
    delta_elevation = (ramp * max_incline_units).astype(np.int16)
    
    # Add to DEM
    # We copy to avoid modifying source in place unexpectedly
    modified_crop = dem_crop.copy()
    
    # In-place addition only on the mask
    modified_crop[south_bank_mask] += delta_elevation[south_bank_mask]
    
    return modified_crop


    
def apply_south_bank_ramp(dem_array, gcm_array, throughline_mask, target_zone_id, incline_m, scale_factor=2):
    """
    4. WRAPPER
    Orchestrates the cropping and execution.
    """
    print(f"--- Applying South Bank Incline (Zone {target_zone_id}) ---")
    
    # A. Get Bounds (Reused from previous step logic)
    rows, cols = np.where(gcm_array == target_zone_id)
    if len(rows) == 0: return dem_array
    
    pad = 50
    r1, r2 = max(0, np.min(rows)-pad), min(gcm_array.shape[0], np.max(rows)+pad)
    c1, c2 = max(0, np.min(cols)-pad), min(gcm_array.shape[1], np.max(cols)+pad)
    
    print(f"  > Crop Window: {r1}:{r2}, {c1}:{c2}")
    
    # B. Extract Crops
    dem_crop = dem_array[r1:r2, c1:c2]
    gcm_crop = gcm_array[r1:r2, c1:c2]
    river_crop = throughline_mask[r1:r2, c1:c2]
    
    # C. Step 1: Get Mask
    sb_mask = get_south_bank_mask(gcm_crop, river_crop, target_zone_id)
    
    # D. Step 2: Get Ramp
    ramp = compute_normalized_ramp(sb_mask, river_crop)
    
    # E. Step 3: Add to DEM
    dem_crop_mod = add_ramp_to_dem(dem_crop, ramp, sb_mask, incline_m, scale_factor)
    
    # F. Paste Back
    final_dem = dem_array.copy()
    final_dem[r1:r2, c1:c2] = dem_crop_mod
    
    return final_dem


import numpy as np
from scipy.ndimage import distance_transform_edt, zoom
import gc
from numba import njit, prange

# -----------------------------------------------------------------------------
# 1. NUMBA KERNEL (Unchanged)
# -----------------------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _warp_kernel(indices_y, indices_x, width_mod, strength_mod, base_radius_px, shape):
    """
    Numba kernel to compute Warp Fields in a single pass.
    Avoids creating huge temporary arrays for 'vec_x', 'vec_y', 'dist'.
    """
    h, w = shape
    
    # Pre-allocate outputs in float32 for calculation stability
    # We cast to float16 later or return these if RAM allows
    out_x = np.zeros((h, w), dtype=np.float32)
    out_y = np.zeros((h, w), dtype=np.float32)
    out_inf = np.zeros((h, w), dtype=np.float32)
    
    # Parallel Loop over rows
    for r in prange(h):
        for c in range(w):
            # 1. Get Target Coordinates (Where the river is)
            # EDT indices are (Row, Col)
            target_r = indices_y[r, c]
            target_c = indices_x[r, c]
            
            # 2. Vector: Target - Current
            dy = float(target_r - r)
            dx = float(target_c - c)
            
            # 3. Distance
            dist = np.sqrt(dx*dx + dy*dy)
            
            # 4. Normalize Vector
            if dist > 0:
                norm_x = dx / dist
                norm_y = dy / dist
            else:
                norm_x = 0.0
                norm_y = 0.0
                
            # 5. Influence (Geology Aware)
            # Retrieve modifiers for this pixel
            w_mod = width_mod[r, c]
            s_mod = strength_mod[r, c]
            
            # Effective Radius
            radius = base_radius_px * w_mod
            if radius < 0.001: radius = 0.001
            
            # Normalized Distance (0.0 at river -> 1.0 at edge)
            d_norm = dist / radius
            
            if d_norm >= 1.0:
                # Outside influence
                inf = 0.0
            else:
                # Inside: Linear falloff -> Cubic Smoothing
                raw_inf = 1.0 - d_norm
                inf = raw_inf * raw_inf * (3.0 - 2.0 * raw_inf)
            
            out_inf[r, c] = inf
            
            # 6. Final Warp Vector
            # Vector * Influence * Strength
            mag = inf * s_mod
            out_x[r, c] = norm_x * mag
            out_y[r, c] = norm_y * mag
            
    return out_x, out_y, out_inf

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def _downsample_inputs(river_mask, geology_mask, factor):
    """
    Downsamples masks using slicing (if 0.5) or nearest-neighbor zoom.
    Returns the small arrays.
    """
    print("  > Downsampling masks...")
    if factor == 0.5:
        # Fast Slicing
        r_small = river_mask[::2, ::2]
        g_small = geology_mask[::2, ::2]
    else:
        # General Zoom
        r_small = zoom(river_mask, factor, order=0)
        g_small = zoom(geology_mask, factor, order=0)
        
    return r_small, g_small

def _compute_edt_indices(river_mask_small):
    """
    Calculates Euclidean Distance Transform indices.
    """
    print("  > Calculating EDT...")
    # 1=River in input -> 0=Target for EDT
    edt_input = (river_mask_small != 1)
    
    # Return only indices to save RAM
    _, indices = distance_transform_edt(edt_input, return_distances=True, return_indices=True)
    return indices

def _map_geology_modifiers(geology_small, zone_params):
    """
    Creates Width and Strength modifier maps based on geology zones.
    """
    print("  > Mapping Geology...")
    h, w = geology_small.shape
    width_mod = np.ones((h, w), dtype=np.float32)
    strength_mod = np.ones((h, w), dtype=np.float32)
    
    for zone_id, params in zone_params.items():
        mask = (geology_small == zone_id)
        if np.any(mask):
            width_mod[mask] = params['width_mod']
            strength_mod[mask] = params['strength_mod']
            
    return width_mod, strength_mod

def _upscale_results(out_x, out_y, out_inf, original_shape, small_shape):
    """
    Upscales the low-res results back to full resolution.
    """
    print("  > Upscaling to full resolution...")
    orig_h, orig_w = original_shape
    small_h, small_w = small_shape
    
    # Calculate precise scale factors
    scale_y = orig_h / small_h
    scale_x = orig_w / small_w
    
    # Zoom and cast to float16 to save RAM
    high_x = zoom(out_x, (scale_y, scale_x), order=1).astype(np.float16)
    high_y = zoom(out_y, (scale_y, scale_x), order=1).astype(np.float16)
    high_inf = zoom(out_inf, (scale_y, scale_x), order=1).astype(np.float16)
    
    return high_x, high_y, high_inf

# -----------------------------------------------------------------------------
# 3. MAIN WRAPPER
# -----------------------------------------------------------------------------
def compute_warp_artifacts(river_mask, geology_mask, zone_params, 
                           pixel_size_m=500.0, base_radius_km=3.0, 
                           downsample_factor=0.5):
    """
    Generates Hydraulic Warp artifacts with extreme optimization.
    Orchestrates the pipeline using helper functions.
    """
    print(f"--- Computing Warp Artifacts (Scale: {downsample_factor}) ---")
    
    # 1. Downsample
    river_small, geo_small = _downsample_inputs(river_mask, geology_mask, downsample_factor)
    
    # Capture actual dimensions for kernels and upscaling
    small_h, small_w = river_small.shape
    orig_h, orig_w = river_mask.shape
    print(f"    Original: {orig_w}x{orig_h} -> Processed: {small_w}x{small_h}")
    
    # 2. EDT Calculation
    indices = _compute_edt_indices(river_small)
    
    # 3. Geology Mapping
    width_mod, strength_mod = _map_geology_modifiers(geo_small, zone_params)
    
    # Cleanup Inputs
    del river_small, geo_small
    gc.collect()
    
    # 4. Numba Execution
    print("  > Running Numba Warp Kernel (Parallel)...")
    eff_pixel_size = pixel_size_m / downsample_factor
    base_radius_px_small = (base_radius_km * 1000.0) / eff_pixel_size
    
    out_x_s, out_y_s, out_inf_s = _warp_kernel(
        indices[0], indices[1],
        width_mod, strength_mod,
        base_radius_px_small,
        (small_h, small_w)
    )
    
    # Cleanup Intermediate
    del indices, width_mod, strength_mod
    gc.collect()
    
    # 5. Upscale
    final_x, final_y, final_inf = _upscale_results(
        out_x_s, out_y_s, out_inf_s, 
        (orig_h, orig_w), 
        (small_h, small_w)
    )
    
    # Cleanup Low Res
    del out_x_s, out_y_s, out_inf_s
    gc.collect()
    
    return final_x, final_y, final_inf



import numpy as np
from scipy.ndimage import distance_transform_edt
from numba import njit, prange
import gc

@njit(parallel=True)
def _carve_kernel(dem_array, edt_array, geology_mask, 
                  scour_multipliers, zone_ids, 
                  base_cut_m, width_factor, 
                  sea_level_units, scale_factor):
    """
    Fused Kernel: Scour Mapping -> Depth Calc -> Unit Scaling -> Subtraction -> Clamping.
    """
    h, w = dem_array.shape
    
    # Create output array (clone input)
    out_dem = dem_array.copy()
    
    # Pre-compute scour map for fast lookup if IDs are small integers?
    # Since zone_ids are sparse (1,2,3,4,5,9), a direct array lookup is tricky unless we map them.
    # For speed, we can iterate simple if-checks or use a small lookup array if max ID is small.
    # Assuming max Zone ID is < 20.
    
    # Parallel Loop
    for r in prange(h):
        for c in range(w):
            dist = edt_array[r, c]
            
            # Optimization: Only process river pixels
            if dist > 0:
                # 1. Get Geology Scour Multiplier
                # Linear search is fast for <10 items, or use direct lookup if possible
                g_val = geology_mask[r, c]
                scour_mult = 1.0 # Default
                
                # Fast lookup matching
                for i in range(len(zone_ids)):
                    if zone_ids[i] == g_val:
                        scour_mult = scour_multipliers[i]
                        break
                
                # 2. Calculate Depth in METERS
                # Depth = (Base + (Dist * Width_Factor)) * Scour
                depth_m = (base_cut_m + (dist * width_factor)) * scour_mult
                
                # 3. Convert to UNITS (1m = 2 units)
                depth_units = int(depth_m * scale_factor)
                
                # 4. Apply to DEM
                current_h = out_dem[r, c]
                new_h = current_h - depth_units
                
                # 5. Safety Clamp (Sea Level)
                if new_h < sea_level_units:
                    new_h = sea_level_units
                    
                out_dem[r, c] = new_h
                
    return out_dem

def apply_hydraulic_carve(dem_array, river_mask, geology_mask, scour_params, 
                          base_cut_m=5.0, width_factor=2.0, sea_level_m=0.0, scale_factor=2.0):
    """
    Carves the river network into the DEM (Artifact B) using Numba.
    
    Memory Optimized: Prevents 4GB RAM overflow by avoiding intermediate float arrays.
    """
    print("--- Applying Hydraulic Carve (Numba Optimized) ---")
    
    # 1. Calculate River Depth Profile (EDT)
    # We must do this in Scipy, but we optimize the output immediately.
    print("  > Calculating River Depth Profile (EDT)...")
    
    # river_mask: 1=River. EDT calculates distance to 0 (Land).
    # This gives distance from bank to center.
    dist_to_bank = distance_transform_edt(river_mask).astype(np.float32)
    
    # 2. Prepare Numba Inputs
    # Deconstruct dict to arrays for Numba
    zone_ids = np.array(list(scour_params.keys()), dtype=np.int16)
    scour_mults = np.array(list(scour_params.values()), dtype=np.float32)
    
    sea_level_units = int(sea_level_m * scale_factor)
    
    # 3. Execute Kernel
    print("  > Executing Fused Carve Kernel...")
    carved_dem = _carve_kernel(
        dem_array, 
        dist_to_bank, 
        geology_mask,
        scour_mults, 
        zone_ids,
        base_cut_m, 
        width_factor,
        sea_level_units,
        scale_factor
    )
    
    # Stats
    # Diff stats
    diff = dem_array.astype(np.float32) - carved_dem.astype(np.float32)
    max_cut_units = np.max(diff)
    print(f"  > Max Cut Applied: {max_cut_units} units ({max_cut_units/scale_factor:.1f}m)")
    
    # Cleanup
    del dist_to_bank
    gc.collect()
    
    return carved_dem