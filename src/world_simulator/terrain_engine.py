import numpy as np
import rasterio
from rasterio import features, warp
from scipy import ndimage
from skimage.transform import resize
import logging
from typing import List, Tuple, Dict, Any, Union
import gc
import os

# Import Layer Definitions
from .layers import (
    TerrainLayer, 
    RiverWarpLayer, 
    TectonicLayer, 
    ErosionLayer, 
    LakeIntegrationLayer, 
    SmartCoastalTaper
)

logger = logging.getLogger(__name__)

class CoordinateEngine:
    """
    The central processor for Hybrid Synthesis.
    
    Manages the 16k Coordinate Grid and the Master Heightmap.
    Implements 'Proposition 1': Low-Res Proxy optimization for River Warping
    to maintain memory stability.
    """

    def __init__(self, shape: Tuple[int, int], profile: Dict, initial_elevation: np.ndarray = None):
        """
        Args:
            shape: (Height, Width) of the target map (e.g., 10000, 16000).
            profile: Rasterio profile (transform, crs, etc.).
            initial_elevation: Optional starting heightmap (float32).
        """
        self.height, self.width = shape
        self.profile = profile
        
        logger.info(f"Initializing Coordinate Engine ({self.width}x{self.height})...")

        # 1. Initialize Elevation
        if initial_elevation is not None:
            if initial_elevation.shape != shape:
                raise ValueError(f"Initial elevation shape {initial_elevation.shape} matches not {shape}")
            self.elevation = initial_elevation.astype(np.float32)
        else:
            self.elevation = np.zeros(shape, dtype=np.float32)

        # 2. Initialize Coordinate Grids (The "Fabric" of space)
        # These are high-res (16k) and consume ~1.2GB RAM together.
        logger.info("Allocating high-res coordinate grids...")
        y_inds, x_inds = np.indices(shape, dtype=np.float32)
        self.coords_y = y_inds
        self.coords_x = x_inds
        
        # Explicit garbage collection to ensure we aren't holding init artifacts
        gc.collect()
        logger.info("Coordinate fabric initialized.")

    def build(self, recipe: List[TerrainLayer]) -> np.ndarray:
        """
        Executes the generation recipe step-by-step.
        """
        for i, layer in enumerate(recipe):
            logger.info(f"Applying Layer {i+1}/{len(recipe)}: {layer.name} ({type(layer).__name__})")
            
            if isinstance(layer, RiverWarpLayer):
                self._apply_river_warp(layer)
            elif isinstance(layer, TectonicLayer):
                self._apply_tectonic(layer)
            elif isinstance(layer, ErosionLayer):
                self._apply_erosion(layer)
            elif isinstance(layer, LakeIntegrationLayer):
                self._apply_lakes(layer)
            elif isinstance(layer, SmartCoastalTaper):
                self._apply_coastal_taper(layer)
            else:
                logger.warning(f"Unknown layer type: {type(layer)}")
                
            # Force GC after every layer to handle large array churn
            gc.collect()

        return self.elevation

    def _apply_river_warp(self, layer: RiverWarpLayer):
        """
        Orchestrates the river warping process using the 'Low-Res Proxy' optimization.
        Refactored for memory efficiency and readability.
        """
        logger.info(f"  > Starting River Warp (Strength={layer.warp_strength}, Influence={layer.influence_km}km)")

        # 1. Setup Proxy Grid (~500m resolution)
        proxy_shape, proxy_transform, scale_factor = self._setup_proxy_grid(target_res_m=500.0)
        logger.info(f"    Proxy Grid: {proxy_shape[1]}x{proxy_shape[0]} (Scale: {scale_factor:.2f})")

        # 2. Rasterize Data (Low Res)
        # Returns the river width values and a boolean-style mask (0=River, 1=Land)
        proxy_width, proxy_river_mask = self._rasterize_river_proxy(layer, proxy_shape, proxy_transform)
        
        # 3. Compute Warp Vectors (Low Res)
        # Calculates the shift (dy, dx) for every pixel in the proxy grid
        proxy_pixel_size = abs(proxy_transform[0])
        shift_y_low, shift_x_low = self._compute_proxy_warp_vectors(
            proxy_river_mask, proxy_width, layer, proxy_pixel_size
        )

        # Cleanup inputs immediately to free RAM for the upscaling step
        del proxy_width, proxy_river_mask
        gc.collect()

        # 4. Upscale and Apply to Main Fabric
        self._upscale_and_apply_warp(shift_y_low, shift_x_low)
        
        logger.info("  > Warp Complete.")

    def _setup_proxy_grid(self, target_res_m: float) -> Tuple[Tuple[int, int], Any, float]:
        """Calculates dimensions and transform for the downsampled proxy grid."""
        pixel_size = abs(self.profile['transform'][0])
        scale_factor = pixel_size / target_res_m
        scale_factor = max(0.1, min(1.0, scale_factor)) # Clamp bounds

        proxy_h = int(self.height * scale_factor)
        proxy_w = int(self.width * scale_factor)

        # Scale the affine transform
        proxy_transform = self.profile['transform'] * self.profile['transform'].scale(
            (self.width / proxy_w), (self.height / proxy_h)
        )
        return (proxy_h, proxy_w), proxy_transform, scale_factor

    def _rasterize_river_proxy(self, layer: RiverWarpLayer, shape: Tuple[int, int], transform: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Rasterizes river widths and creates a presence mask on the proxy grid."""
        logger.info("    Rasterizing river vectors to proxy...")
        
        # Generator for (geometry, width_value) pairs
        width_shapes = ((geom, val) for geom, val in zip(layer.rivers_gdf.geometry, layer.rivers_gdf[layer.width_column]))
        
        proxy_width = features.rasterize(
            shapes=width_shapes,
            out_shape=shape,
            transform=transform,
            all_touched=True, # Critical for capturing small rivers
            fill=0,
            dtype=np.float32
        )
        
        # Create presence mask (0 where river, 1 where land) for Distance Transform
        proxy_river_mask = (proxy_width == 0).astype(np.uint8)
        
        return proxy_width, proxy_river_mask

    def _compute_proxy_warp_vectors(self, mask: np.ndarray, width_raster: np.ndarray, 
                                   layer: RiverWarpLayer, pixel_size: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the vector field (dy, dx) in the low-res domain."""
        logger.info("    Calculating Vector Field (Distance Transform)...")
        
        height, width = mask.shape

        # A. Nearest Neighbor Search (The heavy math step)
        dist_px, indices = ndimage.distance_transform_edt(
            mask, 
            return_distances=True, 
            return_indices=True
        )

        # B. Prepare Coordinate Grids for Vector Math
        grid_y, grid_x = np.indices((height, width), dtype=np.float32)
        
        # Vector = Target (River) - Current Position
        vec_y = indices[0] - grid_y
        vec_x = indices[1] - grid_x
        
        # C. Normalize Vectors
        vec_len = np.sqrt(vec_x**2 + vec_y**2)
        vec_len[vec_len == 0] = 1.0 # Prevent div/0
        
        norm_y = vec_y / vec_len
        norm_x = vec_x / vec_len

        # D. Calculate Magnitude
        influence_px = (layer.influence_km * 1000.0) / pixel_size
        
        # Look up width of the nearest river pixel
        nearest_width = width_raster[indices[0], indices[1]]
        
        # Gravity Well Falloff (Smoothstep)
        falloff = np.clip(1.0 - (dist_px / influence_px), 0.0, 1.0)
        falloff = falloff * falloff * (3 - 2 * falloff) 
        
        shift_mag = nearest_width * layer.warp_strength * falloff
        
        shift_y = norm_y * shift_mag
        shift_x = norm_x * shift_mag
        
        return shift_y, shift_x

    def _upscale_and_apply_warp(self, shift_y_low: np.ndarray, shift_x_low: np.ndarray):
        """Upscales the low-res vectors and adds them to the high-res coordinate fabric."""
        logger.info("    Upscaling and applying warp field...")

        # Resize Y-Shift
        high_res_shift_y = resize(
            shift_y_low, 
            (self.height, self.width), 
            order=1, # Linear interpolation is sufficient and smooth
            mode='edge', 
            anti_aliasing=False, 
            preserve_range=True
        ).astype(np.float32)
        
        # Apply Y immediately and delete to save RAM
        self.coords_y += high_res_shift_y
        del high_res_shift_y
        gc.collect()

        # Resize X-Shift
        high_res_shift_x = resize(
            shift_x_low, 
            (self.height, self.width), 
            order=1, 
            mode='edge', 
            anti_aliasing=False, 
            preserve_range=True
        ).astype(np.float32)

        # Apply X and delete
        self.coords_x += high_res_shift_x
        del high_res_shift_x
        gc.collect()

    def _apply_tectonic(self, layer: TectonicLayer):
        """
        Applies tectonic noise using a memory-safe 'Batched' approach.
        Refactored into smaller components for readability.
        """
        logger.info(f"  > Starting Tectonic Gen: {layer.name} (Mode={layer.ridge_mode})")
        
        # 1. Configure SIMD Engine
        simplex = self._setup_simd_object(layer)
        
        # 2. Batch Calculation
        batch_height = 1024 
        total_batches = (self.height + batch_height - 1) // batch_height
        
        logger.info(f"    Processing in {total_batches} batches of {batch_height} rows...")

        for b in range(total_batches):
            y_start = b * batch_height
            y_end = min((b + 1) * batch_height, self.height)
            current_h = y_end - y_start
            
            # 3. Generate Noise for this strip (Handles all SIMD alignment internally)
            batch_noise = self._generate_tectonic_chunk(simplex, layer, y_start, y_end)

            # 4. Masking
            mask = self._resolve_mask_for_batch(layer.mask_config, y_start, y_end, current_h)
            
            # 5. Apply
            # Elevation += Noise * Amplitude * Mask
            self.elevation[y_start:y_end, :] += batch_noise * layer.amplitude * mask
            
            # 6. Cleanup
            del batch_noise, mask
            gc.collect()
        
        logger.info("  > Tectonic Layer Applied.")

    def _setup_simd_object(self, layer: TectonicLayer):
        """Configures the PyFastNoiseSIMD object based on layer settings."""
        import pyfastnoisesimd as fns
        import os
        
        threads = min(6, os.cpu_count())
        simplex = fns.Noise(numWorkers=threads)
        simplex.seed = layer.seed
        simplex.frequency = layer.frequency
        simplex.noiseType = fns.NoiseType.Cellular
        
        if layer.ridge_mode == "Distance2Sub":
            simplex.cell.returnType = fns.CellularReturnType.Distance2Sub
        else:
            simplex.cell.returnType = fns.CellularReturnType.Distance
            
        return simplex

    def _generate_tectonic_chunk(self, simplex, layer: TectonicLayer, 
                                 y_start: int, y_end: int) -> np.ndarray:
        """
        Generates a specific strip of noise, handling SIMD memory alignment 
        and ridge inversion logic.
        """
        import pyfastnoisesimd as fns
        
        current_h = y_end - y_start
        num_points = current_h * self.width
        
        # A. Allocate Aligned Memory
        # fns.empty_coords allocates slightly more than num_points for alignment
        batch_coords = fns.empty_coords(num_points)
        
        # B. Fill Valid Coordinates
        # Flatten the 2D slice into the 1D buffer, stopping exactly at num_points
        batch_coords[0, :num_points] = self.coords_x[y_start:y_end, :].ravel()
        batch_coords[1, :num_points] = self.coords_y[y_start:y_end, :].ravel()
        batch_coords[2, :] = 0 
        
        # C. Execute SIMD Generation
        # Output array is also padded
        batch_noise = simplex.genFromCoords(batch_coords)
        
        # D. Crop and Reshape
        # Discard the padding bytes and restore 2D shape
        result = batch_noise[:num_points].reshape((current_h, self.width))
        
        # E. Apply Ridge Logic
        # Distance mode usually needs inversion to make centers = hills
        if layer.ridge_mode == "Distance":
             result = 1.0 - result
             
        return result

    def _resolve_mask_for_batch(self, config: Dict, y_start: int, y_end: int, h: int) -> Union[float, np.ndarray]:
        """
        Generates a blending mask for a specific vertical strip of the map.
        Returns 1.0 if no mask is configured.
        """
        if config is None:
            return 1.0
            
        mask_type = config.get('type')
        
        if mask_type == 'altitude':
            # 1. Slice existing elevation
            elev_slice = self.elevation[y_start:y_end, :]
            min_h = config.get('min_height', 0.0)
            fade_h = config.get('fade_height', 100.0)
            
            # Linear Ramp: (Elev - Min) / Fade
            mask = (elev_slice - min_h) / fade_h
            return np.clip(mask, 0.0, 1.0)
            
        elif mask_type == 'vector_shape':
            # This is harder to batch because rasterio wants to rasterize the whole shape.
            # However, since we pre-calculated masks in the Recipe or Utils, 
            # we should assume we might pass a Global Mask ARRAY to the layer 
            # instead of a geometry, OR we rasterize the geometry globally once 
            # (if it fits in RAM) and slice it here.
            
            # Optimization: If the mask was simple geometry, we could check bounding box.
            # For now, let's assume the user passes a geometry and we rasterize 
            # a "Strip Mask".
            
            geometry = config.get('geometry')
            fade_km = config.get('fade_km', 0.0)
            
            # Create a window transform for this strip
            # We shift the Origin Y down by y_start * pixel_height
            window_transform = self.profile['transform'] * self.profile['transform'].translation(0, y_start)
            
            # Rasterize just this strip! 
            # Rasterio handles "Out of window" geometry automatically.
            shapes = ((geom, 1) for geom in geometry.geometry)
            
            strip_mask = features.rasterize(
                shapes=shapes,
                out_shape=(h, self.width),
                transform=window_transform,
                fill=0,
                dtype=np.float32
            )
            
            if fade_km > 0:
                # Distance Transform on a Strip is DANGEROUS (Neighborhood Op!).
                # A fade calculated on a strip won't know about the polygon just outside the strip.
                
                # SOLUTION: For vector masks, it is safer to pre-calculate the 
                # global mask in __init__ or build() if possible, OR accept that 
                # vector masking is expensive. 
                
                # Given strict RAM limits, we might skip the precise fade 
                # or implement a "Coarse Distance Field" globally (like River Warp).
                pass 
                
            return strip_mask

        return 1.0

    def _apply_erosion(self, layer: ErosionLayer):
        logger.info(f"  > [STUB] Erosion Layer '{layer.name}' passed.")
        pass

    def _apply_lakes(self, layer: LakeIntegrationLayer):
        logger.info(f"  > [STUB] Lake Layer '{layer.name}' passed.")
        pass

    def _apply_coastal_taper(self, layer: SmartCoastalTaper):
        logger.info("  > [STUB] Coastal Taper passed.")
        pass