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


###############################################################################
### NEW TerrainEngine
###############################################################################


import numpy as np
from typing import Optional, Tuple, Union, Literal

import pyfastnoisesimd as fns
import gc
import os
import logging

class TerrainEngine:
    """
    The core processor for Phase 4: Terrain Texturing & Noise.
    
    Architecture: Just-In-Time (JIT) Integrated Engine.
    
    This engine does NOT store the full coordinate grid in memory (which would consume ~1.2GB).
    Instead, it accepts pre-computed 'Warp Fields' (from Phase 3) and applies them 
    on-the-fly to generate tectonic noise.
    
    It implements the 'Bedrock & Skin' philosophy by allowing separate passes for:
      - Structural Geometry (Bedrock): Low freq, high amp, specific shapes (Ridges/Domes).
      - Surface Texture (Skin): High freq, low amp, multifractal detail.
    """

    def __init__(self, shape: Tuple[int, int], threads: int = -1):
        """
        Initialize the engine context.

        Args:
            shape: (Height, Width) of the target map (e.g., 10000, 16000).
            threads: Number of CPU threads for SIMD generation. 
                     -1 attempts to auto-detect based on available cores vs RAM safety.
        """
        self.logger = logging.getLogger(__name__)
        self.height, self.width = shape
        
        # CPU Threading Logic
        # We cap at 8 to prevent diminishing returns or system lockup on 16GB machines
        if threads == -1:
            cpu_count = os.cpu_count() or 4
            self._threads = max(1, min(8, cpu_count - 1))
        else:
            self._threads = threads
            
        self.logger.info(f"TerrainEngine Initialized: {self.width}x{self.height} | SIMD Threads: {self._threads}")
        self.logger.info("  > Architecture: JIT Integrated (Low RAM Mode)")



    def apply_tectonic_pass(
        self,
        current_dem: np.ndarray,
        gcm_mask: np.ndarray,
        warp_x: np.ndarray,
        warp_y: np.ndarray,
        influence: np.ndarray,
        target_zone: int,
        mode: str,
        frequency: float,
        amplitude: float,
        seed: int,
        warp_strength: float = 1.0,   # NEW
        perturb_amp: float = 0.0,     # NEW
        edge_dither: float = 0.0,     # NEW
        force_positive: bool = False, # NEW
        invert: bool = False,         # NEW
        sub_selection: Optional[str] = None,
        attenuation: bool = False
    ) -> np.ndarray:
        """
        Executes a single Tectonic or Texturing pass for a specific Geological Zone.

        This method orchestrates the 'Chunking' strategy:
        1. Slices the map into horizontal bands (e.g., 1024 rows) to fit in L3 Cache/RAM.
        2. Sets up the PyFastNoiseSIMD generator for the specific 'mode'.
        3. For each chunk:
           a. Reconstructs the local coordinate grid.
           b. Applies the `warp_x` / `warp_y` offsets to the coordinates.
           c. Generates noise on the distorted grid.
           d. Masks the noise using `gcm_mask` (Target Zone only).
           e. Suppresses noise using `influence` (River channels remain flat).
           f. Blends the result into the accumulator.

        Args:
            current_dem: The accumulator heightmap (Float32).
            gcm_mask: The Geological Control Map (Zone IDs).
            warp_x: Phase 3 Warp Vector Field (X-component).
            warp_y: Phase 3 Warp Vector Field (Y-component).
            influence: Phase 3 Warp Influence (1.0 = River Center, 0.0 = Land).
                       Used to mask noise so rivers aren't blocked by generated ridges.
            target_zone: The specific Zone ID (1-5) to apply this pass to.
            mode: The noise algorithm to use. 
                  Options: "Distance2Sub" (Alpine), "Cellular" (Shield), 
                           "Simplex" (Plateau), "RidgedMulti" (Skin), "Billow".
            frequency: Noise scale (lower = larger features).
            amplitude: Vertical scaling factor (meters).
            seed: Random seed for variety.
            sub_selection: Optional filter for Zone 4 logic.
                           - "North": Applies only where Y > River_Axis.
                           - "South": Applies only where Y < River_Axis.
            attenuation: If True, applies 'Mud Masking' logic (Zone 4b),
                         fading the noise amplitude based on proximity to the river.

        Returns:
            np.ndarray: The modified DEM (current_dem + generated_noise).
        """
        height, width = current_dem.shape
        
        # 1. Configure Main Generator
        generator = self._setup_simd_generator(mode, seed, frequency)
        
        # 2. Configure Perturbation Generator (if needed)
        perturb_generator = None
        if perturb_amp > 0:
            # Use High-Freq Simplex for organic distortion
            perturb_generator = fns.Noise(numWorkers=self._threads)
            perturb_generator.seed = seed + 999
            perturb_generator.frequency = frequency * 2.0 
            perturb_generator.noiseType = fns.NoiseType.Simplex
        
        batch_height = 1024
        total_batches = (height + batch_height - 1) // batch_height
        
        self.logger.info(f"  > Processing {mode} (Z{target_zone}) | Warp: {warp_strength} | Dither: {edge_dither}")

        for b in range(total_batches):
            y_start = b * batch_height
            y_end = min((b + 1) * batch_height, height)
            
            # Extract inputs
            wx_chunk = warp_x[y_start:y_end, :]
            wy_chunk = warp_y[y_start:y_end, :]
            inf_chunk = influence[y_start:y_end, :]
            
            # A. Resolve Mask (New Dithered Logic)
            # Note: We pass the FULL gcm_mask, not a chunk
            mask = self._resolve_zone_mask(
                gcm_mask, y_start, y_end, target_zone, wy_chunk, 
                sub_selection, edge_dither
            )
            
            if not np.any(mask):
                continue
            
            # B. Generate Noise (New Warped Logic)
            raw_noise = self._generate_chunk(
                generator, perturb_generator, 
                y_start, y_end, wx_chunk, wy_chunk,
                warp_strength, perturb_amp
            )
            
            # C. Post-Process: Inversion
            # Ridge/Distance noise is often 0..1. Inverting makes 1..0.
            if invert:
                # If mode is Simplex (-1..1), Invert is just Negate? 
                # Usually 'Invert' implies flip range. 
                # For Distance (0..1), result = 1.0 - val.
                # For Simplex (-1..1), result = -val.
                if mode in ["Distance", "Distance2Sub", "Cellular"]:
                    raw_noise = 1.0 - raw_noise
                else:
                    raw_noise = -raw_noise

            # D. Post-Process: Normalization (Force Positive)
            if force_positive:
                if mode in ["Simplex", "Directional", "FBM", "Billow", "RidgedMulti"]:
                    # Approximate -1..1 -> 0..1
                    raw_noise = (raw_noise + 1.0) / 2.0
                # Distance modes are usually already 0..1 (or inverted to 0..1)
            
            # E. Amplitude & Attenuation
            noise_scaled = raw_noise * amplitude
            
            if attenuation:
                noise_scaled = self._apply_attenuation(noise_scaled, inf_chunk)
            
            # F. Global River Safety
            noise_scaled *= (1.0 - inf_chunk)
            
            # G. Apply to Accumulator
            dem_slice = current_dem[y_start:y_end, :]
            dem_slice[mask] += noise_scaled[mask]
            current_dem[y_start:y_end, :][mask] = dem_slice[mask]
            
            del raw_noise, mask, noise_scaled
            
        gc.collect()
        return current_dem



    def _setup_simd_generator(self, mode: str, seed: int, frequency: float):
        """
        Internal factory that configures the low-level PyFastNoiseSIMD object.
        
        Handles the mapping between our string 'mode' (e.g. "Distance2Sub") 
        and the underlying SIMD library enums/settings.
        """
        # Auto-detect threads if not specified, keeping some headroom for the OS
        if not hasattr(self, '_threads') or self._threads == -2:
            self._threads = max(1, min(8, os.cpu_count() - 2))

        # Initialize
        simd_obj = fns.Noise(numWorkers=self._threads)
        simd_obj.seed = seed
        simd_obj.frequency = frequency
        
        # Mapping Logic
        if mode == "Distance2Sub":
            # Zone 1 Bedrock: Sharp Ridges
            simd_obj.noiseType = fns.NoiseType.Cellular
            simd_obj.cell.returnType = fns.CellularReturnType.Distance2Sub
            
        elif mode == "Cellular":
            # Zone 2 Bedrock: Rounded Domes / Cobblestone
            simd_obj.noiseType = fns.NoiseType.Cellular
            simd_obj.cell.returnType = fns.CellularReturnType.Distance
            
        elif mode == "Simplex":
            # Zone 3 Bedrock: Smooth rolling hills
            simd_obj.noiseType = fns.NoiseType.Simplex
            
        elif mode == "RidgedMulti":
            # Zone 1 Skin: Fractured rock detail
            simd_obj.noiseType = fns.NoiseType.SimplexFractal
            simd_obj.fractal.fractalType = fns.FractalType.RigidMulti
            simd_obj.fractal.octaves = 4
            simd_obj.fractal.lacunarity = 2.0
            simd_obj.fractal.gain = 0.5
            
        elif mode == "Billow":
            # Zone 2 Skin: "Melted" / Scoured look
            simd_obj.noiseType = fns.NoiseType.SimplexFractal
            simd_obj.fractal.fractalType = fns.FractalType.Billow
            simd_obj.fractal.octaves = 3
            
        elif mode == "FBM":
            # Zone 3 Skin: Uniform soil grain
            simd_obj.noiseType = fns.NoiseType.SimplexFractal
            simd_obj.fractal.fractalType = fns.FractalType.FBM
            simd_obj.fractal.octaves = 3
            
        elif mode == "WhiteNoise":
            # Zone 4 Skin: Sand/Silt grain
            simd_obj.noiseType = fns.NoiseType.WhiteNoise
            
        elif mode == "Directional":
            # Zone 4a Bedrock: Levees
            # Note: The "Directionality" comes from the Warp Field acting on this noise,
            # not the noise function itself. We use Simplex as the base.
            simd_obj.noiseType = fns.NoiseType.Simplex
            
        else:
            raise ValueError(f"Unknown Noise Mode: {mode}")

        return simd_obj

        
    def _generate_chunk(
        self, 
        generator, 
        perturb_generator, 
        y_start: int, 
        y_end: int, 
        warp_x_chunk: np.ndarray, 
        warp_y_chunk: np.ndarray,
        warp_strength: float,
        perturb_amp: float
    ) -> np.ndarray:
        """
        Internal worker that generates raw noise for a specific horizontal strip.

        Crucial Step:
        It must generate coordinates `(y, x)` for the chunk, ADD the `warp` offsets,
        and THEN query the noise generator. This is what forces the mountains 
        to bend around the rivers.
        """
        chunk_h = y_end - y_start
        chunk_w = warp_x_chunk.shape[1]
        num_points = chunk_h * chunk_w
        
        # 1. Allocate Aligned Coords
        coords = fns.empty_coords(num_points)
        
        # 2. Build Base Coordinate Grid
        grid_y, grid_x = np.indices((chunk_h, chunk_w), dtype=np.float32)
        grid_y += y_start 
        
        # 3. Apply Internal Perturbation (The "Double-Dip")
        # Breaks up the perfect geometric shapes of Voronoi/Cellular noise
        if perturb_amp > 0 and perturb_generator is not None:
            # We use the SAME coords buffer to generate the perturb noise to save RAM
            # Fill buffer with linear coords
            coords[0, :num_points] = grid_x.ravel()
            coords[1, :num_points] = grid_y.ravel()
            coords[2, :] = 0
            
            # Generate perturbation field
            p_noise = perturb_generator.genFromCoords(coords)
            
            # Apply to grid (Reshaped)
            # We add the noise to the coordinates themselves
            p_reshaped = p_noise[:num_points].reshape((chunk_h, chunk_w))
            grid_x += (p_reshaped * perturb_amp)
            grid_y += (p_reshaped * perturb_amp)
            
            # Cleanup to keep memory tight
            del p_noise, p_reshaped

        # 4. Apply River Warp (The "Gravity" Effect)
        # Scale the unit vectors by the Strength factor
        grid_x += (warp_x_chunk * warp_strength)
        grid_y += (warp_y_chunk * warp_strength)
        
        # 5. Final Generation
        # Refill the coords buffer with the distorted grid
        coords[0, :num_points] = grid_x.ravel()
        coords[1, :num_points] = grid_y.ravel()
        coords[2, :] = 0
        
        # Generate main noise
        noise_flat = generator.genFromCoords(coords)
        
        return noise_flat[:num_points].reshape((chunk_h, chunk_w))



    
    def _resolve_zone_mask(
        self, 
        gcm_full: np.ndarray, 
        y_start: int, 
        y_end: int,
        target_zone: int, 
        warp_y_chunk: np.ndarray,
        sub_selection: Optional[str] = None,
        edge_dither: float = 0.0
    ) -> np.ndarray:
        """
        Internal helper to create the boolean application mask for the chunk.

        Handles:
        1. Matching `gcm_chunk == target_zone`.
        2. The "Fractal Edge" logic (if we decide to dither the edges here).
        3. The `sub_selection` logic for Zone 4 (North vs South split).
        """
        chunk_h, chunk_w = warp_y_chunk.shape
        full_h, full_w = gcm_full.shape
        
        # 1. Coordinate Grid for Lookup
        # If dithering, we need to displace these lookups
        if edge_dither > 0:
            # Generate simple deterministic noise for the dither
            # Using numpy random is fast enough for mask resolution (no SIMD needed)
            # We use a fixed seed based on y_start to ensure deterministic runs
            rng = np.random.default_rng(seed=y_start)
            
            # Generate offsets (-1 to 1) * strength
            dither_y = rng.uniform(-1, 1, (chunk_h, chunk_w)) * edge_dither
            dither_x = rng.uniform(-1, 1, (chunk_h, chunk_w)) * edge_dither
            
            # Calculate sample coordinates
            # y_start + local_y + dither
            sample_y = (np.arange(chunk_h)[:, None] + y_start + dither_y).astype(np.int32)
            sample_x = (np.arange(chunk_w)[None, :] + dither_x).astype(np.int32)
            
            # Clamp to map bounds
            np.clip(sample_y, 0, full_h - 1, out=sample_y)
            np.clip(sample_x, 0, full_w - 1, out=sample_x)
            
            # Perform Dithered Lookup
            gcm_sample = gcm_full[sample_y, sample_x]
            
        else:
            # Standard Direct Slice
            gcm_sample = gcm_full[y_start:y_end, :]

        # 2. Base Mask
        mask = (gcm_sample == target_zone)
        
        # 3. Sub-Selection (River Split)
        if sub_selection and np.any(mask):
            if sub_selection == "North":
                mask = mask & (warp_y_chunk > 0)
            elif sub_selection == "South":
                mask = mask & (warp_y_chunk < 0)
                
        return mask

    def _apply_attenuation(
        self, 
        noise_chunk: np.ndarray, 
        influence_chunk: np.ndarray
    ) -> np.ndarray:
        """
        Internal helper for Zone 4b (Subsiding Plate).
        
        Calculates the "Mud Mask" fade. As `influence` increases (closer to river),
        the noise amplitude should decay, simulating rock being buried under sediment.
        """

        # Influence is 1.0 at river center, 0.0 at edge/land.
        # We want NO noise at river (1.0) and FULL noise at land (0.0).
        
        # Calculate Multiplier: (1.0 - Inf)
        # Result: 0.0 at river -> 1.0 at land
        multiplier = 1.0 - influence_chunk
        
        # Apply Smoothstep to make the transition nicer? 
        # Optional, but linear is usually fine for "burial" simulation.
        # multiplier = multiplier * multiplier * (3 - 2 * multiplier)
        
        return noise_chunk * multiplier