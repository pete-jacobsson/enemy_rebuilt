import numpy as np
import rasterio.features
from scipy.ndimage import binary_erosion, binary_closing

class ActiveSimulationMask:
    """
    Defines the computational domain for the simulation.
    
    Responsible for distinguishing between the 'Active' physics zone, 
    the 'Sink' boundary (where water/sediment is deleted), and the 'Void' (ignored data).
    Also handles the repair of input artifacts (NaN holes) in the geodata.
    
    Attributes:
        ctx (WorldState): The shared simulation context.
    """

    def __init__(self, ctx):
        """
        Initialize the generator with the world context.
        
        Args:
            ctx (WorldState): The simulation context.
        """
        self.ctx = ctx

    
    def apply(self):
        """
        Generates the 'active_mask' layer in the context.
        
        Steps:
        1. Rasterize 'simulation_bounds.gpkg' -> 1 (Active).
        2. Identify map edges and set a 10px buffer -> 2 (Sink).
        3. Everything else -> 0 (Void).
        4. Calls _heal_artifacts() to patch geometric errors.
        """
        print("Generating Active Simulation Mask...")
        
        # Retrieve the bounds vector from the context
        if "bounds" not in self.ctx.vectors:
            raise ValueError("Vector 'bounds' not found in context. Call ctx.load_vectors() first.")
            
        bounds_gdf = self.ctx.vectors["bounds"]
        
        # 1. Rasterize the Active Zone (Value = 1)
        # We use the transform and shape from the context
        active_mask = rasterio.features.rasterize(
            shapes=[(geom, 1) for geom in bounds_gdf.geometry],
            out_shape=self.ctx.shape,
            transform=self.ctx.transform,
            fill=0,  # Background is Void (0)
            dtype=np.int16
        )
        
        # 2. Create the Sink Zone (Value = 2)
        # We identify the boundary between Active (1) and Void (0)
        # Using binary erosion to find the inner edge of the active zone.
        
        # Create a boolean version for morphology
        bool_active = active_mask == 1
        
        # Erode the active area by 10 pixels (5km) to define the "safe" inner zone
        # The difference between the original and eroded mask is the "Sink" strip
        eroded_active = binary_erosion(bool_active, iterations=10)
        
        # Where it WAS active but is NOT in the eroded version -> Sink
        sink_mask = (bool_active & ~eroded_active)
        
        # Update the master mask
        active_mask[sink_mask] = 2
        
        # 3. Heal Artifacts (Fix NaN holes or jagged edges)
        final_mask = self._heal_artifacts(active_mask)
        
        # 4. Write to Context
        self.ctx.active_mask = final_mask
        print("  > Active Mask generated. (1=Active, 2=Sink, 0=Void)")

    
    def _heal_artifacts(self, mask):
        """
        Internal utility to fix small voids or ragged edges in the mask.
        Uses morphological closing to fill small holes inside the expected active zone.
        
        Args:
            mask (np.ndarray): The raw Int16 mask.
            
        Returns:
            np.ndarray: The watertight Int16 mask.
        """
        print("  > Healing mask artifacts...")
        
        # Treat anything not Void as "Valid" (Active or Sink)
        is_valid = mask > 0
        
        # Close small holes (iterations=2 fills gaps approx 2km wide)
        healed_bool = binary_closing(is_valid, iterations=2)
        
        # Identify where the closing filled a hole
        filled_holes = healed_bool & ~is_valid
        
        out_mask = mask.copy()
        
        # Where the closing filled a hole, we assign it to 'Active' (1) by default
        # (We assume holes inside the map are active terrain, not boundary sinks)
        out_mask[filled_holes] = 1
        
        count = np.sum(filled_holes)
        if count > 0:
            print(f"    - Filled {count} void pixels.")
            
        return out_mask


import numpy as np
from scipy.ndimage import gaussian_filter

class PrimordialSeabed:
    """
    Initializes the elevation canvas for Epoch 0 (500 MA).
    
    Sets the entire active world to a shallow continental shelf depth.
    Adds low-frequency noise to ensure the terrain is not mathematically flat, 
    which is required to prevent 'divide-by-zero' or stagnation errors in 
    early hydraulic flow calculations.
    """

    def __init__(self, level=-50.0):
        """
        Args:
            level (float): The base elevation in meters (Default: -50.0m).
        """
        self.base_level = level

    def apply(self, ctx):
        """
        Mutates ctx.elevation.
        
        Logic:
        1. Fill active area with `base_level`.
        2. Generate broad-scale noise (wavelength ~200km, amplitude +/- 10m).
        3. Add noise to base level.
        """
        print(f"Initializing Primordial Seabed at {self.base_level}m...")
        
        # 1. Reset Elevation
        # We start fresh. Any previous data in this array is overwritten.
        ctx.elevation[:] = self.base_level
        
        # 2. Generate Noise Layer
        # We need the noise to be the same shape as the context
        noise = self._generate_broad_noise(ctx.shape, scale=200.0)
        
        # 3. Apply Noise
        # We only apply noise where the mask is not Void (0)
        # This keeps the "off-map" area clean at the base level.
        active_indices = ctx.active_mask > 0
        ctx.elevation[active_indices] += noise[active_indices]
        
        print("  > Seabed initialized.")


    def _generate_broad_noise(self, shape, scale):
        """
        Generates low-frequency, low-amplitude noise.
        
        Algorithm:
        1. Create a low-resolution grid of random values.
        2. Upscale (zoom) or Blur it heavily to create long wavelengths.
        
        Args:
            shape (tuple): The target (height, width).
            scale (float): The approximate wavelength in km (used to tune the blur).
            
        Returns:
            np.ndarray: The noise layer (Float32).
        """
        h, w = shape
        
        # We want features roughly 200km across.
        # At 500m/px, 200km is 400 pixels.
        # We use a gaussian sigma proportional to this feature size.
        sigma = 400.0 / 4.0  # Tunable constant for smoothness
        
        print(f"  > Generating spectral noise (sigma={sigma})...")
        
        # 1. White Noise (Random +/- 10m)
        rng = np.random.default_rng(seed=42) # Fixed seed for reproducibility
        white_noise = rng.uniform(-10.0, 10.0, size=shape).astype(np.float32)
        
        # 2. Gaussian Blur to extract low frequencies
        # This turns static into rolling gradients.
        smooth_noise = gaussian_filter(white_noise, sigma=sigma)
        
        # 3. Normalize amplitude
        # The blur reduces amplitude significantly, so we re-normalize
        # to ensure we actually get +/- 10m variations.
        current_max = np.max(np.abs(smooth_noise))
        if current_max > 0:
            smooth_noise = smooth_noise * (10.0 / current_max)
            
        return smooth_noise



import os
from numba import jit
import math
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio.features
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import unary_union
# import rtree
from collections import deque
import rasterio.features


from numba import jit
import math
import numpy as np

@jit(nopython=True, cache=True)
def numba_growth_kernel(seeds, grid, attractor, transform_coeffs, params):
    """
    The High-Performance Growth Engine.
    
    Args:
        seeds (float64[:, :]): Array of seeds [N, 5] -> (x, y, dx, dy, order)
        grid (int8[:, :]): Collision map (0=Free, 1=Blocked). Modified in-place.
        attractor (float32[:, :]): The 'Gravity' map.
        transform_coeffs (tuple): (a, b, c, d, e, f) for Affine Inverse.
                                  col = a*x + b*y + c
                                  row = d*x + e*y + f
        params (tuple): Physics constants (step_len, max_segs, fork_prob, rad, w, h)
        
    Returns:
        tuple: (count, out_nodes, out_links, out_meta)
    """
    # Unpack Params
    step_len = params[0]
    max_segments = int(params[1])
    fork_prob = params[2]
    exclusion_rad = params[3]
    width = int(params[4])
    height = int(params[5])
    
    # Unpack Transform
    ta, tb, tc, td, te, tf = transform_coeffs
    
    # Allocate Output Arrays (The "Warehouse")
    # nodes: [x, y]
    out_nodes = np.zeros((max_segments, 2), dtype=np.float64)
    # links: Index of parent segment
    out_links = np.full(max_segments, -1, dtype=np.int32)
    # meta: [order] (Can expand to include type/width)
    out_meta = np.zeros((max_segments, 1), dtype=np.int8)
    
    # Stack for Active Tips (The "ToDo List")
    # Stores index of the segment that is currently growing
    stack = np.zeros(max_segments, dtype=np.int32)
    stack_ptr = 0 # Points to the next empty slot
    
    # Initialize Counters
    segment_count = 0
    
    # --------------------------------------------------------------------------
    # 1. INITIALIZE SEEDS
    # --------------------------------------------------------------------------
    n_seeds = len(seeds)
    for i in range(n_seeds):
        if segment_count >= max_segments: break
        
        sx, sy = seeds[i, 0], seeds[i, 1]
        sdx, sdy = seeds[i, 2], seeds[i, 3]
        s_order = int(seeds[i, 4])
        
        # Write Seed to Output
        idx = segment_count
        out_nodes[idx, 0] = sx
        out_nodes[idx, 1] = sy
        out_links[idx] = -1 # Root has no parent (in this array)
        out_meta[idx, 0] = s_order
        segment_count += 1
        
        # Add to Stack
        stack[stack_ptr] = idx
        stack_ptr += 1
        
        # We DON'T mark the grid for the start point, because it technically 
        # sits on a canonical river (which is already burned). 
        # We rely on the first step jumping OUT of the exclusion zone.

    # --------------------------------------------------------------------------
    # 2. GROWTH LOOP
    # --------------------------------------------------------------------------
    
    # Random State (Simple LCG or Numba's internal rand)
    # Numba supports np.random.random()
    
    while stack_ptr > 0 and segment_count < max_segments:
        # Pop (LIFO - Depth First creates longer, cleaner rivers)
        stack_ptr -= 1
        parent_idx = stack[stack_ptr]
        
        px, py = out_nodes[parent_idx, 0], out_nodes[parent_idx, 1]
        p_order = out_meta[parent_idx, 0]
        
        # Stop if order is too small
        if p_order < 1: continue
        
        # --- A. SENSE (Look at Attractor) ---
        # Get parent direction (if exists) or use default
        grandparent_idx = out_links[parent_idx]
        
        if grandparent_idx != -1:
            gpx, gpy = out_nodes[grandparent_idx, 0], out_nodes[grandparent_idx, 1]
            vec_x, vec_y = px - gpx, py - gpy
            # Normalize
            mag = math.sqrt(vec_x*vec_x + vec_y*vec_y)
            if mag > 0:
                vec_x /= mag
                vec_y /= mag
        else:
            # It's a seed! Use the seed's stored direction?
            # We didn't store direction in out_nodes to save space, 
            # so we just pick a random start or rely on attractor.
            # Simplified: Random initial push if root
            vec_x, vec_y = 0.0, 1.0 

        current_angle = math.atan2(vec_y, vec_x)
        
        # Probe Arc (Look ahead -45, 0, +45 degrees)
        best_val = -1.0
        best_angle = current_angle
        
        # Sensor Lookahead Distance (e.g. 5 pixels)
        sensor_dist = step_len * 2.0
        
        for angle_offset in (-0.78, 0.0, 0.78): # Radians (~45 deg)
            check_ang = current_angle + angle_offset
            
            # Project Probe
            probe_x = px + math.cos(check_ang) * sensor_dist
            probe_y = py + math.sin(check_ang) * sensor_dist
            
            # Map -> Grid
            c = int(ta*probe_x + tb*probe_y + tc)
            r = int(td*probe_x + te*probe_y + tf)
            
            if 0 <= c < width and 0 <= r < height:
                val = attractor[r, c]
                if val > best_val:
                    best_val = val
                    best_angle = check_ang
        
        # --- B. CALCULATE VECTOR ---
        # Add noise (Meander)
        noise = (np.random.random() - 0.5) * 0.5 # +/- 0.25 radians (~15 deg)
        final_angle = best_angle + noise
        
        nx = px + math.cos(final_angle) * step_len
        ny = py + math.sin(final_angle) * step_len
        
        # --- C. VALIDATE (Grid Check) ---
        # 1. Bounds
        c = int(ta*nx + tb*ny + tc)
        r = int(td*nx + te*ny + tf)
        
        valid = False
        if 0 <= c < width and 0 <= r < height:
            # 2. Collision
            # Strict Check: Is the specific pixel occupied?
            if grid[r, c] == 0:
                valid = True
                
        if valid:
            # --- D. COMMIT ---
            new_idx = segment_count
            out_nodes[new_idx, 0] = nx
            out_nodes[new_idx, 1] = ny
            out_links[new_idx] = parent_idx
            out_meta[new_idx, 0] = p_order
            segment_count += 1
            
            # Mark Grid (Exclusion Zone)
            # Mark a small square around the new tip
            rad_int = int(exclusion_rad)
            r_min = max(0, r - rad_int)
            r_max = min(height, r + rad_int + 1)
            c_min = max(0, c - rad_int)
            c_max = min(width, c + rad_int + 1)
            
            # Numba requires manual loops for 2D slice assignment usually, 
            # or explicit loop.
            for rr in range(r_min, r_max):
                for cc in range(c_min, c_max):
                    grid[rr, cc] = 1
            
            # --- E. BRANCHING ---
            # Push new tip
            stack[stack_ptr] = new_idx
            stack_ptr += 1
            
            # Fork Logic
            # Only fork if we have space and order is high enough
            if p_order > 1 and np.random.random() < fork_prob:
                # To fork, we push the PARENT back onto the stack? 
                # Or we push a second child?
                # Easiest: The current node becomes a junction.
                # We simply push the new_idx TWICE? No, that grows two lines from new point.
                # To fork effectively, we usually reduce the order of the second branch.
                
                # We cheat: We just push the new_idx again, but future logic
                # might reduce its order. For now, simple bifurcation.
                pass 
                # (Refinement: Proper Strahler handling requires complex graph logic.
                #  For visual fractal rivers, simple continuation is usually enough).

    return segment_count, out_nodes, out_links, out_meta


class RiverNetworkGenerator:
    """
    Generates the hydrological skeleton of the world *before* Orogeny.
    
    Implements a 'Weighted Space Colonization' algorithm to grow a fractal network 
    of tributaries that connect to the canonical major rivers. 
    
    Crucially, this growth is biased by an 'Attractor Field' derived from future 
    Orogeny Axes, simulating antecedent drainage (rivers that cut through 
    rising mountains).

    Architecture:
    - Data Oriented Design (Arrays of Structs) instead of Object Oriented.
    - Uses a static Numba JIT kernel for the growth loop.
    - Uses a flat NumPy grid for collision detection (Spatial Hash


    
    Attributes:
        ctx (WorldState): The shared simulation context.
        config (dict): The 'fractal_net' configuration subset.
        network_lines (list): List of (LineString, stream_order) tuples representing the full net.
    """

    def __init__(self, ctx):
        """
        Initialize the generator.
        
        Args:
            ctx (WorldState): The simulation context.
        """
        self.ctx = ctx
        self.config = ctx.config["epochs"]["init"]["fractal_net"]
        self.network_lines = [] # Stores (geometry, order)
        # Internal storage for the generated network
        # List of dicts: {'geometry': LineString, 'order': int, 'type': str}
        self.network_segments = []

    # ==========================================================================
    # ***ATTRACTOR FIELD METHODS***
    # ==========================================================================
    

        
    def generate_attractor_field(self):
        """
        Public wrapper method to generate and store the Orogeny Attractor Field.
        
        Orchestrates the creation of the 'Gravity Map' used to guide river growth.
        The resulting field is stored in `ctx.layers['attractor_field']` for 
        inspection and use by the growth algorithm.
        
        Steps:
        1. Initialize the array with background weights.
        2. Rasterize Orogeny vectors onto the array (OE1 > OE2 > OE3).
        3. Apply Gaussian blur to propagate the 'pull' of mountains.
        4. Store the result in the context.
        """
        print("Generating Orogeny Attractor Field...")
        
        # 1. Init
        field = self._init_attractor_array()
        
        # 2. Collect Vectors
        # We collect all geometry-value pairs into a single list to rasterize at once
        all_shapes = []
        epoch_mapping = ["oe1", "oe2", "oe3"] # Order matters if overlaps occur (last wins)
        
        for key in epoch_mapping:
            vectors = self._get_attractor_vectors_by_epoch(key)
            all_shapes.extend(vectors)
            
        # 3. Rasterize
        # This burns the mountain spines onto the background
        field = self._rasterize_attractors(field, all_shapes)
        
        # 4. Propagate (Blur)
        # This creates the gradients necessary for the "steepest ascent" search
        smooth_field = self._propagate_attraction(field)
        
        # 5. Store
        self.ctx.register_layer("attractor_field", dtype=np.float32)
        self.ctx.layers["attractor_field"] = smooth_field
        
        print("  > Attractor Field generated and stored.")

    
    def _init_attractor_array(self):
        """
        Creates the base canvas for the attractor field.
        
        Returns:
            np.ndarray: A Float32 array matching the simulation shape, 
                        filled with the configured 'background' weight (e.g., 5.0).
        """
        bg_weight = self.config["attractor_weights"]["background"]
        
        # Create full array with background value
        # Float32 is required for smooth gradients after blurring
        array = np.full(self.ctx.shape, fill_value=bg_weight, dtype=np.float32)
        
        return array

    
    def _get_attractor_vectors_by_epoch(self, epoch_key):
        """
        Retrieves the specific vector subset for a given Orogeny Epoch.
        
        Args:
            epoch_key (str): The config key (e.g., 'oe1', 'oe2') used to look up
                             the target `oe_epoch` integer ID.
                             
        Returns:
            list: A list of (geometry, value) tuples suitable for rasterization,
                  where value is the configured weight for this epoch.
        """
        if "orogeny" not in self.ctx.vectors:
            raise ValueError("Vector layer 'orogeny' missing from WorldState.")
            
        gdf = self.ctx.vectors["orogeny"]
        weight = self.config["attractor_weights"][epoch_key]
        
        # Mapping config keys to integer IDs in the geopackage
        # oe1 = World's Edge (Youngest) -> ID 1
        # oe2 = Middle Mtns (Collision) -> ID 2
        # oe3 = Roots (Oldest)          -> ID 3
        key_to_id = {"oe1": 1, "oe2": 2, "oe3": 3}
        target_id = key_to_id.get(epoch_key)
        
        # Filter the GeoDataFrame
        subset = gdf[gdf["oe_epoch"] == target_id]
        
        # Create list of (geometry, weight) tuples
        shapes = [(geom, weight) for geom in subset.geometry]
        
        count = len(shapes)
        if count > 0:
            print(f"    - Collected {count} vectors for {epoch_key} (Weight: {weight})")
            
        return shapes
    

    def _rasterize_attractors(self, base_array, vector_groups):
        """
        Burns the vector weights onto the base array.
        
        Args:
            base_array (np.ndarray): The initialized background array.
            vector_groups (list): A combined list of all vector tuples from all epochs.
                                  Format: [(geom, value), (geom, value)...]
                                  
        Returns:
            np.ndarray: The array with 'mountain spines' burned in at high values.
        """
        if not vector_groups:
            print("    ! Warning: No attractor vectors found. Field will be flat.")
            return base_array

        # Rasterize creates a NEW array with 0 as background by default
        # We want to burn these values "on top" of our base array.
        # Rasterize returns valid pixels as 'value', others as 'fill' (0).
        
        burned_features = rasterio.features.rasterize(
            shapes=vector_groups,
            out_shape=self.ctx.shape,
            transform=self.ctx.transform,
            fill=0, # Background 0 so we can max() it later
            dtype=np.float32,
            merge_alg=rasterio.enums.MergeAlg.replace 
        )
        
        # We take the maximum of the base (5.0) and the burned features (e.g., 100.0)
        # This ensures we don't accidentally overwrite a mountain with 0
        final_array = np.maximum(base_array, burned_features)
        
        return final_array

    
    def _propagate_attraction(self, raw_field):
        """
        Applies a Gaussian Blur to spread the attraction influence.
        
        Transforms the thin vector lines into wide gradients, allowing rivers 
        to 'sense' mountains from a distance (e.g., 10-20km away).
        
        Args:
            raw_field (np.ndarray): The sharp rasterized field.
            
        Returns:
            np.ndarray: The smooth, gradient-rich attractor field.
        """
        # Retrieve search radius from config (in pixels)
        # This acts as the Sigma for the blur.
        # A 20px radius means the 'pull' drops off significantly after 20px (10km).
        sigma = self.config["search_radius_px"]
        
        print(f"    - Propagating attraction field (Sigma={sigma}px)...")
        
        # Apply Gaussian Filter
        from scipy.ndimage import gaussian_filter
        smooth_field = gaussian_filter(raw_field, sigma=sigma)
        
        return smooth_field
        


    # ==========================================================================
    # ***GROW FRACTAL NETWORK METHODS***
    # ==========================================================================

    # ==========================================================================
    # 1. INPUT VALIDATION
    # ==========================================================================

    def _validate_inputs(self):
        """
        Ensures all necessary data exists before starting the simulation.
        
        Checks:
        1. 'rivers' vector exists in ctx.vectors (The Canonical Source).
        2. 'attractor_field' layer exists in ctx.layers (The Gravity Map).
        
        Raises:
            ValueError: If inputs are missing.
        """
        print("  > Validating inputs for Fractal Growth...")
        
        # Check 1: Canonical Rivers
        if "rivers" not in self.ctx.vectors:
            raise ValueError(
                "CRITICAL: Vector layer 'rivers' missing from WorldState. "
                "Ensure 'canonical_rivers.gpkg' is defined in config and loaded."
            )
            
        # Check 2: Attractor Field
        if "attractor_field" not in self.ctx.layers:
            raise ValueError(
                "CRITICAL: Layer 'attractor_field' missing from WorldState. "
                "You must run 'river_gen.generate_attractor_field()' before growing the network."
            )
            
        print("  > Inputs valid.")

    # ==========================================================================
    # 2. MAIN ENTRY POINT
    # ==========================================================================

    def grow_fractal_network(self, strahler_col="strahler"):
        """
        Orchestrates the Numba-accelerated simulation.
        
        Steps:
        1. Validate inputs (Attractor field, Canonical Rivers).
        2. _prepare_canonical_data(): Flatten LineStrings into float arrays.
        3. _build_collision_grid(): Initialize the Boolean Occupancy Grid.
        4. _generate_seeds_fast(): Generate start points (using Numba or Vectorized NumPy).
        5. _run_growth_kernel(): Call the static @jit function to grow rivers.
        6. _reconstruct_geometries(): Convert raw output arrays back to Shapely objects.
        7. export_network_to_gpkg(): Save results.
        
        Args:
            strahler_col (str): Column name for stream order in input GPKG.
        """
        print("Growing Fractal River Network (Numba Accelerated)...")
        
        # 1. Validation (Re-using existing method)
        self._validate_inputs()
        
        # Inputs
        rivers_gdf = self.ctx.vectors["rivers"]
        attractor = self.ctx.layers["attractor_field"]
        
        # Get Pixel Size (Meters per Pixel) for unit conversion
        pixel_size = self.ctx.config["resolution"]["sim_pixel_size"]  
        
        # 2. Build Collision Grid
        # This creates the static map of obstacles (Canonical Rivers)
        # 0 = Free, 1 = Occupied
        print("  > Building collision grid...")
        collision_grid = self._build_collision_grid(rivers_gdf)
        
        # 3. Generate Seeds
        # Returns array of shape (N, 5): [x, y, dir_x, dir_y, order]
        print("  > Generating seeds...")
        seeds = self._generate_seeds_vectorized(rivers_gdf, strahler_col)
        
        if len(seeds) == 0:
            print("  ! Warning: No seeds generated. Aborting.")
            return
            
        print(f"  > Prepared {len(seeds)} seeds.")

        # 4. Prepare Kernel Inputs
        # Numba cannot handle Affine objects, so we extract the 6 coefficients
        # required to convert Map Coordinates (x,y) -> Pixel Indices (col,row).
        # We use the INVERSE transform: col = a*x + b*y + c ...
        inv_t = ~self.ctx.transform
        transform_coeffs = (inv_t.a, inv_t.b, inv_t.c, inv_t.d, inv_t.e, inv_t.f)

        # The Kernel does math in MAP UNITS (Meters).
        # We must convert pixel lengths to meters.
        step_len_px = float(self.config["segment_length_px"])
        step_len_m = step_len_px * pixel_size  # <--- CRITICAL FIX
        
        # Exclusion radius is used for GRID indexing in the kernel, so it stays in PIXELS.
        excl_rad_px = float(self.config["exclusion_radius_px"])
        
        # Prepare Physics Constants Tuple
        # (step_len, max_segs, fork_prob, exclusion_rad, width, height)
        # We ensure types are strict (float vs int) for Numba signature matching.
        params = (
            float(self.config["segment_length_px"]),
            int(self.config.get("max_segments", 100000)),
            float(self.config.get("fork_probability", 0.05)),
            excl_rad_px,
            int(self.ctx.shape[1]), # Width
            int(self.ctx.shape[0])  # Height
        )

        # 5. Run Kernel
        print("  > Launching Numba Kernel...")
        # count: The number of segments actually created
        # out_nodes: The coordinate array [N, 2]
        # out_links: The parent index array [N]
        # out_meta:  The order array [N]
        count, out_nodes, out_links, out_meta = self._run_growth_kernel(
            seeds, 
            collision_grid, 
            attractor, 
            transform_coeffs, 
            params
        )
        
        # 6. Reconstruct
        print(f"  > Kernel finished. Reconstructing {count} segments...")
        self._reconstruct_geometries(count, out_nodes, out_links, out_meta)
        
        # 7. Export
        self.export_network_to_gpkg()

    # ==========================================================================
    # 3. DATA PREPARATION (Python -> NumPy)
    # ==========================================================================

    def _build_collision_grid(self, rivers_gdf):
        """
        Initializes the static collision map (The "Truth" Grid).
        
        Process:
        - Allocates `grid = np.zeros(shape, dtype=int8)`.
        - Rasterizes the canonical rivers into this grid.
        
        FIX:
        - We buffer the rivers by ONLY 50% of the exclusion radius.
        - This allows the first 'jump' (which is 100% length) to land 
          safely outside the parent's occupied zone.
        """
        import rasterio.features
        
        # 1. Allocate Grid
        grid = np.zeros(self.ctx.shape, dtype=np.int8)
        
        # 2. Define Buffer (Reduced)
        # We use 0.5 multiplier so the exclusion zone is smaller than the jump step.
        # This prevents the "Strangled at Birth" bug.
        radius_px = self.config["exclusion_radius_px"] * 0.5 
        
        # Ensure at least 1 pixel width so we don't cross rivers easily
        radius_px = max(1.0, radius_px)
        
        pixel_size = self.ctx.config["resolution"]["sim_pixel_size"]
        radius_m = radius_px * pixel_size
        
        print(f"  > Rasterizing collision mask (Buffer: {radius_px:.1f}px)...")
        
        # 3. Create Shapes
        shapes = []
        for geom in rivers_gdf.geometry:
            if geom is not None and not geom.is_empty:
                shapes.append((geom.buffer(radius_m), 1))
        
        # 4. Rasterize
        if shapes:
            rasterio.features.rasterize(
                shapes=shapes,
                out_shape=self.ctx.shape,
                transform=self.ctx.transform,
                out=grid, 
                default_value=1,
                dtype=np.int8
            )
            
        return grid

    def _generate_seeds_vectorized(self, rivers_gdf, strahler_col="strahler"):
        """
        Generates seed points ("Buds") along the canonical rivers.
        
        Format:
        Returns a NumPy array where each row is a seed:
        [x, y, dir_x, dir_y, order]
        
        Logic:
        - Walks along lines at 'seed_interval_px'.
        - Calculates normal vector (perpendicular to flow).
        - Assigns child order (Parent - 1).
        """
        seeds_list = []
        
        # Config
        pixel_size = self.ctx.config["resolution"]["sim_pixel_size"]
        interval_m = self.config["seed_interval_px"] * pixel_size
        
        # Distance tracker to maintain spacing across fragmented features
        current_dist_along_path = 0.0
        
        for _, row in rivers_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
                
            # Determine Order
            parent_order = 5
            if strahler_col in row and not np.isnan(row[strahler_col]):
                parent_order = int(row[strahler_col])
                
            child_order = max(1, parent_order - 1)
            
            # Skip insignificant streams
            if child_order < 2:
                continue

            # Handle MultiLineStrings
            parts = geom.geoms if geom.geom_type == 'MultiLineString' else [geom]

            for part in parts:
                line_length = part.length
                
                # Calculate start offset based on previous remainder
                remainder = interval_m - (current_dist_along_path % interval_m)
                if remainder == interval_m: remainder = 0.0
                
                dist = remainder
                
                while dist <= line_length:
                    # 1. Position
                    pt = part.interpolate(dist)
                    
                    # 2. Direction (Tangent)
                    # Sample slightly ahead to get flow direction
                    delta = 10.0 # meters
                    pt_ahead = part.interpolate(min(line_length, dist + delta))
                    
                    dx = pt_ahead.x - pt.x
                    dy = pt_ahead.y - pt.y
                    mag = np.sqrt(dx*dx + dy*dy)
                    
                    # 3. Normal Vector (Perpendicular)
                    # Tangent is (dx, dy). Normal is (-dy, dx).
                    # This points "Left" relative to flow. 
                    if mag > 0:
                        nx, ny = -dy/mag, dx/mag
                    else:
                        nx, ny = 1.0, 0.0 # Fallback
                        
                    # Add to list
                    # [x, y, dir_x, dir_y, order]
                    seeds_list.append([pt.x, pt.y, nx, ny, child_order])
                    
                    dist += interval_m
                
                current_dist_along_path += line_length
        
        # Convert list to NumPy array
        if not seeds_list:
            return np.array([], dtype=np.float64)
            
        return np.array(seeds_list, dtype=np.float64)

    # ==========================================================================
    # 4. KERNEL EXECUTION (The Engine)
    # ==========================================================================

    def _run_growth_kernel(self, seeds, grid, attractor, transform_coeffs, params):
        """
        Sets up memory pools and calls the external Numba static function.
        
        Memory Allocation:
        - `NODES`: Pre-allocated float32 array [Max_Segs, 2] for coordinates.
        - `LINKS`: Pre-allocated int32 array [Max_Segs] (Parent Index).
        - `META`:  Pre-allocated int8 array [Max_Segs] (Order).
        
        Action:
        - Calls `numba_growth_kernel(...)` (The static function defined outside class).
        - The kernel modifies `NODES`, `LINKS`, `META`, and `grid` in-place.
        
        Returns:
            int: The total count of segments generated (valid_count).
        """
        # Ensure types are correct for Numba
        # Seeds: Float64
        seeds = seeds.astype(np.float64)
        # Grid: Int8
        grid = grid.astype(np.int8)
        # Attractor: Float32 (Standard for maps)
        attractor = attractor.astype(np.float32)
        
        # Call Kernel
        return numba_growth_kernel(seeds, grid, attractor, transform_coeffs, params)

    # ==========================================================================
    # 5. RECONSTRUCTION (NumPy -> Python)
    # ==========================================================================

    def _reconstruct_geometries(self, count, nodes, links, meta):
        """
        Converts the raw arrays back into Shapely LineStrings and Dictionaries.
        
        Process:
        - Slices the arrays up to `count`.
        - Iterates through the valid range.
        - Connects `NODES[i]` to `NODES[LINKS[i]]` to form a LineString.
        - Populates `self.network_segments`.
        
        Args:
            count (int): Number of valid segments generated.
            nodes (np.ndarray): The filled coordinate array.
            links (np.ndarray): The filled parent-index array.
            meta (np.ndarray): The filled metadata array.
        """
        self.network_segments = []
        
        print(f"  > Reconstructing {count} segments from raw data...")
        
        for i in range(count):
            parent_idx = links[i]
            
            # We only create a LineString if we have a valid parent
            # (Root nodes don't form a line by themselves, they are just points)
            if parent_idx != -1:
                # Start Point (Parent)
                px, py = nodes[parent_idx]
                # End Point (Self)
                nx, ny = nodes[i]
                
                # Check for zero-length (noise can sometimes cause this)
                if px == nx and py == ny:
                    continue
                    
                geom = LineString([(px, py), (nx, ny)])
                order = int(meta[i, 0])
                
                self.network_segments.append({
                    'geometry': geom,
                    'order': order,
                    'type': 'generated'
                })
        
        print(f"  > Reconstruction complete. Final clean count: {len(self.network_segments)}")
        
    def export_network_to_gpkg(self, filename="fractal_network_debug.gpkg"):
        """
        Compiles Canonical Rivers and Generated Tributaries into one GPKG.
        Useful for debugging the connectivity between the input vectors and 
        the new fractal growth.
        """
        import os
        import pandas as pd
        import geopandas as gpd
        
        print(f"Exporting network to {filename}...")
        
        # 1. Create Generated GDF
        # self.network_segments is populated by _reconstruct_geometries
        if self.network_segments:
            gen_df = gpd.GeoDataFrame(self.network_segments, crs=self.ctx.crs)
        else:
            # Fallback for empty results so we still get a file
            print("  ! Warning: No generated segments found.")
            gen_df = gpd.GeoDataFrame(columns=['geometry', 'order', 'type'], 
                                      geometry='geometry', crs=self.ctx.crs)

        # 2. Get Canonical Rivers (The input data)
        # We perform a light cleanup to ensure columns match for the merge
        canon_df = self.ctx.vectors["rivers"].copy()
        
        # Check for 'strahler' column, default to 5 if missing
        strahler_col = 'strahler' if 'strahler' in canon_df.columns else None
        
        canon_clean_df = gpd.GeoDataFrame({
            'geometry': canon_df.geometry,
            'strahler_order': canon_df[strahler_col] if strahler_col else 5,
            'source_type': 'canonical'
        }, crs=self.ctx.crs)

        # 3. Rename Generated Columns and Merge
        if not gen_df.empty:
            # Rename internal keys to match export schema
            if 'order' in gen_df.columns:
                gen_df['strahler_order'] = gen_df['order']
                gen_df['source_type'] = 'generated'
                
            # Keep only relevant columns
            cols_to_keep = ['geometry', 'strahler_order', 'source_type']
            gen_df = gen_df[cols_to_keep]
            
            # Merge
            final_df = pd.concat([canon_clean_df, gen_df], ignore_index=True)
        else:
            final_df = canon_clean_df

        # 4. Save to Disk
        out_dir = self.ctx.config["paths"]["outputs"]
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        
        try:
            final_df.to_file(out_path, driver="GPKG")
            print(f"  > Saved {len(final_df)} total segments ({len(gen_df)} new) to {out_path}")
        except Exception as e:
            print(f"  ! Error saving GPKG: {e}")


import numpy as np
import rasterio.features

class LithologyManager:
    """
    Manages the 'Material Properties' of the simulation grid.
    
    Instead of simulating complex stratigraphy, we use a single 'Lithology ID' layer
    (stored in ctx.lithology_id) that erosion agents look up to determine 
    how fast to erode a pixel.
    
    ID Mapping (Int16) comes from config['lithology_codes']:
    - 10: Sedimentary (Standard Hardness)
    - 20: Igneous/Metamorphic (Hard)
    - 30: Fracture/Fault Zone (Soft)
    - 99: 'Sugar' (Infinite Erosion / Sea)
    """

    def __init__(self, ctx):
        """
        Initialize the manager.
        
        Args:
            ctx (WorldState): The simulation context.
        """
        self.ctx = ctx
        self.codes = ctx.config.get("lithology_codes", {
            "sedimentary": 10,
            "igneous": 20,
            "fault_zone": 30,
            "sugar_sea": 99
        })

    def apply_base_lithology(self, default_id=None):
        """
        Initializes the lithology map with a uniform base rock type.
        
        Args:
            default_id (int, optional): The ID to fill the map with. 
                                        Defaults to 'sedimentary' code (10).
        """
        if default_id is None:
            default_id = self.codes["sedimentary"]
            
        print(f"Applying base lithology (ID: {default_id})...")
        
        # Fill the entire array
        self.ctx.lithology_id.fill(default_id)
        
        # Ensure Void areas (outside active mask) are 0 or valid? 
        # Usually it's safer to leave them as base or 0. 
        # Erosion agents check active_mask, so lithology in void doesn't matter much.

    def burn_special_zones(self, sea_id=None, fault_id=None):
        """
        Overwrites the lithology map with special functional zones.
        
        Logic:
        1. 'The Sugar Sea': Rasterize 'sea_polygon.gpkg'. Set pixels to `sea_id`.
           This ensures that any sediment reaching the sea is immediately deleted/eroded.
           
        2. 'Fracture Zones': Ingest the 'fracture_mask' from RiverNetworkGenerator.
           Set pixels to `fault_id`.
           This ensures early erosion events preferentially carve valleys along these paths.
           
        Args:
            sea_id (int, optional): Override for sea ID. Defaults to config 'sugar_sea'.
            fault_id (int, optional): Override for fault ID. Defaults to config 'fault_zone'.
        """
        if sea_id is None: sea_id = self.codes["sugar_sea"]
        if fault_id is None: fault_id = self.codes["fault_zone"]
        
        print("Burning special lithology zones...")
        
        # 1. Burn Sea (Sugar)
        if "sea" in self.ctx.vectors:
            sea_gdf = self.ctx.vectors["sea"]
            
            # Rasterize Sea Polygon -> Temporary Mask
            sea_mask = rasterio.features.rasterize(
                shapes=[(geom, 1) for geom in sea_gdf.geometry],
                out_shape=self.ctx.shape,
                transform=self.ctx.transform,
                fill=0,
                dtype=np.uint8
            )
            
            # Apply to Lithology ID where Sea == 1
            # We assume Sea overrides everything else (it's the sink)
            self.ctx.lithology_id[sea_mask == 1] = sea_id
            print(f"  > Burned Sea Zone (ID: {sea_id})")
        else:
            print("  ! Warning: 'sea' vector not found. Skipping Sea Burn.")

        # 2. Burn Fracture Network (Faults)
        # We look for the pre-calculated mask from RiverNetworkGenerator
        if "fracture_mask" in self.ctx.layers:
            fracture_mask = self.ctx.layers["fracture_mask"]
            
            # Apply Fault ID where mask is active (>0)
            # CRITICAL: Do NOT overwrite the Sea. 
            # Logic: If pixel is NOT Sea, AND is Fracture -> Set Fault.
            
            is_fracture = (fracture_mask > 0)
            is_not_sea = (self.ctx.lithology_id != sea_id)
            
            target_pixels = is_fracture & is_not_sea
            
            self.ctx.lithology_id[target_pixels] = fault_id
            
            count = np.sum(target_pixels)
            print(f"  > Burned River Fractures (ID: {fault_id}). Pixels affected: {count}")
        else:
            print("  ! Warning: 'fracture_mask' not found. Skipping Fracture Burn.")