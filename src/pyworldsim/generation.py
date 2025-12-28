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




import numpy as np
import geopandas as gpd
import rasterio.features
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import unary_union

class RiverNetworkGenerator:
    """
    Generates the hydrological skeleton of the world *before* Orogeny.
    
    Implements a 'Weighted Space Colonization' algorithm to grow a fractal network 
    of tributaries that connect to the canonical major rivers. 
    
    Crucially, this growth is biased by an 'Attractor Field' derived from future 
    Orogeny Axes, simulating antecedent drainage (rivers that cut through 
    rising mountains).
    
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
        pass

    # ==========================================================================
    # 2. SEED GENERATION ("The Budding")
    # ==========================================================================

    def _seed_tributaries(self, rivers_gdf):
        """
        Generates starting points ('Buds') for new tributaries along the canonical rivers.
        
        Context Handling:
        The input vector data is often fragmented (112k features). A single logical river 
        might be split into 50 tiny 200m segments. To prevent overcrowding or gaps, 
        this method uses a 'Cumulative Distance' approach.
        
        Algorithm:
        1. Iterate through features in the GeoDataFrame.
        2. Track a `cumulative_dist` counter across feature boundaries.
        3. Place a Seed Point every `seed_interval` (e.g., 5km).
           - If a feature ends before the interval is reached, carry the remainder 
             over to the next feature.
        4. Assign initial Stream Order:
           - Seeds inherit `Parent_Order - 1` (e.g., Canonical=5 -> Seed=4).
        
        Args:
            rivers_gdf (GeoDataFrame): The canonical major rivers.
            
        Returns:
            list: A list of 'Seed' objects (dictionaries with point, direction, order).
        """
        pass

    # ==========================================================================
    # 3. THE GROWTH LOOP
    # ==========================================================================

    def grow_fractal_network(self):
        """
        The Main Orchestrator. Executes the Weighted Space Colonization algorithm.
        
        Steps:
        1. Call _validate_inputs().
        2. Build R-Tree Spatial Index from Canonical Rivers (Collision Map).
        3. Call _seed_tributaries() to populate the initial Queue.
        4. While Queue is not empty:
           a. Pop a candidate tip.
           b. _sense_attractor_gradient() -> Find uphill direction.
           c. _calculate_growth_vector() -> Determine geometry.
           d. _is_segment_valid() -> Check bounds/collisions.
           e. _commit_segment() -> Add to map and R-Tree.
           f. _process_branching() -> Add new tips back to Queue.
           
        5. _compile_network() to finalize data structures.
        """
        pass

    # --- 3.1 Sensing ---
    def _sense_attractor_gradient(self, start_point, current_vector):
        """
        Scans the Attractor Field to find the 'Uphill' direction.
        
        Logic:
        1. Sample the field in an arc (e.g., +/- 90 degrees) ahead of the current flow.
        2. Identify the pixel with the highest attraction value (Steepest Ascent).
        3. If the highest value is lower than the current position's value, 
           apply a penalty (discourage growing downhill), but allow it if no other option exists.
           
        Args:
            start_point (Point): Current tip location.
            current_vector (tuple): (dx, dy) of the previous segment.
            
        Returns:
            float: The target angle (radians) towards the attractor.
        """
        pass

    # --- 3.2 Vector Calculation ---
    def _calculate_growth_vector(self, start_point, target_angle):
        """
        Determines the geometry of the new segment.
        
        Logic:
        1. Apply 'Inertia': Blend the target_angle with the previous angle 
           (Rivers don't turn 90 degrees instantly).
        2. Apply 'Noise': Add Perlin/Random perturbation (Meander).
        3. Project new point: `New = Start + (Vector * Segment_Length)`.
        
        Args:
            start_point (Point): Origin.
            target_angle (float): The optimal uphill direction.
            
        Returns:
            LineString: The proposed new segment geometry.
        """
        pass

    # --- 3.3 Check ---
    def _is_segment_valid(self, candidate_geom, rtree_index):
        """
        Validates the proposed segment against the 'Rules of Nature'.
        
        Checks:
        1. Map Bounds: Is it inside the active simulation area?
        2. Collision/Density: Does it intersect or come too close to 
           existing rivers in the R-Tree? (radius defined in config).
           
        Args:
            candidate_geom (LineString): The line to test.
            rtree_index (Index): The spatial index of the current network.
            
        Returns:
            bool: True if valid, False if blocked.
        """
        pass

    # --- 3.4 Commit ---
    def _commit_segment(self, segment_data, rtree_index):
        """
        Finalizes a valid segment.
        
        Actions:
        1. Adds segment to `self.network_segments` list.
        2. Inserts segment bounds into the R-Tree for future collision checks.
        
        Args:
            segment_data (dict): {'geometry': Line, 'order': int, ...}
            rtree_index (Index): The spatial index to update.
        """
        pass

    # --- 3.5 & 3.6 Branching & Probabilistic Forks ---
    def _process_branching(self, segment_geom, current_order):
        """
        Determines if the river continues, stops, or forks.
        
        Logic:
        1. Energy Check: If `current_order <= 1`, growth stops (Stream too small).
        2. Continuation: Typically adds 1 new tip (continuation of current stream).
        3. Bifurcation (Forking):
           - Uses `fork_probability` from config.
           - If triggered, generates a SECOND tip at the same location.
           - Both tips inherit `current_order` (or `order - 1` depending on Strahler rules).
           
        Args:
            segment_geom (LineString): The just-committed segment.
            current_order (int): The Strahler order of that segment.
            
        Returns:
            list: A list of new 'Tip' objects to add to the Queue.
        """
        pass

    # ==========================================================================
    # 4. COMPILATION
    # ==========================================================================

    def export_network_to_gpkg(self, filename="fractal_network_debug.gpkg"):
        """
        Compiles the internal list of segments into a GeoDataFrame and saves to disk.
        
        Structure:
        - Combines Canonical Rivers (Order 5) and Generated Tributaries.
        - Attributes: ['geometry', 'strahler_order', 'source_type'].
        
        Args:
            filename (str): Name of the output file in the outputs directory.
        """
        pass



        
    def burn_fractures(self):
        """
        Rasterizes the generated river network into a 'fracture_mask'.
        
        Logic:
        1. Create an empty Int16 mask.
        2. Group river segments by Stream Order.
        3. Iterate through orders (1 to 5):
           - Retrieve 'burn_width' from config for this order.
           - Buffer the LineStrings by this width (in map units).
           - Rasterize the buffered polygons into the mask (Value = 1).
        4. Register 'fracture_mask' to the context layers.
        """
        print("Burning River Fractures into Lithology...")
        
        fracture_mask = np.zeros(self.ctx.shape, dtype=np.int16)
        burn_widths = self.config["burn_widths"] # e.g. {5: 6px, 1: 1px}
        pixel_size = self.ctx.config["resolution"]["sim_pixel_size"]
        
        # Sort by order ascending so larger rivers overwrite smaller ones (if needed)
        # actually order doesn't matter if value is just boolean 1, 
        # but buffering size matters.
        
        for geometry, order in self.network_lines:
            # Get width in pixels, convert to meters for buffering
            # (Assuming vectors are in projected CRS meters)
            width_px = burn_widths.get(order, 1)
            width_meters = width_px * pixel_size
            
            # Buffer geometry (Radius = width / 2)
            buffered_geom = geometry.buffer(width_meters / 2)
            
            # Rasterize this single feature (or batch them for speed)
            # rasterio.features.rasterize(...)
            
            # (Ideally we batch all geometries of the same order/width 
            # and rasterize once per order for performance)
        
        # Save to context
        self.ctx.register_layer("fracture_mask", dtype=np.int16)
        self.ctx.layers["fracture_mask"] = fracture_mask
        
        print("  > Fracture mask generated.")


class LithologyManager:
    """
    Manages the 'Material Properties' of the simulation grid.
    
    Instead of simulating complex stratigraphy, we use a single 'Lithology ID' layer
    that erosion agents look up to determine how fast to erode a pixel.
    
    ID Mapping (Int16):
    - 10: Sedimentary (Standard Hardness)
    - 20: Igneous/Metamorphic (Hard)
    - 30: Fracture/Fault Zone (Soft)
    - 99: 'Sugar' (Infinite Erosion / Sea)
    """

    def __init__(self, ctx):
        pass

    def apply_base_lithology(self, ctx, default_id=10):
        """
        Initializes the lithology map.
        
        Args:
            ctx (WorldState): The context.
            default_id (int): The background rock type (Default: 10 Sedimentary).
        """
        pass

    def burn_special_zones(self, ctx, sea_hardness_id=99, fault_hardness_id=30):
        """
        Overwrites the lithology map with special functional zones.
        
        Logic:
        1. 'The Sugar Sea': Rasterize 'sea_polygon.gpkg'. Set pixels to `sea_hardness_id`.
           This ensures that any sediment reaching the sea is immediately deleted/eroded.
           
        2. 'Fracture Zones': Ingest the 'fracture_mask' from RiverNetworkGenerator.
           Set pixels to `fault_hardness_id`.
           This ensures early erosion events preferentially carve valleys along these paths.
        """
        pass