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


import rasterio.features
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Spine:
    """
    Lightweight data container for a single mountain ridge segment.
    Using a dataclass for memory efficiency and type hinting.
    """
    geometry: LineString
    epoch: int
    base_amplitude: float
    width_limit: float
    generation: int
    # Add an ID or parent_id here if we need to trace lineage later
    
class OrogenySpineGenerator:
    """
    Generates a fractal network of mountain spines (ridges) from input tectonic axes.
    
    This class operates strictly on vector geometry (LineStrings). It implements a 
    recursive growth algorithm where main axes are roughened (fractalized) and then 
    sprout orthogonal 'spurs', which in turn roughen and sprout their own spurs.
    """
    def __init__(self, ctx):
        self.ctx = ctx
        # self.config = ctx.config.get("orogeny", {})
        self.active_pool: List[Spine] = [] 


    def generate_structural_vectors(self, generations: int = 3, roughness: float = 0.5):
        """
        Public entry point. Orchestrates the full generation pipeline.
        Renamed from 'run' to be more descriptive.
        
        Args:
            generations (int): How many levels of recursion (0=Main, 1=Spurs, 2=Sub-spurs).
            roughness (float): Magnitude of the fractal displacement.
        """
        # 1. Load Gen 0 (The Canonical Axes) from the context
        # These are currently straight lines from the GIS vector tool.
        self._ingest()
        
        # 2. Prepare Gen 0: Fractalize the Inputs
        # The user's input vectors are too smooth. We must roughen them *before* # starting the growth loop so that spurs branch off a jagged line, not a straight one.
        # (We only apply this to the initial inputs here).
        for spine in self.active_pool:
            if spine.generation == 0:
                spine.geometry = self._fractalize_geometry(spine.geometry, roughness)

        # 3. Recursive Growth Loop
        # We start with the Gen 0 parents we just loaded and roughened.
        current_generation_spines = [s for s in self.active_pool if s.generation == 0]
        
        for i in range(generations):
            # Pass the current generation index to logic
            # Note: We are NOT fractalizing the new spurs inside this loop yet.
            # They will be generated as straight lines for structural verification.
            new_children = self._execute_growth_loop(
                parents=current_generation_spines, 
                roughness=roughness, 
                current_gen_index=i
            )
            
            # The children become the parents for the next iteration
            current_generation_spines = new_children
            
        # 4. Save result to disk
        self._sink_to_gpkg()

    
    
    def _ingest(self):
        """
        Loads the 'orogeny_axes' layer from the context.
        
        Actions:
            1. Iterates through GeoDataFrame rows.
            2. Extracts 'oe_epoch', 'amplitude', 'width_km'.
            3. Converts geometry to Generation 0 Spine objects.
            4. Populates self.active_pool.
        """
        # Validate existence of layer
        if 'orogeny' not in self.ctx.vectors:
            print("Warning: 'orogeny' layer not found in context.")
            return

        gdf = self.ctx.vectors['orogeny']
        required_cols = {'oe_epoch': 1, 'amplitude': 1000.0, 'width_km': 10.0}
        
        # Ensure cols exist
        for col, val in required_cols.items():
            if col not in gdf.columns: gdf[col] = val

        new_spines = []
        for _, row in gdf.iterrows():
            geoms = [row.geometry] if row.geometry.geom_type == 'LineString' else list(row.geometry.geoms)
            for geom in geoms:
                new_spines.append(Spine(
                    geometry=geom,
                    epoch=int(row['oe_epoch']),
                    base_amplitude=float(row['amplitude']),
                    width_limit=float(row['width_km']),
                    generation=0
                ))
        
        self.active_pool.extend(new_spines)
        print(f"  > Ingested {len(new_spines)} tectonic axes.")

    def _execute_growth_loop(self, parents: List[Spine], roughness: float, current_gen_index: int) -> List[Spine]:
        """
        Wrapper for a single generation cycle.
        
        Args:
            parents (List[Spine]): The spines from the previous generation.
            roughness (float): Displacement factor.
            current_gen_index (int): The current generation number (0, 1, etc).
            
        Returns:
            List[Spine]: The newly created children (Generation N+1).
            
        Logic:
            1. Iterate through 'parents'.
            2. Call _fractalize_geometry() on the parent (in-place modification).
            3. Determine how many spurs to spawn (density).
            4. Loop through spur count:
               a. _select_spur_location
               b. _project_spur
               c. _orient_spur
               d. _spur_length
               e. _inherit
               f. _add_to_pool
            5. Return the list of new children.
        """
        pass

    
        
    def _fractalize_geometry(self, geometry: LineString, roughness: float) -> LineString:
        """
        Executes Step A: Fractalization (1D Midpoint Displacement).
        
        Engineered for speed:
        - Uses NumPy vectorization instead of Python loops.
        - Optimization: Avoids Sqrt() calls. 
          Standard Midpoint Displacement: Offset = Unit_Normal * Length * Roughness * Random
          Optimization: Offset = Perpendicular_Vector * Roughness * Random
          (Since Perpendicular_Vector magnitude IS Length, they cancel out).
        
        Args:
            geometry (LineString): The input line.
            roughness (float): Displacement scale relative to segment length.
            
        Returns:
            LineString: A subdivided, jagged line (2N - 1 points).
        """
        coords = np.array(geometry.coords)
        if len(coords) < 2: return geometry

        vectors = coords[1:] - coords[:-1]
        midpoints = (coords[:-1] + coords[1:]) * 0.5
        
        # Perpendiculars (-y, x)
        perp = np.empty_like(vectors)
        perp[:, 0] = -vectors[:, 1]
        perp[:, 1] = vectors[:, 0]
        
        rng = np.random.uniform(-1.0, 1.0, size=(len(vectors), 1))
        
        # Displacement
        displaced_mids = midpoints + (perp * (roughness * rng))
        
        # Interleave
        new_count = len(coords) + len(displaced_mids)
        result = np.empty((new_count, 2), dtype=coords.dtype)
        result[0::2] = coords
        result[1::2] = displaced_mids
        
        return LineString(result)
        return LineString(result_coords)

    
    def _select_spur_location(self, parent_geo: LineString) -> Tuple[float, float, float]:
        """
        Selects a point along the parent spine to sprout a child.
        
        Returns:
            (x, y, tangent_angle): The coords and the local angle of the parent line 
                                   at that point (needed for projection).
        """
        pass

    def _project_spur(self, x: float, y: float, tangent_angle: float, length: float) -> LineString:
        """
        Calculates the geometry of the new spur.
        
        Logic:
            1. Calculate Normal vector (tangent + 90 degrees).
            2. Create a straight line from (x,y) along the Normal for 'length' meters.
            
        Note: The spur starts straight. It will be fractalized in the *next* generation loop when it becomes a parent.
        """
        pass

    def _orient_spur(self, tangent_angle: float) -> float:
        """
        Determines the directionality of the spur (Left vs Right).
        
        Logic:
            Randomly chooses +90 or -90 degrees offset from tangent.
            May also add slight random jitter (+/- 15 deg) so spurs aren't perfectly orthogonal.
        """
        pass

    def _spur_length(self, parent_width: float, current_gen: int) -> float:
        """
        Calculates how long the new spur should be.
        
        Logic:
            Length should decay with generations.
            e.g. Gen 1 = 50% of parent width. Gen 2 = 25% of parent width.
        """
        pass

    def _inherit(self, parent: Spine, new_geometry: LineString) -> Spine:
        """
        Creates the new Child Spine object.
        
        Args:
            parent (Spine): The source of genetic metadata (Epoch, Amp).
            new_geometry (LineString): The projected line.
            
        Returns:
            Spine: The initialized child object with Gen = Parent.Gen + 1.
        """
        pass

    def _add_to_pool(self, new_spine: Spine, child_list: List[Spine]):
        """
        Standardizes adding new items to data structures.
        
        Args:
            new_spine (Spine): The object to store.
            child_list (List): The temporary list for the current loop iteration.
            
        Side Effect:
            Adds to self.active_pool (Master record) AND child_list (Iteration record).
        """
        pass

    def _sink_to_gpkg(self):
        """
        Export handling.
        
        Actions:
            1. Convert self.active_pool (List[Spine]) to a GeoDataFrame.
            2. Save to temporary storage or output path defined in Config.
            3. Log summary stats (e.g. "Generated 4500 ridge segments").
        """
        pass

    def rasterize_structure(self, layer_name: str = "debug_spines"):
        """
        Rasterizes the generated vectors into the WorldState for visual review.
        
        Encodes 'Generation' into pixel brightness:
        - Gen 0 (Main Axis): 255
        - Gen 1 (Spurs): 180
        - Gen 2 (Sub-spurs): 100
        
        Args:
            layer_name (str): The key to store the resulting array in ctx.layers.
        """
        if not self.active_pool:
            print("Warning: No spines to rasterize. Run generate_structural_vectors() first.")
            return

        print(f"Rasterizing {len(self.active_pool)} spines to layer '{layer_name}'...")

        # 1. Define Value Mapping based on Generation
        # (Gen 0 is brightest, Gen N is dimmer)
        gen_brightness = {0: 255, 1: 180, 2: 100, 3: 50}

        # 2. Prepare shapes for rasterio
        # Format: list of (geometry, value) tuples
        shapes = []
        for spine in self.active_pool:
            pixel_val = gen_brightness.get(spine.generation, 50)
            shapes.append((spine.geometry, pixel_val))

        # 3. Rasterize
        # all_touched=True ensures thin lines are drawn even if they don't cover center of pixel
        raster = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=self.ctx.shape,
            transform=self.ctx.transform,
            fill=0,
            all_touched=True, 
            dtype=np.uint8
        )

        # 4. Store in Context
        self.ctx.layers[layer_name] = raster
        print(f"  > Layer '{layer_name}' created.")





        
class RiverNetworkGenerator:
    """
    Bare Metal implementation.
    Passes on complex generation.
    Only handles rasterization of existing Canonical Rivers.
    """
    def __init__(self, ctx):
        self.ctx = ctx
        self.config = ctx.config.get("epochs", {}).get("init", {}).get("fractal_net", {})
        self.network_segments = [] # Placeholder for future generated segments

    def generate_attractor_field(self):
        """
        PLACEHOLDER: Will generate rainfall noise.
        """
        print("  > [Bare Metal] Skipping attractor field generation.")
        pass

    def grow_fractal_network(self, strahler_col="strahler"):
        """
        PLACEHOLDER: Will run the Ghost Terrain physics.
        """
        print("  > [Bare Metal] Skipping fractal network growth.")
        pass

    def burn_fractures(self):
        """
        Creates the 'fracture_mask' layer.
        For this pass, it ONLY burns the Canonical Rivers (the input vectors).
        """

        print("  > Rasterizing river network (Fracture Mask)...")
        
        # 1. Initialize
        shape = self.ctx.shape
        mask = np.zeros(shape, dtype=np.uint8)
        
        # 2. Get Canonical Rivers
        rivers_gdf = self.ctx.vectors.get("rivers")
        
        if rivers_gdf is not None:
            print(f"    - Processing canonical rivers ({len(rivers_gdf)} features)")
            
            # Buffer slightly (e.g. 2 pixels)
            pixel_size = self.ctx.config["resolution"]["sim_pixel_size"]
            buffer_width = 2 * pixel_size
            
            shapes = []
            for geom in rivers_gdf.geometry:
                if geom:
                    shapes.append((geom.buffer(buffer_width), 1))
            
            # 3. Burn
            if shapes:
                rasterio.features.rasterize(
                    shapes=shapes,
                    out_shape=shape,
                    transform=self.ctx.transform,
                    out=mask,
                    default_value=1
                )
        else:
            print("    ! Warning: No 'rivers' vector layer found.")

        # 4. Save to context
        self.ctx.layers["fracture_mask"] = mask
        print("  > Fracture mask created.")





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