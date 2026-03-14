class ProjectCrate:
    """
    This class sets up the RO crate for the project. 
    The role of the RO crate is to ensure re[producibility and enforce data management throughout the process.

    What we will need from the ProjectCrate is to ensure that all of the variables that we put in - beginning at the very first starting raster, get captured, and moved either to the crate folder (if they are files) or to the crate manifest (if they are metadata).

    Types of files expected:
    * geotiffs
    * gpkg

    
    """
    









class OrogenySpineGenerator:
    """
    Generates a fractal network of mountain spines (ridges) from input tectonic axes.
    
    The class uses Monte Carlo draws of major spines followed by generation of spurs in a parallelized process
    Thus the overall workflow is divided into two parts: A. generating spines and B. generating spurs

    __A. Generating spines__
    1. Ingest the input gpkg containing geo axes data
    2. "Drop" spines onto a vector plain corresponding to our geo data and within the "working area" defined by the active mask in pyworldsim world object
    3. Select spines matching the procedural requirements (distance_from_axis, angle_dev)
    4. Iterate steps 2 and 3 until we have desired number of spines or we run out of iters (max 99).
    5. Sink the output into a gpkg object (spines.gpkg)

    Inputs for generating spines
    * ctx: context from the world object defining area in which we are operating
    * spine_save_location (str): where to save the produced gpkgs
    * orogeny_axes (gdf): a gpkg object containing orogeny definitions manually determined in QGIS
    * spine_length (tuple - float32): a tuple of min and max spine lengths.
    * n_spines_target (int16): target number of spines beyond which no more are produced
    * n_spines_iter (int16): number of spines generated in any given iteration
    * distance_from_axes (float32): how many CRS units can a spine be away from an orogeny axis to be accepted (if dist_actual > distance_from_axes: reject). We count the closes element of the line segment. Extracted from the orogeny_axes.
    * angle_dev (float32): how many degrees can a spine be deflected from the orogeny axes before they are rejected (if angle_actual > angle_dev: reject). We count relative to the closest axis.
    * max_iters (int8): maximum number of iterations before completing the algorithm regardless of whether n_spines_target has been met or not: this prevents an endless loop.

    Outputs from generating spines
    * spines_and_spurs (gdf):
        * LineSegments representing the spines
        * serial_id (int32): in the format 10000001, where 1.00.00001 represent respective: 1: spine, 00 - iteration (hence maximum for max iters is 99), 00000 - serial number of spine within the iteration (giving us 99999 spines as theoretical maximum generated).
        * oe_epoch (int8): number of the orogeny epoch associated with the given spine (inherited from the orogeny_axis nearest to the spine).
        * magnitude (float32): used to determine the final max height of the highest peaks of the spine. Inherited from the orogeny_axis nearest to the spine.
        * distance_from_axis (float32): distance from the enarest orogeny axis in crs units
        * deviation_from_axis (float32): angle between the nearest orogeny axis and the proposed spine

    Development plan:
    1. Implement and test drop_spines with _initialize_spines and _propose_spines only. Test visualization rasters are generated with burn_to_raster. .gpkg to feed to burn_to_raster is generated with sink_to_gpkg.
    2. Implement _associate_spines_to_axes. Test spines inherit oe_epoch and magnitude (burn_to_raster). Test spines get distances to the nearest orogeny axis (burn_to_raster for visualization). NOTE: in case of a spine being equidistant implement a random coin flip decision between two.
    3. Implement _reject_spines. This will reject any spines that are not within distance and not at the right angle. Spines are removed from the gdf based on whether they meet the filter values. Test by visualizing outcome with burn_to_raster.
    4. Implement _clean_overlaps. This will look for spines that cross one another. In case of spines crossing one another remove the one with the higher serial number. Test by outcome visualization with burn_to_raster.

    __B. generating spurs__
    Depending on the number of spurs getting generated this could be time consuming. Hence we parallelize.
    1. Split the area between different worker-areas. This is taking the GDF with the spines_and_spurs, sub-dividing it into equal rectangles, and splitting the spines and spurs gdf into that many individual gdfs. Where a spine/spur is crossing the border, assign to sequentially earlier gdf. The area can either be split into 1 (for a single worker) or a number of workers divisible by two. 
    2. Each worker executed the following loop:
        2.1 Choose a spine/spur on which to grow a spur. Longer spines/spurs have greater chance of "sprouting".
        2.2 Choose point on that spine/spur
        2.3 Choose angle at which to grow the spur (draw from a uniform distribution within a user selected range in degrees).
        2.4 Draw spur length (to a max defined by the user. Follow a distrib biased towards shorter spurs).
        2.5 Repeat until n_spurs have been drawn
        2.6 Remove overlaps with other spines/spurs - keep the one with the older serial number.
    3. After completing the first loop, repeat, but grow from spurs generated on that first loop. 
    4. Keep repeating on spurs from the previous loops until the number of repetitions have been reached.
    5. Clean up overlaps at edges between worker areas.
    6. Truncate any spurs outside the working area defined in the ctx.

    Inputs for generating spurs:
    * spines (gdf): generated in the drop_spines()
    * spines_and_spurs_save_location (str): location to save the complete spines and spurs gdf.
    * n_workers (int): number of parallel spur grow processes.
    * max_angle (int8): maximum deflection angle from a parent spur (in degrees)
    * spur_inputs_dict (dict): a dictionary containing inputs for each spur level in the format:
        {"iteration_n": (spurs_to grow, length_ranges), "iteration_n+1": ...}, where:
            * spurs_to_grow (int): how many spurs are to be generated in a given iteration
            * length_ranges (tuple): min and max length of spurs.
        Note that this allows to generate different amount of spurs at each generation and control their size (typically more shorter spurs in "youger" generations).

    Outputs from generating_spurs:
    spines_and_spurs (gdf):
        * Line segments representing spines and spurs
        * serial_id (int32): in the format 20000001, where 1.00.00001 represent respective: 2: spine, 00 - iteration (hence maximum for max iters is 99), 00000 - serial number of spine within the iteration (giving us 99999 spines as theoretical maximum generated).
        * oe_epoch (int8): number of the orogeny epoch associated with the given spine (inherited from the parent spine).
        * magnitude (float32): used to determine the final max height of the highest peaks of the spur. Defined by: magnitude_parent_spine / 1 + iteration number
        * distance_from_axis (float32): distance from the enarest orogeny axis in crs units. Set to NULL for spurs. Deleted on sink_to_gpkg.
        * deviation_from_axis (float32): angle between the nearest orogeny axis and the proposed spine. Set to NULL for spurs. Deleted on sink_to_gpkg.
        
    Development plan:
    1. Implement grow_spurs() and _divide_for_workers(). Test that spines are divided into different worker areas by using burn_to_raster tool set to color code the spines in different areas.
    2. Implement _choose_spine_spur, _choose_point, _choose_angle, _draw_spur. Implement with workers. Test on a single iteration first. Then on multiple iterations.
    3. Implement _remove_overlaps and integrate into main workflow.
    4. Implement _remove overlaps at the end of the generation process
    5. Implement _truncate_spines_spurs_outside_working_area.
    
    
    
    """

    def __init__(self, ctx):
        """
        Initialize the class
        """
        pass


    ###########################################################################
    ### Spine generation
    ###########################################################################
    def drop_spines():
        """
        Monte Carlos the spines of the mountain ridges.

        *Refer to class docstring for inputs/outputs*
        """
        spines = _initialize_spines() ### Creates the empty gdf
        current_iter = 0

        while len(spines_and_spurs) < n_spines_target and current_iter < max_iters:
            spines = _propose_spines(spines, spine_length, ctx) ### ctx provides data on where to drop spines.
            spines = _associate_spines_to_axes(spines)
            spines = _reject_spines(spines, distance_from_axes, angle_dev)
            spines = _clean_overlaps(spines)
            current_iter += 1

        sink_to_gpkg(spines, spine_save_location)

    def _initialize_spines():
        """
        Wrapper for creating an empty gdf.
        """
        pass
        
    def _propose_spines():
        """
        """
        pass

    def _associate_spines_to_axes():
        """
        """
        pass

    def _reject_spines():
        """
        """
        pass

    def _clean_overlaps():
        """
        """
        pass


    ###########################################################################
    ### Spine generation
    ###########################################################################
    def grow_spurs():
        """
        """
        spines_and_spurs = gpd.read_...(spines_save_location)  ### We don't overwrite the spines. This way we don't need to keep regenerating spines in dev/test.

        spines_and_spurs = _assign_worker_areas(spines_and_spurs, n_workers)

        for iteration in length(spur_inputs_dict):
            ###RUN THE PARALLEL WORKER PROCESS.

        # spines_and_spurs = TODO: merging of the individual worker GDFs.
        spines_and_spurs = _clean_overlaps(spines_and_spurs)
        spines_and_spurs = _truncate_spines_spurs_outside_working_area(spines_and_spurs)
        sink_to_gpkg(spines, spine_save_location)
        
    ### RELEVANT WORKER FUNCTION DEFS
    """
    Worker goes through the following loop for the number of spurs it is assigned:
    _choose_parent
    _choose_point_on_parent
    _choose_angle_for_spur
    _draw_spur_length

    After each full iteration of the loop (drawing all spurs for a given generation), runs:
    __clean_overlaps

    After 
    """

    def _assign_worker_areas():
        """
        Adds a column indicating which worker gets which set of starting spines.
        Also determines the proportion of total iteration spurs each worker gets to do.
        """
        pass


    def _choose_parent():
        """
        Choose parent spine/spur with probability proportional to length.
        Initiates the spur in the gdf and inherits data from parent.
        """
        pass

    def _choose_point_on_parent():
        """
        Choose a point on the parent where to grow the proposed spur.
        Use "natural data" empirical probabilities derived from gen AI queries.
        """
        pass
        
    def _choose_angle_for_spur():
        """
        Draw an angle from the parent on which to grow the spur.
        First draw from a uniform distribution between +/- angle (in int degrees - reject anything less than 5 degrees)
        Then flip a coin as to the direction the spur grows
        """
        pass

    def _draw_spur_length():
        """
        Draw a random spur length within the ranges provided by the user.
        We follow the ZZZ distribution
        """
        pass

    def _truncate_spines_and_spurs_outside_working_area():
        """
        Cuts all the relevant spines and spurs to fit within the working area defined by the ctx.
        """
        
            





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




