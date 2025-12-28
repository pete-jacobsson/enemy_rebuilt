import os
import numpy as np
import rasterio
import geopandas as gpd
from pathlib import Path

class WorldState:
    """
    The central mutable context for the geological simulation.
    
    Acts as a container for all spatially referenced arrays (Elevation, Lithology, Masks).
    Manages the 'Shared Memory' architecture where distinct generator classes 
    modify these arrays in-place to simulate geological history.
    
    Attributes:
        config (dict): Global configuration dictionary containing paths and tuning parameters.
        shape (tuple): The (height, width) of the simulation grid (e.g., 2600, 4000).
        transform (Affine): The affine transform for georeferencing (500m pixel size).
        crs (CRS): Coordinate Reference System (e.g., EPSG:3035).
        
        # Primary Layers
        elevation (np.ndarray): Float32. The heightmap in meters. 
                                Kept as Float32 to support sub-meter erosion physics.
        lithology_id (np.ndarray): Int16. Categorical ID of the bedrock material.
                                   Maps to physical properties (Hardness, Permeability) via lookup.
                                   Examples: 10=Sediment, 20=Granite, 30=Fault, 99=Sugar Sea.
        active_mask (np.ndarray): Int16. Simulation bounds.
                                  0 = Void (Ignored), 1 = Active (Physics), 2 = Sink (Boundary).
        water_flux (np.ndarray): Int16. Accumulated flow count per pixel. 
                                 Used to determine river width/power.
        
        # Transient Layers
        layers (dict): Storage for temporary or derived layers (e.g., 'sediment_depth', 'attractor_field').
    """

    def __init__(self, config):
        """
        Initialize the WorldState using the configuration dictionary.
        
        This reads the 'template_tif' to establish the Affine Transform, CRS, 
        and grid dimensions (Height/Width) for the simulation.
        
        Args:
            config (dict): The master configuration dictionary.
        """
        self.config = config
        self.layers = {}  # Dictionary to hold dynamic/temp layers
        self.vectors = {} # Dictionary to hold loaded GeoDataFrames
        
        # Load spatial metadata from the template raster
        template_path = config["paths"]["template_tif"]
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template raster not found at: {template_path}")
            
        with rasterio.open(template_path) as src:
            self.shape = src.shape  # (height, width)
            self.transform = src.transform
            self.crs = src.crs
            
        # Allocate Primary Arrays
        # Elevation: Float32 is required for sub-meter hydraulic precision
        self.elevation = np.zeros(self.shape, dtype=np.float32)
        
        # Lithology: Int16 for categorical IDs (10=Sediment, 20=Granite, etc.)
        self.lithology_id = np.full(self.shape, fill_value=10, dtype=np.int16)
        
        # Active Mask: Int16 (0=Void, 1=Active, 2=Sink)
        self.active_mask = np.zeros(self.shape, dtype=np.int16)
        
        # Water Flux: Int16 (Accumulated flow count, managed by hydro engines)
        self.water_flux = np.zeros(self.shape, dtype=np.int16)

        print(f"WorldState initialized. Grid shape: {self.shape}. CRS: {self.crs}")

    
    def register_layer(self, name, dtype=np.int16, fill_value=0):
        """
        Dynamically allocates a new spatial array in the self.layers dictionary.
        Useful for transient data like 'sediment_depth' or 'attractor_field'.
        
        Args:
            name (str): Key for the layer (e.g., 'sediment').
            dtype (type): Numpy data type (Default: np.int16 for memory efficiency).
            fill_value (int/float): Initial value for the array.
        """
        # Create array with the global shape
        new_layer = np.full(self.shape, fill_value, dtype=dtype)
        self.layers[name] = new_layer
        print(f"Registered layer '{name}' with dtype {dtype} and fill {fill_value}.")

    
    def load_vectors(self):
        """
        Loads the required static vector files defined in config['vectors'].
        Stores them in self.vectors dict.
        """
        vector_paths = self.config["paths"]["vectors"]
        base_dir = self.config["paths"]["inputs"]
        
        print("Loading vector inputs...")
        for key, filename in vector_paths.items():
            full_path = os.path.join(base_dir, filename)
            
            if os.path.exists(full_path):
                # Load with Geopandas
                gdf = gpd.read_file(full_path)

                # --- FIX START: CRS Safety Check ---
                if gdf.crs is None:
                    print(f"  ! WARNING: {key} has NO defined CRS. Assuming match with simulation ({self.crs}).")
                    gdf.set_crs(self.crs, allow_override=True, inplace=True)
                # --- FIX END ---
                
                # Ensure CRS matches the simulation grid
                if gdf.crs != self.crs:
                    print(f"  - Reprojecting {key} to {self.crs}")
                    gdf = gdf.to_crs(self.crs)
                
                self.vectors[key] = gdf
                print(f"  - Loaded {key}: {len(gdf)} features.")
            else:
                print(f"  ! WARNING: Vector file not found: {full_path}")

    
    def save_state(self, filepath):
        """
        Serializes the current state (Elevation + Lithology + Active Layers) 
        to a compressed .npz file.
        
        Renamed from 'save_checkpoint' to match the 'load_or_run' interface.
        
        Args:
            filepath (str): The full destination path for the .npz file.
        """
        # Ensure the directory exists (just in case)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Gather all arrays into a dictionary
        # We explicitly save the primary attributes
        arrays_to_save = {
            "elevation": self.elevation,
            "lithology_id": self.lithology_id,
            "active_mask": self.active_mask,
            "water_flux": self.water_flux
        }
        
        # Add any dynamic layers
        arrays_to_save.update(self.layers)
        
        print(f"Saving state to {filepath}...")
        np.savez_compressed(filepath, **arrays_to_save)
        print("  > Save complete.")

    
    def load_state(self, filepath):
        """
        Restores the simulation state from a checkpoint file.
        
        Args:
            filepath (str): Full path to the .npz file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
            
        print(f"Loading state from {filepath}...")
        data = np.load(filepath)
        
        # Restore Primary Arrays
        if "elevation" in data: self.elevation = data["elevation"]
        if "lithology_id" in data: self.lithology_id = data["lithology_id"]
        if "active_mask" in data: self.active_mask = data["active_mask"]
        if "water_flux" in data: self.water_flux = data["water_flux"]
        
        # Restore Dynamic Layers
        # We iterate over keys in the file that aren't primary attributes
        primary_keys = {"elevation", "lithology_id", "active_mask", "water_flux"}
        for key in data.files:
            if key not in primary_keys:
                self.layers[key] = data[key]
                
        print("  > State loaded.")