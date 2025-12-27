import numpy as np
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
        Initialize the WorldState.
        
        Args:
            config (dict): The master configuration dictionary.
        """
        pass

    def register_layer(self, name, dtype=np.int16, fill_value=0):
        """
        Dynamically allocates a new spatial array in the self.layers dictionary.
        
        Args:
            name (str): Key for the layer.
            dtype (type): Numpy data type (Default: np.int16 for memory efficiency).
            fill_value (int/float): Initial value for the array.
        """
        pass

    def load_vectors(self):
        """
        Loads the required static vector files defined in config['vectors'].
        (Sea, Bounds, Rivers, Orogeny Axes).
        """
        pass

    def save_checkpoint(self, name):
        """
        Serializes the current state (Elevation + Lithology + active layers) to a compressed .npz file.
        
        Args:
            name (str): Filename identifier (e.g., '01_initialization').
        """
        pass

    def load_state(self, filepath):
        """
        Restores the simulation state from a checkpoint file.
        """
        pass