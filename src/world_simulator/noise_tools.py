import numpy as np
import rasterio
import noise
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Dict, Any, Callable

logger = logging.getLogger(__name__)

def generate_fractal_noise(
    template_path: str,
    scale: float = 100.0,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 42,
    mountain_mode: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generates a fractal noise raster based on the dimensions/geotransform 
    of a template raster. Returns the data in-memory.

    :param template_path: Path to an existing raster to define extent/resolution.
    :param scale: Zoom level of noise. Lower = finer, Higher = broader.
    :param octaves: Layers of detail.
    :param seed: Random seed (passed to 'base' in pnoise2).
    :param mountain_mode: If True, uses abs() to create ridgelines.
    :return: Tuple (Numpy Array of noise, Rasterio Profile Dict)
    """
    logger.info(f"Generating Fractal Noise (Seed: {seed}, Scale: {scale}, Octaves: {octaves})...")

    with rasterio.open(template_path) as src:
        # 1. Capture Metadata
        width = src.width
        height = src.height
        profile = src.profile.copy()
        
        # Update profile to float32 (noise is continuous) and single band
        profile.update(dtype=rasterio.float32, count=1, driver='GTiff')

        # 2. Create Empty Array
        noise_array = np.zeros((height, width), dtype=np.float32)

        # 3. Generate Noise (Pixel Loop)
        # Note: 'noise' library is C-optimized, but python looping is the bottleneck.
        # We use tqdm to show progress.
        for y in tqdm(range(height), desc="Noise Generation", leave=False):
            for x in range(width):
                # We use the raw pixel indices (x, y) for generation logic 
                # to maintain consistency with your previous QGIS script.
                raw_val = noise.pnoise2(
                    x / scale,
                    y / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=width,
                    repeaty=height,
                    base=seed
                )

                if mountain_mode:
                    # Billowy/Ridged noise
                    noise_array[y][x] = abs(raw_val)
                else:
                    # Standard Perlin (-1 to 1 normalized to 0 to 1)
                    noise_array[y][x] = (raw_val + 1.0) / 2.0

        # 4. Normalize (Shift to 0 base)
        min_val = np.min(noise_array)
        noise_array = noise_array - min_val
        
        logger.info(f"Noise generated. Range: {np.min(noise_array):.4f} to {np.max(noise_array):.4f}")
        
        return noise_array, profile

def test_noise(
    template_path: str, 
    noise_func: Callable, 
    **kwargs
):
    """
    Test harness for visualizing noise functions without saving to disk.
    
    :param template_path: Path to reference raster.
    :param noise_func: The function to run (e.g. generate_fractal_noise).
    :param kwargs: Parameters to pass to the noise function.
    """
    print(f"--- Testing Noise Function: {noise_func.__name__} ---")
    print(f"Parameters: {kwargs}")

    # Run the function
    arr, profile = noise_func(template_path, **kwargs)

    # Calculate Stats
    print("\n--- Output Statistics ---")
    print(f"Shape: {arr.shape}")
    print(f"Min:   {np.min(arr):.4f}")
    print(f"Max:   {np.max(arr):.4f}")
    print(f"Mean:  {np.mean(arr):.4f}")
    print(f"Std:   {np.std(arr):.4f}")

    # Visualize
    plt.figure(figsize=(10, 8))
    plt.title(f"Noise Preview\nSeed: {kwargs.get('seed', '?')} | Scale: {kwargs.get('scale', '?')}")
    
    # We use a color map that highlights terrain-like features
    img = plt.imshow(arr, cmap='terrain')
    plt.colorbar(img, label="Noise Value")
    plt.axis('off') # Hide pixel coordinates
    plt.show()

    return arr, profile