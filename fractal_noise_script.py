# --- SCRIPT TO GENERATE FRACTAL NOISE RASTER ---

from qgis.core import QgsRasterLayer, QgsRasterDataProvider, QgsProject
import numpy as np
import noise # This is the library you installed
from osgeo import gdal
import tempfile

# --- PARAMETERS TO CHANGE ---

# 1. Name of your template DEM layer in the QGIS Layers Panel
template_layer_name = "04_topo_smooth" 

# 2. Name for the new output layer
output_layer_name = "fractal_noise_250"

# 3. Noise parameters
scale = 250.0  # Lower number = finer noise; Higher = broader features
octaves = 6           # Number of detail layers (6-8 is good)
persistence = 0.5     # How much smaller details matter
lacunarity = 2.0      # Frequency of detail layers

# --- END OF PARAMETERS ---


# Get the template layer from the QGIS project
template_layer = QgsProject.instance().mapLayersByName(template_layer_name)[0]

# Get raster properties from the template
extent = template_layer.extent()
width = template_layer.width()
height = template_layer.height()
crs = template_layer.crs()

# Create an empty numpy array to hold the noise values
noise_array = np.zeros((height, width))

# Generate the fractal (Perlin) noise
for y in range(height):
    for x in range(width):
        # Calculate noise value for each pixel
        noise_array[y][x] = noise.pnoise2(x / scale, 
                                          y / scale, 
                                          octaves=octaves, 
                                          persistence=persistence, 
                                          lacunarity=lacunarity, 
                                          repeatx=width, 
                                          repeaty=height, 
                                          base=0)

print("Noise generation complete. Creating raster layer...")

# Create the output raster file path
# This saves the file in a temporary directory
# Create a proper temporary file path on your disk
output_path = tempfile.gettempdir() + f'/{output_layer_name}.tif'

# Create a new raster layer from the numpy array
driver = gdal.GetDriverByName('GTiff')
output_raster = driver.Create(output_path, width, height, 1, gdal.GDT_Float32)

# Set georeferencing information from the template
# Open the template layer's data source with GDAL to get the geotransform
source_ds = gdal.Open(template_layer.source(), gdal.GA_ReadOnly)
output_raster.SetGeoTransform(source_ds.GetGeoTransform())

output_raster.SetProjection(crs.toWkt())

# Write the noise array to the raster band
output_raster.GetRasterBand(1).WriteArray(noise_array)
output_raster.FlushCache()
output_raster = None # Close the file

# Add the new raster to the QGIS project
result_layer = QgsRasterLayer(output_path, output_layer_name, 'gdal')
if result_layer.isValid():
    QgsProject.instance().addMapLayer(result_layer)
    print(f"Successfully created and loaded layer: {output_layer_name}")
else:
    print("Error: Could not create the raster layer.")