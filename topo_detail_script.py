# --- SCRIPT TO COMBINE TOPOGRAPHY, CARVING, AND FRACTAL NOISE ---

from qgis.core import QgsRasterLayer, QgsProject
import numpy as np
from osgeo import gdal, osr
import tempfile

# --- PARAMETERS ---
# 1. Base Layer Name (must be loaded in QGIS)
TOPO_BASE_NAME = "wfrp_empire_topo_high_res"
OUTPUT_LAYER_NAME = "wfrp_empire_topo_detail"

# --- LAYER NAMES ---
LAKES = "wfrp_empire_lakes_high_res"
RIVERS_MINOR = "wfrp_empire_rivers_high_res_bw"
RIVERS_MAJOR = "wfrp_empire_rivers_main_high_res"
NORTH_BLOB = "wfrp_empire_north_blob_high_res"

N1000 = "fractal_noise_1000"
N500 = "fractal_noise_500"
N100 = "fractal_noise_100"
N50 = "fractal_noise_50"

# --- MULTIPLIERS ---
CARVE_DEPTH = 0.0
MULT_1000 = 300.0
MULT_500 = 300.0
MULT_100 = 1600.0
MULT_50 = 1600.0
HALF_MULT_FACTOR = 0.5 

# --- NEW: NODATA SETTING ---
OUTPUT_NODATA = -9999.0

# -----------------------------------------------------------------
# --- FUNCTION TO READ AND CONVERT LAYER TO NUMPY ARRAY ---
def get_safe_array(layer_name, extent, width, height, fill_value=0):
    """
    Reads QGIS layer data based on the extent/size of the base topography, 
    converts to NumPy array, and replaces NoData with fill_value.
    """
    layer = QgsProject.instance().mapLayersByName(layer_name)
    if not layer:
        raise ValueError(f"Error: Layer '{layer_name}' not found in QGIS project.")
    
    provider = layer[0].dataProvider()
    
    # Arguments passed positionally (band, extent, width, height)
    block = provider.block(
        1,  # Band number
        extent, 
        width, 
        height
    )
    
    if block.isValid():
        # FIX: Convert QByteArray to NumPy array using frombuffer
        # We also need the GDT (GeoData Type) to correctly interpret the bytes
        gdt = provider.dataType(1) 
        np_dtype = gdal.GetDataTypeName(gdt).lower()
        
        # Read the raw byte data
        raw_data = block.data()
        
        # Convert the byte data into a NumPy array with the correct dtype
        arr = np.frombuffer(raw_data, dtype=np_dtype).reshape(height, width)
        
        # Get NoData value from the layer
        no_data_val = provider.sourceNoDataValue(1)
        
        # Replace NoData with the specified fill_value (0 for masks/noise)
        if no_data_val is not None and np.issubdtype(arr.dtype, np.number):
            arr[arr == no_data_val] = fill_value
        
        # Return the processed array
        return arr, layer[0].crs()
    
    raise IOError(f"Error reading raster block for layer: {layer_name}")

# -----------------------------------------------------------------
# --- MAIN EXECUTION ---

# 1. Load the Base Topo (Read its properties and array)
print("Loading Base Topography...")
topo_base_layer = QgsProject.instance().mapLayersByName(TOPO_BASE_NAME)[0]
topo_base_ds = gdal.Open(topo_base_layer.source(), gdal.GA_ReadOnly)
geotransform = topo_base_ds.GetGeoTransform()
width = topo_base_ds.RasterXSize
height = topo_base_ds.RasterYSize
crs = topo_base_layer.crs()
topo_base_array = topo_base_ds.GetRasterBand(1).ReadAsArray()
topo_base_ds = None # Close dataset

# Store the reference grid parameters used by the new get_safe_array function
ref_extent = topo_base_layer.extent()
ref_width = width
ref_height = height

# 2. Load all other layers, replacing NoData with 0.0 for masks/noise
# Define a WRAPPER function that correctly passes the reference grid
def load_aux_layer(name):
    # This now correctly passes the required grid parameters
    return get_safe_array(name, ref_extent, ref_width, ref_height, fill_value=0.0)[0] 

# Load auxiliary layers
L = load_aux_layer(LAKES)
R = load_aux_layer(RIVERS_MINOR)
M = load_aux_layer(RIVERS_MAJOR)
NORTH = load_aux_layer(NORTH_BLOB)
N1000 = load_aux_layer(N1000)
N500 = load_aux_layer(N500)
N100 = load_aux_layer(N100)
N50 = load_aux_layer(N50)

# Ensure all arrays have the same shape as the base topo
assert topo_base_array.shape == L.shape, "Array shapes do not match!"

# 3. CORE CALCULATION using NUMPY: Perform the full equation
print("Performing core calculation...")

# --- A. INITIAL TOPOGRAPHY AND CARVING ---

# Start with the Base Topo
final_topo = topo_base_array.astype(np.float32)

# Carve Minor Rivers (R > 0)
final_topo[R > 0] -= CARVE_DEPTH

# Carve Major Rivers (M > 0)
final_topo[M > 0] -= CARVE_DEPTH


# --- B. NOISE MASKING AND ADDITION ---

# Define the CLEAN LAND MASK (1 where NO lake, NO minor river, NO major river)
land_mask = (L < 0) & (R < 0) & (M < 0)

# 1. Base Noise (N1000) - Apply everywhere on clean land
noise_total = N1000 * MULT_1000

# 2. N100 (Fine Noise) at 1000+
# (Altitude > 1000) * N100 * 800
altitude_mask_1000 = final_topo > 1000
noise_total += altitude_mask_1000 * N100 * MULT_100

# 3. N500 (Medium Noise) at 500+ OR North
# (Altitude > 500 OR North > 0) * N500 * 400
altitude_mask_500_or_north = (final_topo > 500) | (NORTH > 0)
noise_total += altitude_mask_500_or_north * N500 * MULT_500

# 4. N50 (Finer Noise) at 2000+
# (Altitude > 2000) * N50 * 1600
altitude_mask_2000 = final_topo > 2000
noise_total += altitude_mask_2000 * N50 * MULT_50


# Apply the TOTAL noise only where the LAND MASK is TRUE (1)
final_topo += noise_total * land_mask


# --- C. HALF NOISE FOR MAJOR RIVER VALLEYS ---

# 1. Define the TERRACE MASK: In the Major River Valley (M > 0) AND NOT on the Minor River Channel (R=0)
# We assume the deepest/flattest part is defined by R and M having values.
# If R and M have non-zero values where the river runs, R==0 and M==0 will define land.
# To identify the TERRACES (the carved M area, but not the R carved area):
terrace_mask = (M > 0) & (R < 0)


# Add half of the Base Noise (N1000) only to the TERRACE MASK
final_topo[terrace_mask] += (N1000[terrace_mask] * MULT_1000) * HALF_MULT_FACTOR

# --- D. FINAL CLEANUP: Convert negative values to NoData ---
print("Converting negative values to NoData...")

# Use NumPy to find all values less than -200 and replace them with the defined NoData constant
# The -200 means that anything that accidentally became a depression, gets retained
final_topo[final_topo < -200] = OUTPUT_NODATA



# 4. WRITE THE OUTPUT RASTER
print("Writing final raster...")

output_path = tempfile.gettempdir() + f'/{OUTPUT_LAYER_NAME}.tif'
driver = gdal.GetDriverByName('GTiff')
output_raster = driver.Create(output_path, width, height, 1, gdal.GDT_Float32)

# Set georeferencing information
output_raster.SetGeoTransform(geotransform)
srs = osr.SpatialReference()
srs.ImportFromWkt(crs.toWkt())
output_raster.SetProjection(srs.ExportToWkt())

# Write the final array to the raster band
output_raster.GetRasterBand(1).WriteArray(final_topo)
output_raster.GetRasterBand(1).SetNoDataValue(OUTPUT_NODATA)
output_raster.FlushCache()
output_raster = None 

# 5. ADD TO QGIS PROJECT
result_layer = QgsRasterLayer(output_path, OUTPUT_LAYER_NAME, 'gdal')
if result_layer.isValid():
    QgsProject.instance().addMapLayer(result_layer)
    print(f"\nSUCCESS! Loaded layer: {OUTPUT_LAYER_NAME}")
else:
    print("Error: Could not create the final raster layer.")

# --- END OF SCRIPT ---