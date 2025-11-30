# Imports necessary QGIS and Processing modules
from qgis.core import QgsProject
import processing

# --- USER: PLEASE EDIT THIS LINE ---
# Enter the exact name of your split river polygon layer as it appears in the Layers Panel
INPUT_LAYER_NAME = "927_reivers_poly_gridded"
# -----------------------------------

# Define the path for the final output file
# You can change this path if you want to save the file somewhere else
project_path = QgsProject.instance().homePath()
FINAL_OUTPUT_PATH = f'{project_path}/physical_rasters/929_rivers_centerline.gpkg'

# Find the input layer in the current project
input_layer = QgsProject.instance().mapLayersByName(INPUT_LAYER_NAME)[0]

# Check if the layer was found
if not input_layer:
    print(f"Error: Layer '{INPUT_LAYER_NAME}' not found. Please check the name and try again.")
else:
    print("Step 1: Converting polygons to boundary lines using GRASS...")
    lines = processing.run("grass7:v.to.lines", {
        'input': input_layer,
        'type': 2,  # 2 corresponds to 'boundary'
        'output': 'memory:'
    })['output']

    print("Step 2: Extracting vertices from lines...")
    vertices = processing.run("native:extractvertices", {
        'INPUT': lines,
        'OUTPUT': 'memory:'
    })['OUTPUT']

    print("Step 3: Creating Voronoi diagram (this may take a moment)...")
    voronoi = processing.run("grass7:v.voronoi", {
        'input': vertices,
        'output': 'memory:'
    })['output']

    print("Step 4: Clipping Voronoi diagram to create centerlines...")
    centerlines = processing.run("native:clip", {
        'INPUT': voronoi,
        'OVERLAY': input_layer,
        'OUTPUT': FINAL_OUTPUT_PATH
    })

    print(f"Success! Final centerline layer created at: {FINAL_OUTPUT_PATH}")
    
    # Add the final result to the map
    iface.addVectorLayer(FINAL_OUTPUT_PATH, "Final Centerlines", "ogr")