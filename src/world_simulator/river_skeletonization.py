import geopandas as gpd
from shapely.ops import unary_union
from centerline.geometry import Centerline
from typing import Tuple, Optional
import os

from abc import ABC, abstractmethod
from shapely.geometry.base import BaseGeometry
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm  # specific import for progress bar
from centerline.geometry import Centerline

import momepy
import networkx as nx

import logging
from .logger import setup_logger  # <--- Import your new tool

###############################################################################

logger = logging.getLogger(__name__)



setup_logger(log_name="wfrp_phys_geo")


###############################################################################

def load_and_project(filepath: str, processing_epsg: int = None) -> Tuple[gpd.GeoDataFrame, any]:
    """
    Loads a spatial file. 
    If the file is already in a projected (metric) CRS, it keeps it as is.
    Only reprojects if the data is Geographic (Lat/Lon) or if a specific EPSG is forced.
    """
    logger.info(f"Reading {filepath}...")
    gdf = gpd.read_file(filepath)
    original_crs = gdf.crs

    # CHECK 1: Is it already projected? (Meters/Feet/etc)
    if gdf.crs.is_projected and processing_epsg is None:
        logger.info(f"Data is already projected ({gdf.crs.name}). Keeping original CRS.")
        # We return the same CRS for both to indicate no reprojection needed at the end
        return gdf, original_crs

    # CHECK 2: Is it Geographic? (Degrees)
    # If it is Lat/Lon, we MUST project it, but for a custom planet, 
    # we can't guess the EPSG. 
    if not gdf.crs.is_projected and processing_epsg is None:
        raise ValueError(
            "Input data is in Degrees (Geographic). "
            "Please project it to your custom WFRP metric CRS before running this script, "
            "or provide a specific processing_epsg."
        )

    # CHECK 3: User forced a specific EPSG (Only use if you are sure)
    if processing_epsg is not None and gdf.crs.to_epsg() != processing_epsg:
        logger.info(f"WARNING: Reprojecting to EPSG:{processing_epsg}. "
              "Ensure this EPSG matches your planet's ellipsoid!")
        gdf = gdf.to_crs(epsg=processing_epsg)
        
    return gdf, original_crs

###############################################################################

def bridge_discontinuous_polygons(gdf: gpd.GeoDataFrame, gap_tolerance: float) -> gpd.GeoDataFrame:
    """
    Performs morphological closing (Buffer -> Union -> Negative Buffer)
    to connect adjacent polygons.
    """
    logger.info(f"Bridging gaps with a tolerance of {gap_tolerance} meters...")
    
    # 1. Dilate (Buffer)
    # Buffering creates rounded corners; cap_style=3 (square) can sometimes help 
    # preserve shape, but default (round) is usually best for rivers.
    buffered = gdf.geometry.buffer(gap_tolerance)
    
    # 2. Union (Merge)
    # Compresses all shapes into a single MultiPolygon
    merged_geom = unary_union(buffered)
    
    # 3. Erode (Negative Buffer)
    # Returns the shape to approx original size, but bridges remain
    eroded_geom = merged_geom.buffer(-gap_tolerance)
    
    # 4. Re-structure
    # Create a new GDF and explode MultiPolygons into individual Polygons
    # (e.g., if you have two distinct river systems in the file)
    new_gdf = gpd.GeoDataFrame(geometry=[eroded_geom], crs=gdf.crs)
    new_gdf = new_gdf.explode(index_parts=False).reset_index(drop=True)
    
    logger.info(f"Merged into {len(new_gdf)} continuous river system(s).")
    return new_gdf

###############################################################################

def simplify_geometries(gdf: gpd.GeoDataFrame, tolerance: float) -> gpd.GeoDataFrame:
    """
    Simplifies the geometries to remove 'buffer bloat' (unnecessary vertices).
    This significantly speeds up downstream processing.
    """
    logger.info(f"Simplifying geometries with tolerance {tolerance}...")
    
    # We copy to avoid SettingWithCopy warnings
    cleaned_gdf = gdf.copy()
    
    # .simplify() is vectorised and very fast in GeoPandas/GEOS
    cleaned_gdf['geometry'] = cleaned_gdf.geometry.simplify(tolerance)
    
    return cleaned_gdf

###############################################################################

class BaseSkeletonizer(ABC):
    """
    Abstract Interface for any river skeletonization algorithm.
    """
    @abstractmethod
    def run(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        pass

class VoronoiSkeletonizer(BaseSkeletonizer):
    """
    Implementation using the 'centerline' library (Voronoi Method).
    """
    def __init__(self, interpolation_distance: float):
        self.interp_dist = interpolation_distance

    @staticmethod
    def _worker_logic(geom: BaseGeometry, dist: float) -> BaseGeometry:
        """
        Pure static method for parallel execution. 
        Must not reference 'self' to ensure pickle safety.
        """
        if geom is None or geom.is_empty:
            return None
        try:
            cl = Centerline(geom, interpolation_distance=dist)
            return cl.geometry
        except Exception:
            return None

    def run(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        n_cores = max(1, cpu_count() - 1)
        logger.info(f"Skeletonizing using Voronoi Method on {n_cores} cores...")
        logger.info(f"Vertex Density: {self.interp_dist}m")

        geometries = gdf.geometry.tolist()
        
        # Create partial function to pass the distance setting
        worker_func = partial(self._worker_logic, dist=self.interp_dist)
        
        results = []
        
        with Pool(processes=n_cores) as pool:
            # imap_unordered for responsive progress bar
            iterator = pool.imap_unordered(worker_func, geometries, chunksize=1)
            
            for result in tqdm(iterator, total=len(geometries), desc="Processing Rivers"):
                if result is not None:
                    results.append(result)

        return gpd.GeoDataFrame(geometry=results, crs=gdf.crs)

###############################################################################



def prune_short_tributaries(gdf: gpd.GeoDataFrame, min_length: float, max_iters: int = 100) -> gpd.GeoDataFrame:
    """
    Strategy 1: Recursive Leaf Pruning (The 'Haircut').
    
    1. Converts the GeoDataFrame lines into a NetworkX Graph.
    2. Identifies 'leaf' nodes (endpoints with only 1 connection).
    3. If the edge connecting to a leaf is shorter than 'min_length', it is pruned.
    4. REPEATS this process recursively (so if a short branch had a tiny branch 
       of its own, both are removed).
       
    :param gdf: GeoDataFrame containing river centerlines (LineStrings)
    :param min_length: Threshold in meters. Anything shorter is cut.
    """
    logger.info(f"Pruning tributaries shorter than {min_length}m (Max iterations: {max_iters})...")
    
    # 1. Convert to Graph
    exploded_gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    G = momepy.gdf_to_nx(exploded_gdf, approach='primal')
    
    # 2. Recursive Pruning Loop
    clean_pass = 0
    
    while clean_pass < max_iters:
        # Find leaves (degree 1)
        leaves = [node for node, degree in G.degree() if degree == 1]
        edges_to_remove = []
        
        for leaf in leaves:
            # Handle isolated nodes (degree 0) just in case
            neighbors = list(G.neighbors(leaf))
            if not neighbors: continue
                
            neighbor = neighbors[0]
            
            # Access edge data (handling MultiGraph potential)
            # We take the first edge found between these nodes
            edge_data = G.get_edge_data(leaf, neighbor)
            key = list(edge_data.keys())[0] # Usually 0
            attributes = edge_data[key]
            
            geom = attributes.get('geometry')
            length = geom.length if geom else 0
            
            if length < min_length:
                edges_to_remove.append((leaf, neighbor))
        
        # Exit condition: No short leaves found
        if not edges_to_remove:
            logger.info(f"Pruning converged after {clean_pass} passes.")
            break
            
        # Remove edges
        G.remove_edges_from(edges_to_remove)
        G.remove_nodes_from(list(nx.isolates(G)))
        
        clean_pass += 1
        logger.debug(f"Pass {clean_pass}: Removed {len(edges_to_remove)} segments.")

    if clean_pass >= max_iters:
        logger.warning(f"Pruning hit max_iters ({max_iters})! Some artifacts may remain.")

    # 3. Reconstruct
    logger.info("Reconstructing river network...")
    clean_gdf = momepy.nx_to_gdf(G, points=False, lines=True)
    clean_gdf.set_crs(gdf.crs, inplace=True)
    
    return clean_gdf

###############################################################################

def save_results(gdf: gpd.GeoDataFrame, output_path: str, target_crs: any):
    """
    Reprojects data back to original CRS and saves to GPKG.
    """
    print("Reprojecting and saving output...")
    
    # Reproject if necessary
    if target_crs and gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
        
    gdf.to_file(output_path, driver="GPKG")
    print(f"Successfully saved to: {output_path}")

###############################################################################

def run_river_skeleton_pipeline(input_file: str, output_file: str, 
                       gap_size: float, interp_dist: float, 
                       prune_threshold: float = 5000, 
                       prune_iterations: int = 100): # e.g. 2km pruning
    
    # 1. Load
    gdf, orig_crs = load_and_project(input_file, processing_epsg=None)
    
    # 2. Connect
    connected_gdf = bridge_discontinuous_polygons(gdf, gap_tolerance=gap_size)
    
    # 3. Simplify (Pre-Skeletonization)
    cleaned_gdf = simplify_geometries(connected_gdf, tolerance=interp_dist / 2)
    
    # 4. Skeletonize
    skeletonizer = VoronoiSkeletonizer(interpolation_distance=interp_dist)
    lines_gdf = skeletonizer.run(cleaned_gdf)
    
    # 5. NEW STEP: Prune The Hair (The Haircut)
    clean_gdf = prune_short_tributaries(lines_gdf, 
                                        min_length=prune_threshold,
                                        max_iters=prune_iterations)
    # ---------------------------------------------------------
    
    # 6. Save (Save the PRUNED version)
    save_results(clean_gdf, output_file, orig_crs)