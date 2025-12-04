from dataclasses import dataclass
import geopandas as gpd
import logging

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import momepy
import noise

import rasterio
from rasterio import features

import networkx as nx
import numpy as np
import pandas as pd

from scipy import ndimage
import shapely
from shapely.geometry import Point, LineString

from tqdm import tqdm
from typing import List, Optional, Union, Dict, Any


# Ensure we have a logger
logger = logging.getLogger(__name__)
from .logger import setup_logger
setup_logger(log_name="wfrp_phys_geo")


###############################################################################
###############################################################################
###############################################################################

class HydrologyAnalyzer:
    """
    v2.0: Physical Interface Approach.
    
    Strategy:
    1. Clip Rivers by Sea/Lakes (Remove overlaps).
    2. Walk Upstream from Sea Interfaces.
    3. Bridge across Lakes.
    4. Walk Upstream from Lake Inlets.
    5. Discard anything not connected to the sea.
    """
    
    def __init__(self, river_gdf: gpd.GeoDataFrame, sea_gdf: gpd.GeoDataFrame, lake_gdf: gpd.GeoDataFrame):
        self.raw_rivers = river_gdf
        self.sea = sea_gdf
        self.lakes = lake_gdf
        
        # State
        self.clean_rivers = None
        self.G = None           # Undirected
        self.DiG = None         # Directed
        self.visited = set()    # Nodes confirmed connected to sea
        
        # Interfaces
        self.sea_nodes = set()
        self.lake_nodes = {}    # {node_id: lake_index}
        
    def run(self) -> gpd.GeoDataFrame:
        logger.info("--- Phase 1: Hydrology v2.0 (Physical Clip Strategy) ---")
        
        # 1. Physical Cleanup
        self._preprocess_geometry()
        self._build_graph()
        
        # 2. Identify Interfaces
        # self._identify_interfaces()
        
        # 3. Primary Walk (Sea -> Land)
        self._propagate_flow_from_sea()
        
        # 4. Secondary Walk (Lake -> Land)
        self._propagate_flow_from_lakes()
        
        # 5. Reconstruct
        return self._reconstruct_geodataframe()

    def _preprocess_geometry(self):
        """
        Step 1: FILTERING Strategy (The 'Leg in the Sea').
        
        Logic:
        1. Identify segments COMPLETELY inside Sea/Lakes -> Delete them (Noise).
        2. Identify segments PARTIALLY inside Sea/Lakes -> Keep them (The Anchor Legs).
        3. Identify segments COMPLETELY outside -> Keep them (Inland Rivers).
        """
        logger.info("Step 1: Filtering geometry (Removing fully submerged segments)...")
        
        # 1. Prepare Water Polygon (Sea + Lakes)
        # We assume sea_gdf and lake_gdf are Polygons.
        # We explicitly specificy the columns to avoid index collision errors in sjoin
        sea_geom = self.sea[['geometry']].copy()
        lake_geom = self.lakes[['geometry']].copy()
        water = pd.concat([sea_geom, lake_geom], ignore_index=True)
        
        # 2. Identify "Fully Submerged" Segments (Noise)
        # Predicate 'within': The river segment is 100% inside the water.
        fully_inside_indices = []
        
        # processing in chunks or using sjoin is faster than iterating
        inside_matches = gpd.sjoin(self.raw_rivers, water, how='inner', predicate='within')
        fully_inside_indices = inside_matches.index.unique()
        
        logger.info(f"  Found {len(fully_inside_indices)} segments fully inside water (to be removed).")
        
        # 3. Filter the Dataframe
        # We keep everything that is NOT in the 'fully_inside' list.
        # This preserves the "Legs" (partially inside) and "Inland" (fully outside).
        self.clean_rivers = self.raw_rivers.drop(fully_inside_indices).reset_index(drop=True)
        
        logger.info(f"  Filtering complete. {len(self.raw_rivers)} -> {len(self.clean_rivers)} segments.")
        
        # 4. Identify The "Legs" (Interface Segments) for Phase 2
        # Now we look for segments that INTERSECT the sea in the *cleaned* dataset.
        # These are our starting anchors.
        self.sea_legs_indices = []
        
        # Using sjoin again on the clean set
        sea_intersects = gpd.sjoin(self.clean_rivers, self.sea[['geometry']], how='inner', predicate='intersects')
        self.sea_legs_indices = sea_intersects.index.unique()
        
        logger.info(f"  Identified {len(self.sea_legs_indices)} 'Legs in the Sea' (Anchor Segments).")

    def _build_graph(self):
        """Builds the undirected graph from the cleaned lines."""
        logger.info("  Building topology from clipped vectors...")
        self.G = momepy.gdf_to_nx(self.clean_rivers, approach='primal')

    def _orient_anchor_segment(self, line_geom, water_geom):
        """
        Helper: Given a LineString and a Water Polygon (Sea/Lake), 
        determine which end is the sink (in water) and which is the source (on land).
        
        Returns:
            (source_node, target_node) -> The Flow Direction (Land -> Water)
            OR
            None (if ambiguous/error)
        """
        start_pt = Point(line_geom.coords[0])
        end_pt = Point(line_geom.coords[-1])
        
        # Check intersection with water
        # Note: We use 'intersects' because the point might be exactly on the boundary
        # due to the previous cleaning step, or slightly inside.
        start_in_water = start_pt.intersects(water_geom)
        end_in_water = end_pt.intersects(water_geom)
        
        if start_in_water and not end_in_water:
            # Start is Sink. Flow is End(Land) -> Start(Water).
            return (line_geom.coords[-1], line_geom.coords[0])
            
        elif end_in_water and not start_in_water:
            # End is Sink. Flow is Start(Land) -> End(Water).
            return (line_geom.coords[0], line_geom.coords[-1])
            
        else:
            # Ambiguous case: Both in water (should be filtered) or Both on land (touching edge).
            # In 'Leg in the Sea' strategy, valid anchors usually have one clearly inside/touching.
            return None

    def _propagate_flow_from_sea(self):
        """
        Step 3: Anchor Strategy - Primary Walk.
        
        1. Find river segments that intersect the Sea.
        2. Orient them (Land -> Sea).
        3. Walk upstream from the Land Node.
        4. Stop if we hit a Lake.
        """
        logger.info("Step 3: Walking upstream from Sea Anchors...")
        
        self.DiG = nx.DiGraph()
        self.visited = set()
        queue = []
        
        # --- A. Identify Sea Anchors ---
        # Find lines touching the sea
        # We need the geometry of the sea to check endpoints
        sea_union = self.sea.unary_union
        
        # Sjoin to find candidate segments
        candidates = gpd.sjoin(self.clean_rivers, self.sea[['geometry']], how='inner', predicate='intersects')
        
        logger.info(f"  Found {len(candidates)} candidate anchor segments connecting to Sea.")

        # --- B. Initialize Queue from Anchors ---
        for idx, row in candidates.iterrows():
            geom = row.geometry
            
            # Use Helper to figure out direction
            orientation = self._orient_anchor_segment(geom, sea_union)
            
            if orientation:
                source, target = orientation # Source is Land, Target is Sea
                
                # Add this first edge to the Directed Graph
                # We define flow as Land -> Sea (Source -> Target)
                # Find the graph edge key
                if self.G.has_edge(source, target):
                    edge_data = self.G.get_edge_data(source, target)
                    base_data = edge_data[0] if 0 in edge_data else edge_data
                    self.DiG.add_edge(source, target, **base_data)
                    
                    self.visited.add(target) # The sea node is 'done'
                    self.visited.add(source) # The land node is visited
                    
                    # Seed the queue with the LAND node to walk upstream
                    queue.append(source)

        logger.info(f"  Seeded queue with {len(queue)} coastal nodes. Starting upstream walk...")

        # --- C. BFS Upstream ---
        self._run_upstream_bfs(queue, stop_at_lakes=True)


    def _propagate_flow_from_lakes(self):
        """
        Step 4: Anchor Strategy - Secondary Walk.
        
        1. Identify Lakes that are 'Active' (connected to the sea).
        2. Find river segments intersecting those lakes.
        3. Orient them (Land -> Lake).
        4. Walk upstream.
        """
        logger.info("Step 4: Walking upstream from Connected Lakes...")
        
        # 1. Identify Lake Nodes to define 'Active' Lakes
        # We need a quick lookup of which nodes touch which lakes
        node_points = [Point(n) for n in self.G.nodes]
        node_gdf = gpd.GeoDataFrame({'node_id': list(self.G.nodes)}, geometry=node_points, crs=self.raw_rivers.crs)
        lake_nodes_gdf = gpd.sjoin(node_gdf, self.lakes, how='inner', predicate='intersects')
        
        # 2. Find Connected Lakes
        # A lake is connected if ANY of its boundary nodes were visited in the Sea Walk
        connected_lake_indices = set()
        
        for _, row in lake_nodes_gdf.iterrows():
            if row['node_id'] in self.visited:
                connected_lake_indices.add(row['index_right'])
                
        if not connected_lake_indices:
            logger.warning("  No lakes found connected to the sea network.")
            return

        logger.info(f"  Found {len(connected_lake_indices)} lakes connected to the main network.")
        
        # 3. Filter Lakes to only the connected ones
        active_lakes = self.lakes.iloc[list(connected_lake_indices)]
        lakes_union = active_lakes.unary_union
        
        # 4. Find Anchors (Rivers flowing INTO these lakes)
        # We look for unvisited rivers touching these lakes
        candidates = gpd.sjoin(self.clean_rivers, active_lakes[['geometry']], how='inner', predicate='intersects')
        
        queue = []
        
        for idx, row in candidates.iterrows():
            geom = row.geometry
            
            # Optimization: Skip if we already oriented this line (e.g. the outlet)
            # We check endpoints against visited
            u, v = geom.coords[0], geom.coords[-1]
            if u in self.visited and v in self.visited:
                continue

            # Orient: Land -> Lake
            orientation = self._orient_anchor_segment(geom, lakes_union)
            
            if orientation:
                source, target = orientation # Source is Land, Target is Lake
                
                # Check if we already visited the Land Source (loop/redundant)
                if source in self.visited: 
                    continue
                    
                if self.G.has_edge(source, target):
                    edge_data = self.G.get_edge_data(source, target)
                    base_data = edge_data[0] if 0 in edge_data else edge_data
                    self.DiG.add_edge(source, target, **base_data)
                    
                    self.visited.add(target) 
                    self.visited.add(source)
                    queue.append(source)
                    
        logger.info(f"  Seeded queue with {len(queue)} lake-inlet nodes.")
        
        # 5. BFS Upstream (No need to stop at lakes anymore, or maybe we do for nested lakes?)
        # For simplicity, we don't stop at nested lakes in this pass, or we recurse. 
        # Let's set stop_at_lakes=True if you have chains of lakes (Lake -> River -> Lake).
        self._run_upstream_bfs(queue, stop_at_lakes=True)


    def _run_upstream_bfs(self, queue, stop_at_lakes=False):
        """
        Shared logic for walking upstream from a set of starting nodes.
        """
        # Pre-calculate lake node set for fast lookup if stopping
        lake_stop_set = set()
        if stop_at_lakes:
            # Get all nodes touching ANY lake
            node_points = [Point(n) for n in self.G.nodes]
            node_gdf = gpd.GeoDataFrame({'node_id': list(self.G.nodes)}, geometry=node_points, crs=self.raw_rivers.crs)
            matches = gpd.sjoin(node_gdf, self.lakes, how='inner', predicate='intersects')
            lake_stop_set = set(matches.node_id.tolist())

        while queue:
            curr = queue.pop(0) # BFS
            
            # Find neighbors in the Undirected Graph
            neighbors = list(self.G.neighbors(curr))
            
            for nbr in neighbors:
                if nbr in self.visited:
                    continue
                
                # ORIENTATION LOGIC:
                # We are at 'curr' (Downstream). 'nbr' is new.
                # Therefore flow is nbr -> curr (Upstream -> Downstream).
                
                edge_data = self.G.get_edge_data(nbr, curr)
                base_data = edge_data[0] if 0 in edge_data else edge_data
                self.DiG.add_edge(nbr, curr, **base_data)
                
                self.visited.add(nbr)
                
                # STOPPING CONDITION
                if stop_at_lakes and nbr in lake_stop_set:
                    # 'nbr' is a Lake Node. We reached an inlet.
                    # We oriented the edge flowing INTO the lake, but we stop walking.
                    # This lake will be picked up in the next phase (if recursive).
                    continue
                else:
                    queue.append(nbr)

                # # --- DEBUG: FORCE SINGLE BRANCH ---
                # break # <--- ADD THIS HERE
                # # This forces the walker to pick ONLY the first valid upstream neighbor
                # # and ignore any forks.
                # # ----------------------------------

    def _reconstruct_geodataframe(self) -> gpd.GeoDataFrame:
        """Step 5: Reconstruct and UPDATE GRAPH GEOMETRY."""
        logger.info("Step 5: Reconstructing and syncing graph geometry...")
        
        oriented_lines = []
        
        # Iterate over directed edges: u (Source) -> v (Target)
        for u, v, data in self.DiG.edges(data=True):
            original_geom = data.get('geometry')
            if original_geom is None: continue

            u_point = Point(u)
            start_pt = Point(original_geom.coords[0])
            end_pt = Point(original_geom.coords[-1])
            
            # Robust Orientation Check
            if u_point.distance(start_pt) < u_point.distance(end_pt):
                final_geom = original_geom
            else:
                final_geom = LineString(list(original_geom.coords)[::-1])
            
            # CRITICAL FIX: Update the Graph object with the fixed geometry
            # This ensures Phase 3 (Widths) grabs the correct arrow direction
            self.DiG[u][v]['geometry'] = final_geom

            oriented_lines.append({
                'geometry': final_geom,
                'source_node': str(u),
                'target_node': str(v),
                'original_id': data.get('original_id', None)
            })
            
        result = gpd.GeoDataFrame(oriented_lines, crs=self.raw_rivers.crs)
        dropped = len(self.raw_rivers) - len(result)
        logger.info(f"  Final Count: {len(result)}. Dropped {dropped} disconnected segments.")
        return result



    # def _propagate_flow_from_sea(self):
    #     """
    #     Step 3: DEBUG MODE - SINGLE ANCHOR ANALYSIS.
    #     Only picks the first detected sea anchor and prints decision logic.
    #     """
    #     print("\n--- DEBUG: Analyzing First Anchor ---")
        
    #     self.DiG = nx.DiGraph()
    #     self.visited = set()
    #     queue = []
        
    #     # 1. Find Anchors
    #     # We need the sea geometry union for the intersection check
    #     sea_union = self.sea.unary_union
    #     candidates = gpd.sjoin(self.clean_rivers, self.sea[['geometry']], how='inner', predicate='intersects')
        
    #     if len(candidates) == 0:
    #         print("CRITICAL: No river segments intersect the sea!")
    #         return

    #     # 2. Pick ONLY the first candidate
    #     first_idx = candidates.index[0]
    #     row = candidates.loc[first_idx]
    #     geom = row.geometry
        
    #     print(f"Selected Anchor Segment Index: {first_idx}")
    #     print(f"  Geometry Start: {geom.coords[0]}")
    #     print(f"  Geometry End:   {geom.coords[-1]}")
        
    #     # 3. Orient the Anchor
    #     orientation = self._orient_anchor_segment(geom, sea_union)
        
    #     if orientation:
    #         source, target = orientation # Source (Land) -> Target (Sea)
            
    #         # Print the logic
    #         print(f"  Orientation Decision:")
    #         print(f"    Land Node (Source): {source}")
    #         print(f"    Sea Node (Target):  {target}")
            
    #         # Add to graph
    #         if self.G.has_edge(source, target):
    #             edge_data = self.G.get_edge_data(source, target)
    #             base_data = edge_data[0] if 0 in edge_data else edge_data
    #             self.DiG.add_edge(source, target, **base_data)
                
    #             self.visited.add(target)
    #             self.visited.add(source)
    #             queue.append(source)
    #             print(f"  --> Anchor added. Queue seeded with Land Node.")
    #         else:
    #             print("  ERROR: Edge not found in base graph (Graph mismatch).")
    #     else:
    #         print("  ERROR: Could not determine which end is in the sea. (Ambiguous Intersection)")

    #     # 4. Start Walking (Limited)
    #     print("\n--- DEBUG: Walking Upstream (Max 10 Steps) ---")
    #     self._run_upstream_bfs_debug(queue, max_steps=10)


    # def _run_upstream_bfs_debug(self, queue, max_steps=10):
    #     """
    #     Debug version of BFS. Prints step-by-step traversal.
    #     """
    #     steps = 0
        
    #     while queue:
    #         if steps >= max_steps:
    #             print(f"  [Limit Reached] Stopped after {steps} segments.")
    #             break
                
    #         curr = queue.pop(0) # 'curr' is the Downstream Node
            
    #         neighbors = list(self.G.neighbors(curr))
    #         valid_upstream = [n for n in neighbors if n not in self.visited]
            
    #         print(f"  Step {steps + 1}: At Node {curr}")
    #         print(f"    Found {len(neighbors)} neighbors. {len(valid_upstream)} are upstream.")
            
    #         for nbr in valid_upstream:
    #             # Logic: Flow is Nbr -> Curr
    #             edge_data = self.G.get_edge_data(nbr, curr)
    #             base_data = edge_data[0] if 0 in edge_data else edge_data
    #             self.DiG.add_edge(nbr, curr, **base_data)
                
    #             self.visited.add(nbr)
                
    #             print(f"    -> Link Added: {nbr} (Up) -> {curr} (Down)")
                
    #             # Single branch logic (as requested previously)
    #             queue.append(nbr)
    #             steps += 1
    #             break # Force single branch






def validate_arrow_directions(oriented_gdf, G):
    """
    Checks if the physical arrows align with the topological source nodes.
    """
    print("--- VALIDATION: Checking Arrow Directions ---")
    
    misaligned_count = 0
    total = len(oriented_gdf)
    
    for idx, row in oriented_gdf.iterrows():
        # Get the geometry start point (Base of the arrow)
        geom_start = Point(row.geometry.coords[0])
        
        # Get the Source Node from the DataFrame columns (convert str back to tuple if needed)
        # Note: In the previous step we converted to str for GPKG. 
        # But we can look up the edge in the graph using the index logic or just parsing.
        # Let's assume we can parse the string "(x, y)" back to tuple, or use the graph if available.
        
        # Easier way: The GDF has 'source_node' column as string "(123.1, 456.2)"
        # We need to parse that string to calculate distance.
        src_str = row['source_node']
        # Remove parens and split
        src_clean = src_str.replace('(', '').replace(')', '').split(',')
        src_x = float(src_clean[0])
        src_y = float(src_clean[1])
        source_point = Point(src_x, src_y)
        
        # Measure distance
        dist = geom_start.distance(source_point)
        
        # Threshold: If the arrow starts more than 1 meter away from the source node, 
        # it is likely pointing the wrong way (or the node is at the other end).
        if dist > 1.0:
            misaligned_count += 1
            if misaligned_count < 5:
                print(f"FAIL at Index {idx}: Arrow Base is {dist:.2f}m away from Source Node.")
                
    print(f"Validation Complete.")
    print(f"  Total Rivers: {total}")
    print(f"  Misaligned Arrows: {misaligned_count}")
    
    if misaligned_count == 0:
        print("  SUCCESS: All arrows originate at the upstream node.")
    else:
        print("  FAILURE: Some arrows are reversed or disconnected.")

# --- RUN THE VALIDATOR ---
# validate_arrow_directions(oriented_rivers, analyzer.G)

def validate_topology_continuity(gdf):
    """
    Chains through the rivers to ensure the End Point of one line 
    is physically close to the Start Point of the next line.
    """
    print("--- TOPOLOGY CONTINUITY CHECK ---")
    
    # Create a spatial index for speed
    # We will look for lines that touch the END of the current line
    sindex = gdf.sindex
    
    errors = 0
    checked = 0
    
    # Check 100 random rivers to save time
    sample = gdf.sample(100) if len(gdf) > 100 else gdf
    
    for idx, row in sample.iterrows():
        # This line ends at:
        end_pt = Point(row.geometry.coords[-1])
        
        # Find lines that theoretically start here (Target Node matches Source Node of others)
        # We rely on the 'target_node' attribute string we saved
        target_node_id = row['target_node']
        
        # Find downstream neighbors in the dataframe
        downstream_rows = gdf[gdf['source_node'] == target_node_id]
        
        if downstream_rows.empty:
            continue # Reached the sea or a sink
            
        checked += 1
        
        # For every downstream neighbor, the START point should be close to our END point
        for _, down_row in downstream_rows.iterrows():
            start_pt_down = Point(down_row.geometry.coords[0])
            dist = end_pt.distance(start_pt_down)
            
            if dist > 1.0: # 1 meter tolerance
                print(f"DISCONTINUITY at Index {idx}:")
                print(f"  River A ends at {end_pt}")
                print(f"  River B starts at {start_pt_down}")
                print(f"  Gap: {dist:.2f}m")
                print(f"  (This means River B is flowing BACKWARDS relative to River A)")
                errors += 1
                
    print(f"Checked {checked} junctions.")
    if errors == 0:
        print("SUCCESS: Water flows continuously from line to line.")
    else:
        print(f"FAILURE: Found {errors} breaks in flow direction.")

# Run it
# validate_topology_continuity(rivers_with_width)





def plot_flow(oriented_gdf, sea_gdf, lake_gdf=None):
    """
    Visualizes the river network with flow direction arrows.
    Fixed to prevent 'Red Blob' and Legend warnings.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 1. Plot Base Layers
    sea_gdf.plot(ax=ax, color='#e0f7fa', edgecolor='none', zorder=1)
    if lake_gdf is not None:
        lake_gdf.plot(ax=ax, color='#81d4fa', edgecolor='none', zorder=2)
        
    # 2. Plot River Lines (Black background lines)
    oriented_gdf.plot(ax=ax, color='black', linewidth=0.8, alpha=0.5, zorder=3)
    
    # 3. Calculate Arrows
    # Get start and end points
    coords = oriented_gdf.geometry.apply(lambda geom: (geom.centroid.x, geom.centroid.y, 
                                                       geom.coords[-1][0] - geom.coords[0][0], 
                                                       geom.coords[-1][1] - geom.coords[0][1]))
    X, Y, U, V = zip(*coords)
    X, Y, U, V = np.array(X), np.array(Y), np.array(U), np.array(V)
    
    # CRITICAL FIX: Normalize U, V to unit vectors
    # This ensures a 5km long river and a 500m river get the same size arrow
    norm = np.sqrt(U**2 + V**2)
    
    # Avoid division by zero
    mask = norm > 0
    X, Y = X[mask], Y[mask]
    U, V = U[mask] / norm[mask], V[mask] / norm[mask]
    
    # 4. Quiver Plot (The Arrows)
    # scale=30 means "30 data units per arrow unit". Adjust if still too big/small.
    # width=0.002 makes the shaft thin.
    ax.quiver(X[::10], Y[::10], U[::10], V[::10], 
              color='red', 
              zorder=4,
              pivot='mid',      # Center arrow on the river segment
              units='inches',   # Define size relative to the plot window, not map coordinates
              scale=10,         # Higher number = Shorter arrows (Counter-intuitive!)
              width=0.005,      # Thin shaft
              headwidth=5,      # Proportional head
              headlength=5,
              alpha=0.8)
    
    # 5. Manual Legend (Fixes the warnings)
    sea_patch = mpatches.Patch(color='#e0f7fa', label='Sea')
    lake_patch = mpatches.Patch(color='#81d4fa', label='Lakes')
    river_line = mlines.Line2D([], [], color='black', linewidth=1, label='River Channel')
    flow_arrow = mlines.Line2D([], [], color='red', marker='>', linestyle='None',
                              markersize=10, label='Flow Direction')
    
    handles = [sea_patch, river_line, flow_arrow]
    if lake_gdf is not None:
        handles.insert(1, lake_patch)
        
    ax.legend(handles=handles, loc='upper right')
    
    plt.title("Hydrologically Oriented River Network")
    plt.axis('equal') # Ensure map isn't stretched
    plt.show()

###############################################################################
###############################################################################
###############################################################################

class CalculateFlowMagnitude:
    """
    A processing engine that calculates hydrological statistics 
    (Flow Accumulation and Strahler Stream Order) for a river DAG.
    """

    def __init__(self, graph: nx.DiGraph):
        # We work on a copy to avoid mutating the original graph unexpectedly
        # though usually in pipelines modifying in place is acceptable.
        self.graph = graph

    def run(self) -> nx.DiGraph:
        """
        Orchestrates the flow calculation pipeline.
        """
        logger.info("--- Phase 2: Flow Accumulation (Class-Based) ---")

        self._break_cycles()
        self._initialize_attributes()
        
        # Get execution order
        topo_order = self._get_topological_order()
        
        if topo_order:
            self._propagate_flow(topo_order)
            self._map_attributes_to_edges()
        else:
            logger.error("Skipping flow calculation due to topological errors.")

        return self.graph

    def _break_cycles(self):
        """Step 1: Detect and break cycles to enforce DAG structure."""
        if not nx.is_directed_acyclic_graph(self.graph):
            logger.warning("Graph contains cycles! Attempting to break them...")
            
            # Find cycles (returns a generator of lists)
            cycles = list(nx.simple_cycles(self.graph))
            logger.info(f"Found {len(cycles)} cycles.")
            
            count = 0
            for cycle in cycles:
                # Remove the edge returning to the start of the cycle (u -> v)
                # Cycle is [u, v, w, u] represented as nodes [u, v, w]
                u, v = cycle[-1], cycle[0]
                
                if self.graph.has_edge(u, v):
                    self.graph.remove_edge(u, v)
                    count += 1
            
            logger.info(f"Broke {count} cyclic edges.")
        else:
            logger.info("Graph is acyclic (DAG). No repairs needed.")

    def _initialize_attributes(self):
        """Step 2: Set baseline values for accumulation and order."""
        # flow_acc: Total number of upstream segments + 1 (itself)
        nx.set_node_attributes(self.graph, 1, 'flow_acc')
        # strahler: Stream order (1 = Headwater)
        nx.set_node_attributes(self.graph, 1, 'strahler')

    def _get_topological_order(self) -> list:
        """Step 3: Sort nodes from Headwaters -> Sea."""
        try:
            topo_order = list(nx.topological_sort(self.graph))
            logger.info(f"Topological sort successful. Processing {len(topo_order)} nodes.")
            return topo_order
        except nx.NetworkXUnfeasible:
            logger.critical("Graph still has cycles after repair attempt. Cannot compute flow.")
            return []

    def _propagate_flow(self, topo_order: list):
        """Step 4: Iterate downstream to sum accumulation and calculate hierarchy."""
        for node in topo_order:
            # Get upstream neighbors (predecessors)
            predecessors = list(self.graph.predecessors(node))
            
            if not predecessors:
                continue # It's a source, keep default values (1, 1)
                
            # --- A. Flow Accumulation (Simple Sum) ---
            incoming_flow = sum(self.graph.nodes[pred]['flow_acc'] for pred in predecessors)
            self.graph.nodes[node]['flow_acc'] += incoming_flow
            
            # --- B. Strahler Order (Hierarchical) ---
            incoming_strahlers = [self.graph.nodes[pred]['strahler'] for pred in predecessors]
            
            if incoming_strahlers:
                max_s = max(incoming_strahlers)
                # Strahler Rule: If two streams of max order join, order increments.
                # Otherwise, it stays the same.
                if incoming_strahlers.count(max_s) >= 2:
                    self.graph.nodes[node]['strahler'] = max_s + 1
                else:
                    self.graph.nodes[node]['strahler'] = max_s

    def _map_attributes_to_edges(self):
        """Step 5: Write node calculations to outgoing edges for vector mapping."""
        # Iterate over (u, v) pairs.
        # Since DiGraph is simple, we don't need 'keys=True'
        for u, v in self.graph.edges():
            # The properties of the edge u->v are defined by the accumulation at u
            flow_val = self.graph.nodes[u]['flow_acc']
            strahler_val = self.graph.nodes[u]['strahler']
            
            self.graph[u][v]['magnitude'] = flow_val
            self.graph[u][v]['strahler'] = strahler_val
            
        logger.info("Flow attributes mapped to edges.")



# --- VISUALIZER ---

def plot_river_hierarchy(DiG, sea_gdf, lake_gdf=None):
    """
    Visualizes the river network with line thickness based on Magnitude
    and color based on Strahler Order.
    """
    fig, ax = plt.subplots(figsize=(15, 15), facecolor='white')
    
    # 1. Context Layers
    sea_gdf.plot(ax=ax, color='#e0f7fa', edgecolor='none', zorder=1)
    if lake_gdf is not None:
        lake_gdf.plot(ax=ax, color='#81d4fa', edgecolor='none', zorder=2)
        
    # 2. Extract Edge Data for Plotting
    lines = []
    strahlers = []
    magnitudes = []
    
    for u, v, data in DiG.edges(data=True):
        if 'geometry' in data:
            lines.append(data['geometry'])
            # Default to 1 if missing
            strahlers.append(data.get('strahler', 1))
            magnitudes.append(data.get('magnitude', 1))
            
    if not lines:
        print("No river edges to plot!")
        return

    # 3. Create GeoDataFrame for Plotting
    # We use a temporary GDF to leverage matplotlib plotting
    plot_gdf = gpd.GeoDataFrame({
        'geometry': lines, 
        'strahler': strahlers,
        'magnitude': magnitudes
    }, crs=sea_gdf.crs)
    
    # 4. Dynamic Styling
    # Width: Logarithmic scaling so the Reik doesn't cover the whole map
    # We add 0.5 base width so source streams are visible
    linewidths = (np.log(plot_gdf['magnitude']) * 0.8) + 0.5
    
    # Color: Colormap based on Strahler Order
    # Low Order = Blue/Purple, High Order = Yellow/Red
    cmap = cm.get_cmap('plasma') 
    norm = Normalize(vmin=plot_gdf['strahler'].min(), vmax=plot_gdf['strahler'].max())
    
    # 5. Plot
    # We pass the calculated linewidths directly
    plot_gdf.plot(ax=ax, 
                  column='strahler', 
                  cmap=cmap, 
                  linewidth=linewidths,
                  zorder=3,
                  legend=True,
                  legend_kwds={'label': "Strahler Stream Order", 'shrink': 0.5})
    
    plt.title(f"River Network Hierarchy\n(Thickness = Flow Magnitude, Color = Stream Order)")
    plt.axis('equal')
    plt.show()




###############################################################################
###############################################################################
###############################################################################



def assign_river_widths(G: nx.DiGraph, 
                        min_width: float = 10.0, 
                        max_width: float = 500.0, 
                        scale_factor: float = 1.5) -> gpd.GeoDataFrame:
    """
    Converts flow magnitude into physical river width using a logarithmic scaling function.
    
    FORMULA:
    Width = Min_Width + (log(Magnitude) * Scale_Factor)
    (Capped at max_width).
    
    Args:
        G: The enriched river DiGraph (from Step 2).
        min_width: Width of a source stream (Order 1) in meters.
        max_width: Hard cap for the widest river in meters.
        scale_factor: Multiplier to control widening rate.
        
    Returns:
        gpd.GeoDataFrame: River vectors with a 'width' column (in meters).
    """
    print(f"--- Phase 3: Physical Width Derivation ---")
    print(f"Params: Base={min_width}m, Cap={max_width}m, Scale={scale_factor}")
    
    river_segments = []
    
    # Iterate over edges
    # We use G.edges(data=True) to access attributes
    for u, v, data in G.edges(data=True):
        if 'geometry' not in data:
            continue
            
        # 1. Get Magnitude (Default to 1 if missing)
        # Ensure magnitude is at least 1 to avoid log(0) errors
        mag = max(data.get('magnitude', 1), 1)
        
        # 2. Apply Logarithmic Scaling
        # Natural Log (ln) grows fast initially then slows down, mimicking real rivers
        # formula: Base Width + (ln(Flow) * Factor)
        calc_width = min_width + (np.log(mag) * scale_factor)
        
        # 3. Apply Cap
        final_width = min(calc_width, max_width)
        
        # 4. Store Data
        river_segments.append({
            'geometry': data['geometry'],
            'source_node': str(u),
            'target_node': str(v),
            'magnitude': mag,
            'strahler': data.get('strahler', 1),
            'width': final_width
        })
    
    # 5. Convert to GeoDataFrame
    # We assume the graph nodes store the CRS, or we grab it from the geometry if possible.
    # Since we don't have the CRS object passed in explicitly, we will assume
    # the user handles CRS assignment or we infer it later. 
    # Ideally, pass CRS in or grab from a node attribute if stored.
    # For now, we return a GDF without CRS and set it outside.
    gdf = gpd.GeoDataFrame(river_segments)
    
    print(f"assigned widths to {len(gdf)} segments.")
    print(f"Width Range: {gdf['width'].min():.2f}m to {gdf['width'].max():.2f}m")
    
    return gdf

# --- VISUALIZER ---

def plot_river_physics(width_gdf, sea_gdf, lake_gdf=None):
    """
    Visualizes the ACTUAL physical footprint of the rivers.
    It buffers the lines by (width / 2) to show the river as a polygon.
    """
    fig, ax = plt.subplots(figsize=(15, 15), facecolor='white')
    
    # 1. Plot Context
    sea_gdf.plot(ax=ax, color='#e0f7fa', edgecolor='none', zorder=1)
    if lake_gdf is not None:
        lake_gdf.plot(ax=ax, color='#81d4fa', edgecolor='none', zorder=2)
        
    # 2. Create Physical Representation (Polygons)
    # We create a temporary column for visualization only
    # buffer(distance) -> distance is radius, so we use width / 2
    print("Buffering geometry for visualization...")
    
    # Check if CRS is projected (meters). If Lat/Lon, this visualization will fail/look huge.
    if width_gdf.crs and not width_gdf.crs.is_projected:
        print("WARNING: GDF is in Degrees! Buffering by meters will fail.")
    
    buffered_rivers = width_gdf.copy()
    buffered_rivers['geometry'] = buffered_rivers.geometry.buffer(buffered_rivers['width'] / 2)
    
    # 3. Plot Rivers
    # We color by width to emphasize the difference
    buffered_rivers.plot(ax=ax, 
                         column='width', 
                         cmap='Blues', 
                         legend=True,
                         legend_kwds={'label': "River Width (meters)"},
                         zorder=3)
    
    plt.title("Physical River Footprint (Buffered Geometry)")
    plt.axis('equal')
    plt.show()



###############################################################################
###############################################################################
###############################################################################

def save_hydro_network(gdf: gpd.GeoDataFrame, output_path: str):
    """
    Saves the river network to GPKG.
    
    1. Preserves Geometry Direction (Flow).
    2. Converts Tuple IDs (Nodes) to Strings so GPKG doesn't crash.
    """
    print(f"Saving hydrology network to {output_path}...")
    
    # Work on a copy so we don't break the active notebook object
    save_gdf = gdf.copy()
    
    # DATA TYPE FIX: 
    # NetworkX nodes are often tuples: (34500.1, 56000.2)
    # GPKG attribute tables only accept String, Int, Float.
    # We convert source/target columns to string representation.
    if 'source' in save_gdf.columns:
        save_gdf['source_node'] = save_gdf['source_node'].astype(str)
    
    if 'target' in save_gdf.columns:
        save_gdf['target_node'] = save_gdf['target_node'].astype(str)
        
    # Save to file
    save_gdf.to_file(output_path, driver="GPKG")
    print("Save complete.")



###############################################################################
###############################################################################
###############################################################################




