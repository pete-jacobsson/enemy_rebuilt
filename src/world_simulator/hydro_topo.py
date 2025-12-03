import geopandas as gpd
import logging

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import momepy

import networkx as nx
import numpy as np

import shapely
from shapely.geometry import Point, LineString

from tqdm import tqdm


# Ensure we have a logger
logger = logging.getLogger(__name__)
from .logger import setup_logger
setup_logger(log_name="wfrp_phys_geo")


###############################################################################
###############################################################################
###############################################################################


class HydrologyAnalyzer:
    """
    A stateful engine that converts a raw, undirected river network into a 
    hydrologically oriented Directed Acyclic Graph (DAG).
    """
    
    def __init__(self, river_gdf: gpd.GeoDataFrame, sea_gdf: gpd.GeoDataFrame, lake_gdf: gpd.GeoDataFrame = None):
        self.raw_rivers = river_gdf
        self.sea = sea_gdf
        self.lakes = lake_gdf
        
        # Internal State
        self.G = None       # The undirected base graph
        self.DiG = None     # The directed result graph
        self.sinks = set()  # Set of nodes that drain to sea
        
    def run(self) -> gpd.GeoDataFrame:
        """
        Orchestrates the analysis pipeline.
        """
        logger.info("--- Phase 1: Hydrological Orientation (Class-Based) ---")
        
        self._build_base_topology()
        
        if self.lakes is not None:
            self._add_virtual_lake_connections()
            
        self._identify_sea_sinks()
        self._orient_network_bfs()
        
        return self._reconstruct_geodataframe()

    def _build_base_topology(self):
        """Step 1: Convert LineStrings to an undirected NetworkX graph."""
        logger.info("Building base topology...")
        # Explode to ensure simple lines
        self.exploded_rivers = self.raw_rivers.explode(index_parts=False).reset_index(drop=True)
        # Primal approach: Nodes = Junctions, Edges = Rivers
        self.G = momepy.gdf_to_nx(self.exploded_rivers, approach='primal')

    def _add_virtual_lake_connections(self):
        """Step 2: Bridge gaps by connecting river mouths/sources to lake centers."""
        logger.info("Integrating canonical lakes...")
        
        # Get graph nodes as geometry
        node_points = [Point(n) for n in self.G.nodes]
        node_gdf = gpd.GeoDataFrame({'node_id': list(self.G.nodes)}, geometry=node_points, crs=self.raw_rivers.crs)
        
        # Spatial Join: Find nodes touching lakes
        lake_nodes = gpd.sjoin(node_gdf, self.lakes, how='inner', predicate='intersects')
        
        count = 0
        for lake_idx, group in lake_nodes.groupby('index_right'):
            # Create Virtual Node at Lake Centroid
            lake_geom = self.lakes.geometry.iloc[lake_idx]
            virtual_node = (lake_geom.centroid.x, lake_geom.centroid.y)
            
            self.G.add_node(virtual_node, type='virtual_lake')
            
            # Connect real nodes to virtual node with 0 weight
            for real_node in group.node_id:
                self.G.add_edge(real_node, virtual_node, weight=0, type='virtual_connection')
                self.G.add_edge(virtual_node, real_node, weight=0, type='virtual_connection')
                count += 1
                
        logger.info(f"Added {count} connections to virtual lake nodes.")

    def _identify_sea_sinks(self):
        """Step 3: Find nodes that touch the sea polygon."""
        logger.info("Identifying discharge points...")
        
        node_points = [Point(n) for n in self.G.nodes]
        node_gdf = gpd.GeoDataFrame({'node_id': list(self.G.nodes)}, geometry=node_points, crs=self.sea.crs)
        
        # Spatial Join
        matches = gpd.sjoin(node_gdf, self.sea, how='inner', predicate='intersects')
        self.sinks = set(matches.node_id.tolist())
        
        if not self.sinks:
            logger.warning("No river mouths found touching the sea! Check your CRS or sea polygon.")
        else:
            logger.info(f"Found {len(self.sinks)} river mouths.")

    def _orient_network_bfs(self):
        """Step 4: Reverse BFS from Sea Sinks to orient flow direction."""
        logger.info("Orienting flow direction (Sea -> Source)...")
        
        self.DiG = nx.DiGraph()
        queue = list(self.sinks)
        visited = set(self.sinks)
        
        # Progress bar
        pbar = tqdm(total=len(self.G.nodes), desc="Orienting")
        
        while queue:
            current_node = queue.pop(0)
            pbar.update(1)
            
            # Look at neighbors in the Undirected Graph
            neighbors = list(self.G.neighbors(current_node))
            
            for nbr in neighbors:
                if nbr not in visited:
                    # If we are at 'current' and found 'nbr', then flow is Nbr -> Current
                    
                    # Handle Momepy MultiGraph structure (get first edge key)
                    edge_data = self.G.get_edge_data(nbr, current_node)
                    base_data = edge_data[0] if 0 in edge_data else edge_data
                    
                    # Check if this is a virtual lake connection
                    is_virtual = (self.G.nodes[current_node].get('type') == 'virtual_lake') or \
                                 (self.G.nodes[nbr].get('type') == 'virtual_lake')
                    
                    # We only add REAL rivers to the final directed graph
                    if not is_virtual:
                        self.DiG.add_edge(nbr, current_node, **base_data)
                    
                    # We traverse everything (including lakes) to find upstream sources
                    visited.add(nbr)
                    queue.append(nbr)
        
        pbar.close()

    def _reconstruct_geodataframe(self) -> gpd.GeoDataFrame:
        """Step 5: Reconstruct with explicit logging and validation."""
        logger.info("Reconstructing geometry...")
        
        oriented_lines = []
        flip_count = 0
        keep_count = 0
        weird_geometry_count = 0
        
        # Iterate over the directed edges (Flow goes u -> v)
        for i, (u, v, data) in enumerate(self.DiG.edges(data=True)):
            original_geom = data.get('geometry')
            
            if original_geom is None:
                continue

            # 1. Coordinate Extraction
            # We trust the Graph (u is Source). We check the Geometry.
            u_point = Point(u)
            
            geom_start = Point(original_geom.coords[0])
            geom_end = Point(original_geom.coords[-1])
            
            # 2. Distance Check
            dist_u_to_start = u_point.distance(geom_start)
            dist_u_to_end = u_point.distance(geom_end)
            
            # 3. Decision Logic
            # We want the geometry to start near u.
            if dist_u_to_start < dist_u_to_end:
                # Case A: Geometry already starts near Source. Keep it.
                final_geom = original_geom
                keep_count += 1
                decision = "KEEP"
            else:
                # Case B: Geometry starts near Target. Flip it.
                final_geom = LineString(list(original_geom.coords)[::-1])
                flip_count += 1
                decision = "FLIP"

            # 4. DEBUG LOGGING (First 5 examples only)
            if i < 5:
                logger.info(f"--- River Segment {i} ---")
                logger.info(f"  Source Node (u): {u}")
                logger.info(f"  Geom Start: {geom_start.coords[0]}")
                logger.info(f"  Geom End:   {geom_end.coords[0]}")
                logger.info(f"  Dist u->Start: {dist_u_to_start:.4f}")
                logger.info(f"  Dist u->End:   {dist_u_to_end:.4f}")
                logger.info(f"  Decision: {decision}")

            # 5. Sanity Check for "Weird" Geometries
            # If u is far from BOTH ends, something is wrong with the graph/geom mapping
            if min(dist_u_to_start, dist_u_to_end) > 1.0: # 1 meter tolerance
                weird_geometry_count += 1
                if weird_geometry_count < 5:
                     logger.warning(f"  [WARNING] Node u is far from both ends! Min Dist: {min(dist_u_to_start, dist_u_to_end)}")

            oriented_lines.append({
                'geometry': final_geom,
                'source_node': str(u),
                'target_node': str(v),
                'original_id': data.get('original_id', None)
            })
            
        result_gdf = gpd.GeoDataFrame(oriented_lines, crs=self.raw_rivers.crs)
        
        logger.info(f"Reconstruction Stats:")
        logger.info(f"  Kept Original Direction: {keep_count}")
        logger.info(f"  Flipped Direction:       {flip_count}")
        if weird_geometry_count > 0:
            logger.warning(f"  'Floating' Segments (Node mismatch > 1m): {weird_geometry_count}")
            
        return result_gdf



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
validate_topology_continuity(rivers_with_width)





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
            'source': u,
            'target': v,
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
        save_gdf['source'] = save_gdf['source'].astype(str)
    
    if 'target' in save_gdf.columns:
        save_gdf['target'] = save_gdf['target'].astype(str)
        
    # Save to file
    save_gdf.to_file(output_path, driver="GPKG")
    print("Save complete.")




