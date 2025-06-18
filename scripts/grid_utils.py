"""
Utility functions for recursively dividing the globe into hierarchical grids
and filtering them based on land intersection using shapefiles.

Key Components:
- generate_recursive_grids(): Recursively divides globe into Level 8 grid tiles
- intersects_land(): Checks if a tile overlaps land using Natural Earth shapefiles
- get_all_grid_bboxes(): Returns all valid (non-empty) land grid tiles
- log_empty_grid(): Records empty grid IDs during image fetching
- load_empty_grids(): Loads previously logged empty grid IDs

Author: Alexander Zarboulas
Date: 2025-06-18
"""

#Import libraries
import os
import geopandas as gpd
from shapely.geometry import box

#Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAND_SHAPEFILE_PATH = os.path.join(BASE_DIR, "../data/ne_50m_land/ne_50m_land.shp")
EMPTY_GRIDS_PATH = "../data/empty_grids.txt" # Path to store empty grid IDs (will be skipped)

#Load land shapefile using GeoPandas
land_gdf = gpd.read_file(LAND_SHAPEFILE_PATH)

def generate_recursive_grids(lat_range, lon_range, level, prefix=""):
    """
    Recursively divides the globe into a hierarchical grid structure.

    Args:
        lat_range (tuple): (lat_min, lat_max)
        lon_range (tuple): (lon_min, lon_max)
        level (int): Depth of recursion (1 to 8)
        prefix (str): Grid ID prefix used to build hierarchical keys

    Returns:
        dict: {grid_id: [lat_min, lon_min, lat_max, lon_max]}
    """
    if level == 0:
        return {prefix.strip("/"): [lat_range[0], lon_range[0], lat_range[1], lon_range[1]]}

    mid_lat = (lat_range[0] + lat_range[1]) / 2
    mid_lon = (lon_range[0] + lon_range[1]) / 2

    quadrants = {
        "0": ((mid_lat, lat_range[1]), (lon_range[0], mid_lon)),  # NW
        "1": ((mid_lat, lat_range[1]), (mid_lon, lon_range[1])),  # NE
        "2": ((lat_range[0], mid_lat), (lon_range[0], mid_lon)),  # SW
        "3": ((lat_range[0], mid_lat), (mid_lon, lon_range[1])),  # SE
    }

    results = {}
    for key, (lat, lon) in quadrants.items():
        new_prefix = f"{prefix}/{key}"
        subgrids = generate_recursive_grids(lat, lon, level - 1, new_prefix)
        results.update(subgrids)
    return results

def intersects_land(bbox):
    """
    Checks if a given bounding box intersects with landmass.

    Args:
        bbox (list): [lat_min, lon_min, lat_max, lon_max]

    Returns:
        bool: True if bbox intersects land, False otherwise.
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    poly = box(lon_min, lat_min, lon_max, lat_max)
    return land_gdf.intersects(poly).any()

def log_empty_grid(grid_id):
    """
    Logs a grid ID that failed image fetching (i.e., empty).

    Args:
        grid_id (str): Hierarchical ID of the grid to log
    """
    os.makedirs(os.path.dirname(EMPTY_GRIDS_PATH), exist_ok=True)
    with open(EMPTY_GRIDS_PATH, "a", encoding="utf-8") as f:
        f.write(grid_id + "\n")

def load_empty_grids():
    """
    Loads grid IDs that were previously marked as empty.

    Returns:
        set: Set of grid IDs to skip
    """
    if not os.path.exists(EMPTY_GRIDS_PATH):
        return set()
    with open(EMPTY_GRIDS_PATH, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def get_all_grid_bboxes(level=8):
    """
    Generates and filters grid bounding boxes based on land intersection
    and previous image-fetching results.

    Args:
        level (int): Depth of grid subdivision (default is 8)

    Returns:
        dict: {grid_id: [lat_min, lon_min, lat_max, lon_max]} for usable grids
    """
    LAT_RANGE = (-90, 90)
    LON_RANGE = (-180, 180)

    all_grids = generate_recursive_grids(LAT_RANGE, LON_RANGE, level)
    land_grids = {
        grid_id: bbox
        for grid_id, bbox in all_grids.items()
        if intersects_land(bbox)
    }

    empty_grids = load_empty_grids()
    filtered_grids = {
        grid_id: bbox
        for grid_id, bbox in land_grids.items()
        if grid_id not in empty_grids
    }

    print(f"{len(filtered_grids)} usable grids out of {len(all_grids)} total at level {level} ({len(land_grids)} intersect land)")
    return filtered_grids
