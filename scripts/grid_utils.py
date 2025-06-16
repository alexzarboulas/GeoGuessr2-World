import os
import geopandas as gpd
from shapely.geometry import box

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAND_SHAPEFILE_PATH = os.path.join(BASE_DIR, "../data/ne_50m_land/ne_50m_land.shp")
EMPTY_GRIDS_PATH = "../data/empty_grids.txt" # Path to store empty grid IDs (will be skipped)

#Generate grid bounding boxes recursively (splitting into quadrants)
def generate_recursive_grids(lat_range, lon_range, level, prefix=""):
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
    #dictionary, where key is quadrant number and value is tuple of (lat_range, lon_range)
    for key, (lat, lon) in quadrants.items():
        new_prefix = f"{prefix}/{key}"
        subgrids = generate_recursive_grids(lat, lon, level - 1, new_prefix)
        results.update(subgrids)
    return results


# Load land shapefile using GeoPandas
land_gdf = gpd.read_file(LAND_SHAPEFILE_PATH)


#function to check if a bounding box intersects with land
def intersects_land(bbox):

    lat_min, lon_min, lat_max, lon_max = bbox
    poly = box(lon_min, lat_min, lon_max, lat_max)
    return land_gdf.intersects(poly).any()

#logging empty grids, based on response form Mapillary API on fetching images
def log_empty_grid(grid_id):

    os.makedirs(os.path.dirname(EMPTY_GRIDS_PATH), exist_ok=True)
    with open(EMPTY_GRIDS_PATH, "a", encoding="utf-8") as f:
        f.write(grid_id + "\n")

#returns a set of grid IDs that are marked as empty, will be skipped
def load_empty_grids():

    if not os.path.exists(EMPTY_GRIDS_PATH):
        return set()
    with open(EMPTY_GRIDS_PATH, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


#Overall grid filter function
#returns dictionary of VALID grid IDs and their bounding boxes at specified level
def get_all_grid_bboxes(level=8):

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
