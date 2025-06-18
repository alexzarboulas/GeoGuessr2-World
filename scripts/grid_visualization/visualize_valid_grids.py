"""
Generates an interactive HTML map of all Level 8 grid cells, color-coded by status:

- Green: Grid intersects land and has usable image data
- Blue: Grid is ocean-only
- Gray: Grid intersects land but no usable images were found

The script uses shapely and geopandas to define and process grid cells,
and Folium to render the results in an HTML map.

Author: Alexander Zarboulas
Date: 2025-06-18
"""

#Import libraries
import folium
import geopandas as gpd
from shapely.geometry import box
from grid_utils import generate_recursive_grids, intersects_land, load_empty_grids

#Configuration
LEVEL = 8
LAT_RANGE = (-90, 90)
LON_RANGE = (-180, 180)

#Generate all grid bounding boxes for the specified level
all_grids = generate_recursive_grids(LAT_RANGE, LON_RANGE, LEVEL)

#Filter grids that intersect land
land_grids = {
    grid_id: bbox
    for grid_id, bbox in all_grids.items()
    if intersects_land(bbox)
}

#Load empty land grids from ../data/empty_grids.txt
empty_grids = load_empty_grids()

#Build a GeoDataFrame with grid IDs, geometries, and statuses (colors)
records = []
for grid_id, bbox in all_grids.items():
    #Reorder coordinates to (lon_min, lat_min, lon_max, lat_max)
    lon_min, lat_min, lon_max, lat_max = bbox[1], bbox[0], bbox[3], bbox[2]
    geom = box(lon_min, lat_min, lon_max, lat_max)

    #Determine grid status
    if grid_id not in land_grids:
        status = "water"
    elif grid_id in empty_grids:
        status = "empty"
    else:
        status = "valid"

    records.append({"grid_id": grid_id, "geometry": geom, "status": status})


#Create GeoDataFrame from records
gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

#Initialize a folium map
m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")

#Add grid tiles and colors
for _, row in gdf.iterrows():
    status = row["status"]
    if status == "valid":
        color = "green"
    elif status == "empty":
        color = "gray"
    else:  # water
        color = "blue"

    geo_json = folium.GeoJson(
        row["geometry"],
        style_function=lambda x, col=color: {
            "fillColor": col,
            "color": "black",
            "weight": 0.2,
            "fillOpacity": 0.6 if col != "gray" else 0.1
        }
    )
    geo_json.add_to(m)

#Save to HTML file
m.save("valid_grids_map.html")
print("Map saved to 'valid_grids_map.html'")
