import folium
import geopandas as gpd
from shapely.geometry import box
from grid_utils import generate_recursive_grids, intersects_land, load_empty_grids

LEVEL = 8

#Generate grid bounding boxes recursively
LAT_RANGE = (-90, 90)
LON_RANGE = (-180, 180)
all_grids = generate_recursive_grids(LAT_RANGE, LON_RANGE, LEVEL)
land_grids = {
    grid_id: bbox
    for grid_id, bbox in all_grids.items()
    if intersects_land(bbox)
}
empty_grids = load_empty_grids()

#Build a "geodataframe" with grid IDs, geometries, and statuses (colors)
records = []
for grid_id, bbox in all_grids.items():
    lon_min, lat_min, lon_max, lat_max = bbox[1], bbox[0], bbox[3], bbox[2]  # reorder
    geom = box(lon_min, lat_min, lon_max, lat_max)

    if grid_id not in land_grids:
        status = "water"
    elif grid_id in empty_grids:
        status = "empty"
    else:
        status = "valid"

    records.append({"grid_id": grid_id, "geometry": geom, "status": status})

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
print("✅ Interactive map saved to 'valid_grids_map.html'")
