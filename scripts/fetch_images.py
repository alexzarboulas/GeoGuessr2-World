"""
Fetches and filters Mapillary images for each Level-8 grid cell using the Mapillary API.
Filters based on brightness and spatial redundancy (using KDTree).
Saves images and metadata under `../data/raw_images/<grid_id>/`.
Designed for batch processing with ThreadPoolExecutor across ~16,000 grid cells.

Author: Alexander Zarboulas
Date: 2025-06-18
"""

#Import libraries
import os
import requests
import time
import csv
from PIL import Image 
from io import BytesIO  
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from scipy.spatial import KDTree
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from grid_utils import get_all_grid_bboxes, log_empty_grid


#Confirguration
ACCESS_TOKEN = "MLY|30161963110055518|c4771e278fd07ec1c609fe91c5c13f4e"
OUTPUT_FOLDER = "../data/raw_images"
METADATA_PATH = "../data/metadata.csv"
IMAGES_PER_REGION = 20
BRIGHTNESS_THRESHOLD = 40
DISTANCE_THRESHOLD_DEGREES = 0.0007
REGIONS = dict(list(get_all_grid_bboxes(level=8).items()))


#Initialize geocoder with rate limiting, and reverse geocoding function
geolocator = Nominatim(user_agent="geo_project")
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

def is_bright_enough(img):
    """
    Returns True if the image has brightness above the configured threshold.

    Args:
        img (PIL.Image): The image to check.

    Returns:
        bool: True if bright enough, False otherwise.
    """
    grayscale = img.convert("L")
    mean_brightness = np.array(grayscale).mean()
    return mean_brightness > BRIGHTNESS_THRESHOLD


def is_far_enough(coord, visited_tree, threshold_rad):
    """
    Determine whether a new image coordinate is sufficiently far from all previous images.

    Args:
        coord (np.ndarray): Coordinate of the new image (in radians).
        visited_tree (KDTree or None): KDTree of existing coordinates.
        threshold_rad (float): Minimum distance in radians.

    Returns:
        bool: True if the coordinate is far enough from others, False otherwise.
    """
    if visited_tree is None:
        return True
    dist, _ = visited_tree.query(coord)
    return dist >= threshold_rad




MAX_PAGES_PER_REGION = 1
def fetch_images(region_name, bbox):
    """
    Fetch and filter images for a given region using the Mapillary API.

    Args:
        region_name (str): The name/ID of the region (grid cell).
        bbox (tuple): Bounding box as (lat_min, lon_min, lat_max, lon_max).

    Returns:
        list: A list of metadata rows for successfully saved images.
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    url = "https://graph.mapillary.com/images"

    all_rows = []
    visited_coords = []
    visited_tree = None
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    page = 0
    images_downloaded = 0

    while images_downloaded < IMAGES_PER_REGION and page < MAX_PAGES_PER_REGION:
        params = {
            "access_token": ACCESS_TOKEN,
            "fields": "id,thumb_2048_url,geometry",
            "bbox": ",".join(map(str, [lon_min, lat_min, lon_max, lat_max])), 
            "limit": 10,
            "page": page
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error fetching images for {region_name} (page {page}): {response.status_code}")
            break

        data = response.json().get('data', [])
        if not data:
            print(f"No more images available for {region_name}")
            log_empty_grid(region_name)
            break


        for item in data:
            try:
                img_id = item["id"]
                img_url = item["thumb_2048_url"]
                lon, lat = item["geometry"]["coordinates"]
                coord = np.radians([lat, lon])

                if not is_far_enough(coord, visited_tree, np.radians(DISTANCE_THRESHOLD_DEGREES)):
                    continue

                img_resp = requests.get(img_url)
                img = Image.open(BytesIO(img_resp.content)).convert("RGB")
                if not is_bright_enough(img):
                    continue


                grid_folder = os.path.join(OUTPUT_FOLDER, region_name)
                os.makedirs(grid_folder, exist_ok=True)
                filename = f"{img_id}.jpg"
                filepath = os.path.join(grid_folder, filename)      
                img.save(filepath)


                country = region_name

                all_rows.append([filename, lat, lon, region_name])
                visited_coords.append(coord)
                visited_tree = KDTree(visited_coords) if visited_coords else None

                images_downloaded += 1
                print(f"{filename} saved ({images_downloaded}/{IMAGES_PER_REGION})")

                if images_downloaded >= IMAGES_PER_REGION:
                    break

            except Exception as e:
                print(f"Failed to process image {item.get('id', 'unknown')}: {e}")

        page += 1
        time.sleep(0.1)

    if images_downloaded == 0:
        log_empty_grid(region_name)


    return all_rows


MAX_WORKERS = 20

def process_region(args):
    """
    Wrapper function for parallel processing of a single region.

    Args:
        args (tuple): A (region_name, bbox) pair.

    Returns:
        list: Metadata rows for successfully processed images in the region.
    """
    region_name, bbox = args
    print(f"🔄 Starting region: {region_name}")
    return fetch_images(region_name, bbox)

#Main script
if __name__ == "__main__":
    all_rows = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_region, item) for item in REGIONS.items()]
        for future in as_completed(futures):
            try:
                region_rows = future.result()
                all_rows.extend(region_rows)
            except Exception as e:
                print(f"Error in one region: {e}")

    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "lat", "lon", "grid_label"])
        writer.writerows(all_rows)

    print(f"\nAll done! {len(all_rows)} images saved to {OUTPUT_FOLDER}")