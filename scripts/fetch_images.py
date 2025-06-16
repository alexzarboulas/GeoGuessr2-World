import os #File operations
import requests #HTTP requests to Mapillary API
import time #To avoid hitting API rate limits
import csv #For writing metadata
from PIL import Image 
from io import BytesIO  
from geopy.geocoders import Nominatim #Reverse geocoding to get country names
from geopy.extra.rate_limiter import RateLimiter #To avoid hitting geocoding API rate limits
from scipy.spatial import KDTree #For fast distance checking between images
import numpy as np
from grid_utils import get_all_grid_bboxes, log_empty_grid #Writing this file
from concurrent.futures import ThreadPoolExecutor, as_completed #For parallel processing of regions

# Mapillary API access token ->
#Replace yours here
ACCESS_TOKEN = "MLY|30161963110055518|c4771e278fd07ec1c609fe91c5c13f4e"

#Paths for images and metadata
OUTPUT_FOLDER = "../data/raw_images"
METADATA_PATH = "../data/metadata.csv"


#Configuration: images per region, brightness threshold, distance threshold (between images), and regions with bounding boxes
IMAGES_PER_REGION = 20
BRIGHTNESS_THRESHOLD = 40
DISTANCE_THRESHOLD_DEGREES = 0.0007
REGIONS = dict(list(get_all_grid_bboxes(level=8).items()))


#Initialize geocoder with rate limiting, and reverse geocoding function
geolocator = Nominatim(user_agent="geo_project")
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

#Brightness helper function
def is_bright_enough(img):
    grayscale = img.convert("L")
    mean_brightness = np.array(grayscale).mean()
    return mean_brightness > BRIGHTNESS_THRESHOLD

#Distance helper function, using KDTree for fast nearest neighbor search
def is_far_enough(coord, visited_tree, threshold_rad):
    if visited_tree is None:
        return True  # No previous images, so any new image is fine
    dist, _ = visited_tree.query(coord)  # Find nearest neighbor
    return dist >= threshold_rad  # Check if distance is above threshold



#Main function to query Mapillary API and process, download images
MAX_PAGES_PER_REGION = 1
def fetch_images(region_name, bbox):
    lat_min, lon_min, lat_max, lon_max = bbox #Unpack bounding box
    url = "https://graph.mapillary.com/images" #Mapillary API endpoint

    all_rows = []
    visited_coords = []
    visited_tree = None
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    page = 0 #Pagination for API requests
    images_downloaded = 0 #Count of successfully downloaded images

    while images_downloaded < IMAGES_PER_REGION and page < MAX_PAGES_PER_REGION: #Loop until enough images are downloaded
        params = {
            "access_token": ACCESS_TOKEN,
            "fields": "id,thumb_2048_url,geometry",
            "bbox": ",".join(map(str, [lon_min, lat_min, lon_max, lat_max])), 
            "limit": 10,  #Max images per API call
            "page": page #Pagination parameter
        }

        response = requests.get(url, params=params)
        if response.status_code != 200: #Check for successful response
            print(f"Error fetching images for {region_name} (page {page}): {response.status_code}")
            break

        data = response.json().get('data', []) #Extract image data
        if not data:
            print(f"No more images available for {region_name}")
            log_empty_grid(region_name)
            break


        for item in data: #Process each image in the response
            try:
                #Extract image metadata
                img_id = item["id"]
                img_url = item["thumb_2048_url"]
                lon, lat = item["geometry"]["coordinates"]
                coord = np.radians([lat, lon])

                #Check distance from previously downloaded images
                if not is_far_enough(coord, visited_tree, np.radians(DISTANCE_THRESHOLD_DEGREES)):
                    continue

                #Download and check brightness of the image
                img_resp = requests.get(img_url)
                img = Image.open(BytesIO(img_resp.content)).convert("RGB")
                if not is_bright_enough(img):
                    continue


                #Save image to disk
                grid_folder = os.path.join(OUTPUT_FOLDER, region_name)
                os.makedirs(grid_folder, exist_ok=True)
                filename = f"{img_id}.jpg"
                filepath = os.path.join(grid_folder, filename)      
                img.save(filepath)


                #Reverse geocode to get country name
                #location = reverse((lat, lon))
                #country = location.raw["address"].get("country", region_name)

                #Without reverse geocoding, much faster
                country = region_name

                #Store metadata
                all_rows.append([filename, lat, lon, region_name])
                visited_coords.append(coord)
                visited_tree = KDTree(visited_coords) if visited_coords else None

                #Increment count and print status
                images_downloaded += 1
                print(f"{filename} saved ({images_downloaded}/{IMAGES_PER_REGION})")

                #Break if enough images have been downloaded
                if images_downloaded >= IMAGES_PER_REGION:
                    break

            except Exception as e:
                print(f"Failed to process image {item.get('id', 'unknown')}: {e}")

        #Prepare for next page of results
        page += 1
        time.sleep(0.1)

    if images_downloaded == 0:
        log_empty_grid(region_name)


    #Return all collected metadata for this region
    return all_rows


MAX_WORKERS = 20 # Number of parallel threads (depends on your machine)

def process_region(args):
    region_name, bbox = args
    print(f"🔄 Starting region: {region_name}")
    return fetch_images(region_name, bbox)

#Main script execution
if __name__ == "__main__":
    all_rows = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_region, item) for item in REGIONS.items()]
        for future in as_completed(futures):
            try:
                region_rows = future.result()
                all_rows.extend(region_rows)
            except Exception as e:
                print(f"❌ Error in one region: {e}")

    # Save metadata
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "lat", "lon", "grid_label"])
        writer.writerows(all_rows)

    print(f"\n✅ All done! {len(all_rows)} images saved to {OUTPUT_FOLDER}")