"""
Converts the nested Level 8 folder structure into flat image datasets
for hierarchy levels L1, L3, and L5.

- Truncates grid IDs to target depth (e.g., 0/1/2/3/0/1/1/2 → 0 for L1)
- Copies and renames images into folders like `data/hierarchy_L3/0_1_2`
- Prepares clean datasets for model training by organizing images
  into class-based folders without nesting.

Author: Alexander Zarboulas
Date: 2025-06-18
"""

#Import libraries
import os
import shutil

#Configuration
level8_data_path = "../data/raw_images"
output_base_path = "../data"

#Target hierarchy levels with their respective depths
hierarchy_levels = {
    "L1": 1,
    "L3": 3,
    "L5": 5,
}


def truncate_grid_id(grid_path, level):
    """
    Truncates a nested grid path to the target hierarchy depth.

    Args:
        grid_path (str): The relative path (e.g., '0/1/2/3/0/1/1/2')
        level (int): The number of levels to retain

    Returns:
        str: Truncated path (e.g., '0_1_2'), or None if too shallow
    """
    parts = grid_path.split(os.sep)
    return os.sep.join(parts[:level]) if len(parts) >= level else None

#Process each hierarchy level
for level_name, level_depth in hierarchy_levels.items():
    print(f"Processing {level_name}...")
    output_dir = os.path.join(output_base_path, f"hierarchy_{level_name}")
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(level8_data_path):
        rel_path = os.path.relpath(root, level8_data_path)
        rel_parts = rel_path.split(os.sep)

        #Only process directories that represent full L8 grid depth
        if len(rel_parts) != 8:
            continue

        truncated = truncate_grid_id(rel_path, level_depth)
        if truncated is None:
            continue

        class_folder = os.path.join(output_dir, truncated)
        os.makedirs(class_folder, exist_ok=True)

        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            src_path = os.path.join(root, fname)

            #Prefix filename with hash to avoid collisions
            unique_name = f"{hash(src_path) & 0xffff}_{fname}"
            dst_path = os.path.join(class_folder, unique_name)

            shutil.copy2(src_path, dst_path)

print("Hierarchy L1, L3, and L5 datasets created in ../data")
#L8 remains nested under raw_images
