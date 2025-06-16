import os
import shutil

#Paths
level8_data_path = "../data/raw_images"
output_base_path = "../data"
hierarchy_levels = {
    "L1": 1,
    "L3": 3,
    "L5": 5,
}

#Return truncated path based on the level (1, 3, 5)
def truncate_grid_id(grid_path, level):

    parts = grid_path.split(os.sep)
    return os.sep.join(parts[:level]) if len(parts) >= level else None

#Flattens the folders for each hierarchy level
for level_name, level_depth in hierarchy_levels.items():
    print(f"📁 Processing {level_name}...")
    output_dir = os.path.join(output_base_path, f"hierarchy_{level_name}")
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(level8_data_path):
        rel_path = os.path.relpath(root, level8_data_path)
        rel_parts = rel_path.split(os.sep)

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

            # 🛠 Flatten: avoid duplicate filenames by prefixing with hash
            unique_name = f"{hash(src_path) & 0xffff}_{fname}"
            dst_path = os.path.join(class_folder, unique_name)

            shutil.copy2(src_path, dst_path)

print("Hierarchy L1, L3, and L5 datasets created")
#L8 remains in raw_images
