import os
import sys
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import folium
import geopandas as gpd
from shapely.geometry import box
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from grid_utils import generate_recursive_grids
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Configuration ──────────────────────────────────────────────────────────────
IMAGE_PATH = "tests/test_USA.jpg"
TOP_K      = 100    # final number of L8 predictions to keep
LEVEL      = 8          # grid depth
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_filename = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
output_map_path = os.path.join(SCRIPT_DIR, "results", f"test_prediction_map_{image_filename}.html")

# ─── Utilities ─────────────────────────────────────────────────────────────────
def load_model(model_path, label_map_path):
    """Load a ResNet-50 + label maps and return (model, label_map, inv_map)."""
    label_map     = torch.load(label_map_path)
    inv_label_map = {v: k for k, v in label_map.items()}
    num_classes   = len(label_map)

    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state.get("model_state_dict", state))

    model.to(DEVICE).eval()
    return model, label_map, inv_label_map


def get_adaptive_k(conf, thresholds=(0.85, 0.65, 0.40), max_k=6):
    if conf >= thresholds[0]:
        return 1
    if conf >= thresholds[1]:
        return 2
    if conf >= thresholds[2]:
        return 3
    return max_k


def masked_child_logits(parent_idxs, parent_probs,
                        parent2child, child_out, num_child_classes):
    """
    Build a logits tensor where *only* descendant classes get finite values.
    Everything else stays -inf, so the softmax assigns them zero probability.
    """
    device = child_out.device
    logits = torch.full((num_child_classes,), float("-inf"), device=device)

    for idx, p in zip(parent_idxs, parent_probs):
        idx   = idx.item()
        p     = p.item()
        for c in parent2child.get(idx, []):
            val = p * child_out[c]                    # simple linear fusion
            logits[c] = val if torch.isinf(logits[c]) else logits[c] + val

    return logits
# ────────────────────────────────────────────────────────────────────────────────

# ─── Load mappings & models ────────────────────────────────────────────────────
l1_to_l3 = torch.load(os.path.join(SCRIPT_DIR, "../label_maps/l1_to_l3.pth"))
l3_to_l5 = torch.load(os.path.join(SCRIPT_DIR, "../label_maps/l3_to_l5.pth"))
l5_to_l8 = torch.load(os.path.join(SCRIPT_DIR, "../label_maps/l5_to_l8.pth"))

l1_model, _, inv_l1_map = load_model(
    os.path.join(SCRIPT_DIR, "../../models/hierarchy_L1.pth"),
    os.path.join(SCRIPT_DIR, "../label_maps/label_map_L1.pth"))

l3_model, _, inv_l3_map = load_model(
    os.path.join(SCRIPT_DIR, "../../models/hierarchy_L3.pth"),
    os.path.join(SCRIPT_DIR, "../label_maps/label_map_L3.pth"))

l5_model, _, inv_l5_map = load_model(
    os.path.join(SCRIPT_DIR, "../../models/hierarchy_L5.pth"),
    os.path.join(SCRIPT_DIR, "../label_maps/label_map_L5.pth"))

l8_model, _, inv_l8_map = load_model(
    os.path.join(SCRIPT_DIR, "../../models/hierarchy_L8.pth"),
    os.path.join(SCRIPT_DIR, "../label_maps/label_map_L8.pth"))

# ─── Image pre-processing ──────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])
image = transform(Image.open(IMAGE_PATH).convert("RGB")).unsqueeze(0).to(DEVICE)

# ─── Hierarchical prediction ───────────────────────────────────────────────────
with torch.no_grad():
    # ── L1 ────────────────────────────────────────
    l1_probs  = torch.softmax(l1_model(image), dim=1)
    l1_k      = get_adaptive_k(l1_probs.max().item())
    l1_top_p, l1_top_i = torch.topk(l1_probs, l1_k)

    # ── L3 ────────────────────────────────────────
    l3_out    = l3_model(image)[0]                      # compute once
    l3_logits = masked_child_logits(l1_top_i[0], l1_top_p[0],
                                    l1_to_l3, l3_out, len(inv_l3_map))
    l3_probs  = torch.softmax(l3_logits, dim=0)
    l3_k      = get_adaptive_k(l3_probs.max().item())
    l3_top_p, l3_top_i = torch.topk(l3_probs, l3_k)

    # ── L5 ────────────────────────────────────────
    l5_out    = l5_model(image)[0]
    l5_logits = masked_child_logits(l3_top_i, l3_top_p,
                                    l3_to_l5, l5_out, len(inv_l5_map))
    l5_probs  = torch.softmax(l5_logits, dim=0)
    l5_k      = get_adaptive_k(l5_probs.max().item())
    l5_top_p, l5_top_i = torch.topk(l5_probs, l5_k)

    # ── L8 ────────────────────────────────────────
    l8_out    = l8_model(image)[0]
    l8_logits = masked_child_logits(l5_top_i, l5_top_p,
                                    l5_to_l8, l8_out, len(inv_l8_map))
    l8_probs  = torch.softmax(l8_logits, dim=0)
    topk_p, topk_i = torch.topk(l8_probs, TOP_K)

# ─── Debug printouts ───────────────────────────────────────────────────────────
print("L1 top probs:", l1_top_p)
print("L3 top probs:", l3_top_p)
print("L5 top probs:", l5_top_p)
print("L8 top 10 probs:", topk_p[:10])

# ─── Decode predicted grid IDs ────────────────────────────────────────────────
topk_idxs   = topk_i.cpu().tolist()
topk_probs  = topk_p.cpu().tolist()
topk_grids  = [inv_l8_map[i].replace("\\", "/") for i in topk_idxs]

# ─── Grid lookup & visualisation ───────────────────────────────────────────────
LAT_RANGE = (-90, 90)
LON_RANGE = (-180, 180)
all_grids = generate_recursive_grids(LAT_RANGE, LON_RANGE, LEVEL)

records = []
for rank, (grid_id, prob) in enumerate(zip(topk_grids, topk_probs)):
    if grid_id not in all_grids:
        continue
    lat_min, lon_min, lat_max, lon_max = all_grids[grid_id]
    geom   = box(lon_min, lat_min, lon_max, lat_max)
    color  = ("purple" if rank == 0 else
              "red"    if rank < 5  else
              "orange" if rank < 10 else
              "yellow")
    records.append({"grid_id": grid_id,
                    "geometry": geom,
                    "color": color,
                    "probability": prob})

gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")
for _, row in gdf.iterrows():
    folium.GeoJson(
        row["geometry"],
        style_function=lambda _,
            col=row["color"]: {"fillColor": col,
                               "color":      "black",
                               "weight":     0.5,
                               "fillOpacity": 0.6},
        tooltip=f"{row['grid_id']} ({row['probability']:.2%})"
    ).add_to(m)

m.save(output_map_path)
print("Saved map")
