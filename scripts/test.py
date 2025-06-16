import torch
label_map = torch.load("/label_maps/label_map_L1.pth")
print(label_map)
print(f"Number of classes: {len(label_map)}")