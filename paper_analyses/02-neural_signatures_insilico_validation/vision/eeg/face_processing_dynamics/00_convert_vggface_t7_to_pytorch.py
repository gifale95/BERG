# Convert the VGG Face weights `vgg_face_torch.tar.gz` into a '.pth' file.

# The `vgg_face_torch.tar.gz` are downloaded from:
# https://www.robots.ox.ac.uk/~vgg/software/vgg_face/

# And then extracted using `tar -xvzf vgg_face_torch.tar.gz`.

import torch
import torch.nn as nn
from torchvision.models import vgg16
import torchfile
import numpy as np

# -------------------------------
# Paths
# -------------------------------
TORCH7_PATH = "../vgg_face_torch/VGG_FACE.t7"  # path to the downloaded file
OUTPUT_PTH = "../vgg_face_pytorch.pth"   # output PyTorch weights file

# -------------------------------
# 1) Load Torch7 VGG-Face
# -------------------------------
print("Loading Torch7 model...")
torch7 = torchfile.load(TORCH7_PATH)
modules = torch7['modules']  # top-level sequential modules

# -------------------------------
# 2) Flatten nested modules and extract weight layers
# -------------------------------
def extract_weight_layers(modules, prefix=""):
    """
    Recursively traverse Torch7 modules to find conv/fc layers with weights.
    Returns list of tuples: (layer_name, weight, bias)
    """
    layers = []
    for idx, m in enumerate(modules):
        layer_name = f"{prefix}_{idx}"
        # If module has nested modules
        if hasattr(m, 'modules') and m.modules:
            layers.extend(extract_weight_layers(m.modules, prefix=layer_name))
        else:
            # Try weight/bias attributes
            w = getattr(m, 'weight', None)
            b = getattr(m, 'bias', None)
            # Some TorchObjects store them in _parameters list
            if w is None and hasattr(m, "_parameters"):
                params = m._parameters
                if params and len(params) >= 2:
                    w, b = params[0], params[1]
            if w is not None and b is not None:
                layers.append((layer_name, w, b))
    return layers

weight_layers = extract_weight_layers(modules)
print(f"Found {len(weight_layers)} layers with weights.")

# -------------------------------
# 3) Create PyTorch VGG16
# -------------------------------
model = vgg16()
# Adjust classifier to match VGG-Face
model.classifier[6] = nn.Linear(4096, 2622)
state_dict = model.state_dict()

# -------------------------------
# 4) Copy weights into PyTorch
# -------------------------------
print("Copying weights into PyTorch model...")
# List of PyTorch keys for weights/biases in order
pytorch_keys = [k for k in state_dict.keys() if "weight" in k or "bias" in k]

for (layer_name, w, b), key_w, key_b in zip(weight_layers, pytorch_keys[::2], pytorch_keys[1::2]):
    target_w_shape = state_dict[key_w].shape
    w = torch.tensor(w).view(target_w_shape)
    b = torch.tensor(b).view_as(state_dict[key_b])
    state_dict[key_w] = w
    state_dict[key_b] = b

model.load_state_dict(state_dict)
print("Weights copied successfully.")

# -------------------------------
# 5) Save PyTorch weights
# -------------------------------
torch.save(state_dict, OUTPUT_PTH)
print(f"Saved PyTorch weights to {OUTPUT_PTH}")
