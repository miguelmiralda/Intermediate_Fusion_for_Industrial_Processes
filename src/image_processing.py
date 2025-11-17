#This module has image resizing and normalization related content...

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Dict
from PIL import Image, UnidentifiedImageError
import pandas as pd
import torch
from torchvision import transforms
# import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn


def load_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as im:
            return im.copy()
    except FileNotFoundError:
        raise
    except UnidentifiedImageError as e:
        raise RuntimeError(f"Unreadable/corrupt image: {path}") from e

def build_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def process_image_file(
    src_path: Path,
    dest_root: Path,
    img_size: int = 224,
    keep_name: bool = False,
    save_png: bool = False
) -> Dict:
    """
    Process a single image: load, transform to tensor (normalized for ResNet),
    and optionally save a PNG for visualization.
    Returns metadata including the tensor and saved path (if applicable).
    """
    meta = {
        "source_path": str(src_path),
        "status": "ok",
        "tensor": None,
        "saved_path": None,
    }

    try:
        img = load_image(src_path)
    except FileNotFoundError:
        meta["status"] = "missing"
        return meta
    except RuntimeError:
        meta["status"] = "corrupt"
        return meta

    # Apply transform
    transform = build_transform(img_size)
    img_tensor = transform(img)
    meta["tensor"] = img_tensor

    if save_png:
        # Save a visualizable PNG version
        dest_root.mkdir(parents=True, exist_ok=True)
        out_name = f"{src_path.stem}.png" if keep_name else f"{src_path.stem.replace(' ', '_').lower()}.png"
        out_path = dest_root / out_name

        # Use a transform that only resizes + center crops (no normalization) for visualization
        vis_transform = transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
        ])
        vis_img = vis_transform(img)
        vis_img.save(out_path, format="PNG", optimize=True)
        meta["saved_path"] = str(out_path)

    return meta

def batch_process_images(image_names: Iterable[str],
                         src_dir: Path,
                         dest_dir: Path,
                         img_size: int = 224,
                         keep_name: bool = False,
                         save_png: bool = True) -> pd.DataFrame:

    #Process a list of image filenames from src_dir → dest_dir.
    rows: List[Dict] = []
    for name in image_names:
        meta = process_image_file(Path(src_dir) / name,
                                  Path(dest_dir),
                                  img_size=img_size,
                                  keep_name=keep_name,
                                  save_png=save_png)
        meta["image_name"] = name
        meta["image_id"] = Path(meta["saved_path"]).stem if meta["saved_path"] else Path(name).stem
        rows.append(meta)
    return pd.DataFrame(rows)

def load_resnet_encoder(device="cpu"):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    # Remove the final classification layer
    encoder = nn.Sequential(*list(model.children())[:-1])
    # Now encoder outputs: [batch, 2048, 1, 1]

    encoder.to(device)
    encoder.eval()
    return encoder

def encode_image(tensor, encoder, device="cpu"):
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        features = encoder(tensor)      # shape [1, 2048, 1, 1]
        features = features.flatten(1)  # shape [1, 2048]
    return features.squeeze(0).cpu()    # shape [2048]