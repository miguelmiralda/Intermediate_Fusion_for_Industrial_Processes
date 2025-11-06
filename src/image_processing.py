#This module has image resizing and normalization related content...

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, List, Dict
from PIL import Image, UnidentifiedImageError
import pandas as pd

def load_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as im:
            return im.copy()
    except FileNotFoundError:
        raise
    except UnidentifiedImageError as e:
        raise RuntimeError(f"Unreadable/corrupt image: {path}") from e

def to_mode(img: Image.Image, mode: str = "RGB") -> Image.Image:
    #Standardizing the color mode...
    if img.mode == mode:
        return img
    # common special cases (e.g., 16-bit)
    if mode == "RGB":
        return img.convert("RGB")
    if mode == "L":
        # convert via RGB to avoid palette quirks
        return img.convert("RGB").convert("L")
    return img.convert(mode)

def resize_and_crop(img: Image.Image,
                    size: Tuple[int, int] = (224, 224),
                    crop: str = "center") -> Image.Image:
    target_w, target_h = size
    w, h = img.size

    #scaling, so the smaller edge fits target...
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)

    #computing the crop box
    x_extra, y_extra = new_w - target_w, new_h - target_h
    positions = {
        "center":       (x_extra // 2,           y_extra // 2),
        "top-left":     (0,                       0),
        "top":          (x_extra // 2,            0),
        "top-right":    (x_extra,                 0),
        "left":         (0,                       y_extra // 2),
        "right":        (x_extra,                 y_extra // 2),
        "bottom-left":  (0,                       y_extra),
        "bottom":       (x_extra // 2,            y_extra),
        "bottom-right": (x_extra,                 y_extra),
    }
    if crop not in positions:
        crop = "center"
    x0, y0 = positions[crop]
    box = (x0, y0, x0 + target_w, y0 + target_h)
    return img.crop(box)

def process_image_file(src_path: Path,
                       dest_root: Path,
                       size: Tuple[int, int] = (224, 224),
                       mode: str = "RGB",
                       crop: str = "center",
                       keep_name: bool = False) -> Dict:

    #Load → standardize mode → resize/crop → save PNG.
    meta = {
        "source_path": str(src_path),
        "status": "ok",
        "orig_w": None, "orig_h": None, "orig_mode": None,
        "final_w": size[0], "final_h": size[1], "final_mode": mode,
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

    meta["orig_w"], meta["orig_h"] = img.size
    meta["orig_mode"] = img.mode

    img = to_mode(img, mode=mode)
    img = resize_and_crop(img, size=size, crop=crop)

    dest_root.mkdir(parents=True, exist_ok=True)

    if keep_name:
        out_name = f"{src_path.stem}.png"
    else:
        # safe, normalized name
        out_name = f"{src_path.stem.replace(' ', '_').lower()}.png"

    out_path = dest_root / out_name
    img.save(out_path, format="PNG", optimize=True)
    meta["saved_path"] = str(out_path)
    return meta

def batch_process_images(image_names: Iterable[str],
                         src_dir: Path,
                         dest_dir: Path,
                         size: Tuple[int, int] = (224, 224),
                         mode: str = "RGB",
                         crop: str = "center",
                         keep_name: bool = False) -> pd.DataFrame:

    #Process a list of image filenames from src_dir → dest_dir.
    rows: List[Dict] = []
    for name in image_names:
        meta = process_image_file(Path(src_dir) / name,
                                  Path(dest_dir),
                                  size=size, mode=mode, crop=crop, keep_name=keep_name)
        # include a normalized `image_id` for downstream merges
        meta["image_name"] = name
        meta["image_id"] = Path(meta["saved_path"]).stem if meta["saved_path"] else Path(name).stem
        rows.append(meta)
    return pd.DataFrame(rows)