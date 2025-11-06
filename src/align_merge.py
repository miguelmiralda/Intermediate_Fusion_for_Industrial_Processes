#This module is used for defining alignment logic between sensor & image, merging, and exporting the processed final dataset...

from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import numpy as np

from sensor_processing import (
    load_sensor_csv, clean_sensor_df, resample_sensor, extract_window
)
from image_processing import (
    batch_process_images
)

# ---------- Configuration ----------
SET_ID = 2
IMG_SIZE = (224, 224)
IMG_MODE = "RGB"
IMG_CROP = "center"

SENSOR_RATE_HZ = 1000        # resampled rate
WINDOW_SECONDS = 1.0         # symmetric window around image time (±0.5s)

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parents[1]          # project root
DATA_DIR = BASE_DIR / "data"
RAWSETS_DIR = DATA_DIR / "rawsets" / f"Set{SET_ID}"
IMG_DIR = RAWSETS_DIR / "images"
SENSOR_DIR = RAWSETS_DIR / "sensordata"

LABELS_CSV = DATA_DIR / "rawsets" / "labels.csv"
SETS_CSV = DATA_DIR / "rawsets" / "sets.csv"

PROC_ROOT = DATA_DIR / "processed" / f"set{SET_ID}"
PROC_IMG_DIR = PROC_ROOT / "images"
PROC_SENSOR_DIR = PROC_ROOT / "sensordata"
PROC_ROOT.mkdir(parents=True, exist_ok=True)
PROC_IMG_DIR.mkdir(parents=True, exist_ok=True)
PROC_SENSOR_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Utilities ----------
def _to_utc(ts: pd.Series) -> pd.Series:
    s = pd.to_datetime(ts, errors="coerce", utc=False)
    #localizing as UTC...
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s

#Load metadata
print("Loading labels and set metadata...")
labels = pd.read_csv(LABELS_CSV)
sets_meta = pd.read_csv(SETS_CSV)

labels = labels[labels["Set"] == SET_ID].copy()
print(f"Rows for Set{SET_ID}: {len(labels)}")

# Basic cleaning of key columns
labels["ImageName"] = labels["ImageName"].astype(str).str.strip()
labels["SensorName"] = labels["SensorName"].astype(str).str.strip()

#Process images (standardize & save)...
print("Processing images ...")
image_names = labels["ImageName"].tolist()
img_meta = batch_process_images(
    image_names=image_names,
    src_dir=IMG_DIR,
    dest_dir=PROC_IMG_DIR,
    size=IMG_SIZE,
    mode=IMG_MODE,
    crop=IMG_CROP,
    keep_name=False
)
img_meta.to_csv(PROC_ROOT / "image_meta.csv", index=False)

# Map name → plain dict (not Series), so boolean checks are unambiguous
img_lookup = {row["image_name"]: row.to_dict() for _, row in img_meta.iterrows()}

#Preparing labels (timestamps, join set params)
labels["ImageDateTime"] = _to_utc(labels["ImageDateTime"])
labels["SensorDateTime"] = _to_utc(labels["SensorDateTime"])

# Join cutting parameters from sets.csv on Set id
if "Set" in sets_meta.columns:
    set_params = sets_meta[sets_meta["Set"] == SET_ID].copy()
    # carry all columns except duplicates
    for col in set_params.columns:
        if col in ["Set"]:
            continue
        labels[col] = set_params[col].values[0] if len(set_params) else np.nan

#Align + window sensors per image
kept_rows = []
drop_counts = {
    "image_missing_or_corrupt": 0,
    "sensor_file_missing": 0,
    "sensor_empty_or_bad": 0,
    "insufficient_window_coverage": 0,
    "timestamp_missing": 0,
}

print("Aligning sensor windows around image timestamps ...")
# print("First label indices:", list(labels.index[:5]))

# Quick probe
sample_sen = labels["SensorName"].iloc[0]
sdf = load_sensor_csv(SENSOR_DIR / str(sample_sen))
print("Probe parsed rows:", len(sdf), "ts nulls:", sdf["timestamp"].isna().sum())

#Lists which SensorName files are missing on disk...
unique_sensors = sorted(set(labels["SensorName"].astype(str).str.strip()))
missing_sensor_files = []
for n in unique_sensors:
    p = SENSOR_DIR / n
    if not p.is_file():
        missing_sensor_files.append(n)

print("Missing sensor files (unique count={}):".format(len(missing_sensor_files)))
for n in missing_sensor_files[:25]:
    print("  [MISS_SENSOR]", n)
if len(missing_sensor_files) > 25:
    print("  ... ({} more)".format(len(missing_sensor_files) - 25))

#Main functional loop...
for idx, row in labels.iterrows():
# for i, (_, row) in enumerate(labels.reset_index(drop=True).iterrows()):
    imgname = row["ImageName"]
    senname = row["SensorName"]

    # image must have processed OK
    meta = img_lookup.get(imgname)
    if (meta is None) or (meta.get("status") != "ok") or (not meta.get("saved_path")):
        drop_counts["image_missing_or_corrupt"] += 1
        continue


    # timestamp needed
    anchor_time = row["SensorDateTime"] if pd.notna(row.get("SensorDateTime")) else row["ImageDateTime"]

    # sensor file path
    senname = str(row["SensorName"]).strip()
    sen_path = SENSOR_DIR / senname
    if not sen_path.is_file():
        drop_counts["sensor_file_missing"] += 1
        continue

    # load/clean/resample
    try:
        sdf = load_sensor_csv(sen_path)
        sdf = clean_sensor_df(sdf)
        if len(sdf) == 0:
            drop_counts["sensor_empty_or_bad"] += 1
            continue
        sdf = resample_sensor(sdf, rate_hz=SENSOR_RATE_HZ)
    except Exception:
        drop_counts["sensor_empty_or_bad"] += 1
        continue

    # extract centered window
    win = extract_window(sdf, anchor_time=anchor_time, window_seconds=WINDOW_SECONDS)
    if win is None or len(win) == 0:
        drop_counts["insufficient_window_coverage"] += 1
        continue

    # save window as npz (one file per sample), use processed image_id as base
    image_id = meta["image_id"]
    sensor_npz = PROC_SENSOR_DIR / f"{image_id}.npz"
    np.savez_compressed(
        sensor_npz,
        timestamp=win["timestamp"].astype("datetime64[ns]").values,
        accel=win["accel"].values,
        acoustic=win["acoustic"].values,
        force_x=win["force_x"].values,
        force_y=win["force_y"].values,
        force_z=win["force_z"].values,
        rate_hz=SENSOR_RATE_HZ
    )

    # record merged row
    kept_rows.append({
        "set": SET_ID,
        "image_name": imgname,
        "image_id": image_id,
        "image_path": meta["saved_path"],
        "sensor_name": senname,
        "sensor_window_path": str(sensor_npz),
        "image_time_utc": anchor_time.isoformat(),
        "sensor_window_start_utc": win["timestamp"].iloc[0].isoformat(),
        "sensor_window_end_utc": win["timestamp"].iloc[-1].isoformat(),
        "wear": row.get("wear", np.nan),
        "type": row.get("type", None),
    })

#Persist merged index + log
merged_df = pd.DataFrame(kept_rows)
merged_df.to_csv(PROC_ROOT / "merged.csv", index=False)

log = {
    "set": SET_ID,
    "params": {
        "img_size": IMG_SIZE,
        "img_mode": IMG_MODE,
        "img_crop": IMG_CROP,
        "sensor_rate_hz": SENSOR_RATE_HZ,
        "window_seconds": WINDOW_SECONDS,
    },
    "counts": {
        "labels_rows": int(len(labels)),
        "images_processed_ok": int((img_meta["status"] == "ok").sum()),
        "samples_kept": int(len(merged_df)),
        **drop_counts
    },
}
with open(PROC_ROOT / "preprocessing_log.json", "w") as f:
    json.dump(log, f, indent=2)

print(f"\n=== Merging Summary (Set{SET_ID}) ===")
print(json.dumps(log["counts"], indent=2))
print(f"\nSaved:\n  {PROC_ROOT/'merged.csv'}\n  {PROC_ROOT/'preprocessing_log.json'}")
print("Done.")