#Entry point. In this module, we have done orchestrates full pipeline, reading/writing the files, loading the metadata...

import pandas as pd
from pathlib import Path

#Dataset path definition
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "rawsets" / "Set2"
IMG_DIR = DATA_DIR / "images"
SENSOR_DIR = DATA_DIR / "sensordata"
LABELS_CSV = BASE_DIR / "data" / "rawsets" / "labels.csv"
SETS_CSV = BASE_DIR / "data" / "rawsets" / "sets.csv"

#Loading the metadata from labels.csv and sets.csv. And Using only Set2 for processing...
print("Loading metadata...") #For flow checking, can be removed.
labels_df = pd.read_csv(LABELS_CSV)
sets_df = pd.read_csv(SETS_CSV)

labels_set2 = labels_df[labels_df["Set"] == 1].copy()
print(f"Loaded {len(labels_set2)} labels rows for Set2.") #For flow checking, can be removed.

#Checking whether the files (sensor & image) listed in labels.csv actually exists...
missing_images, missing_sensors = [], []
for idx, row in labels_set2.iterrows():
    img_path = IMG_DIR / str(row["ImageName"])
    name = row["SensorName"]
    if pd.isna(name):
        print(f"Warning: missing SensorName at row {idx} — skipping")
        continue
    if isinstance(name, float) and name.is_integer():
        name_str = str(int(name))
    else:
        name_str = str(name)
    sen_path = SENSOR_DIR / name_str

    if not img_path.is_file():
        missing_images.append(row["ImageName"])
    if not sen_path.is_file():
        missing_sensors.append(row["SensorName"])

#Detecting duplicates...
dup_images = labels_set2["ImageName"].duplicated().sum()
dup_sensors = labels_set2["SensorName"].duplicated().sum()

# --- 5. Print summary ---
print("\n=== File Verification Summary ===")
print(f"Images listed in labels.csv:  {labels_set2['ImageName'].nunique()}")
print(f"Sensor files listed:          {labels_set2['SensorName'].nunique()}")
print(f"Missing images:               {len(missing_images)}")
print(f"Missing sensors:              {len(missing_sensors)}")
print(f"Duplicate image names:        {dup_images}")
print(f"Duplicate sensor names:       {dup_sensors}")

print("\nVerification complete.")