#This module is used for defining alignment logic between sensor & image, merging, and exporting the processed final dataset...

from __future__ import annotations
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np

from sensor_processing import load_sensor_csv, clean_sensor_df, resample_sensor, extract_window
from image_processing import batch_process_images

#Common factors in all sets
IMG_SIZE = (600, 400)
IMG_MODE = "RGB"
IMG_CROP = "center"

SENSOR_RATE_HZ = 1000        # resampled rate
WINDOW_SECONDS = 1.0         # symmetric window around image time (±0.5s)

def _parse_time_naive(ts: pd.Series) -> pd.Series:
    #Parse date-times and ensuring they are naive...
    s = pd.to_datetime(ts, errors="coerce", utc=False)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s

def _paths_for_set(base_dir: Path, set_id: int):
    data_dir = base_dir / "data"
    rawset_dir = data_dir / "rawsets" / f"Set{set_id}"
    img_dir = rawset_dir / "images"
    sensor_dir = rawset_dir / "sensordata"

    proc_root = data_dir / "processed" / f"set{set_id}"
    proc_img_dir = proc_root / "images"
    proc_sensor_dir = proc_root / "sensordata"
    proc_root.mkdir(parents=True, exist_ok=True)
    proc_img_dir.mkdir(parents=True, exist_ok=True)
    proc_sensor_dir.mkdir(parents=True, exist_ok=True)

    labels_csv = data_dir / "rawsets" / "labels.csv"
    sets_csv = data_dir / "rawsets" / "sets.csv"
    return {
        "data_dir": data_dir,
        "rawset_dir": rawset_dir,
        "img_dir": img_dir,
        "sensor_dir": sensor_dir,
        "proc_root": proc_root,
        "proc_img_dir": proc_img_dir,
        "proc_sensor_dir": proc_sensor_dir,
        "labels_csv": labels_csv,
        "sets_csv": sets_csv,
    }

#Processing the sets - Core of this module...
def process_set(set_id: int, *, overwrite: bool = False) -> dict:
    #Process a single set; returns the counts dict...
    base_dir = Path(__file__).resolve().parents[1]  #project root
    P = _paths_for_set(base_dir, set_id)

    #Skip if outputs exist and no overwriting...
    merged_csv = P["proc_root"] / "merged.csv"
    log_json   = P["proc_root"] / "preprocessing_log.json"
    if (not overwrite) and merged_csv.exists() and log_json.exists():
        #Read counts if present...
        try:
            counts = json.loads(log_json.read_text())["counts"]
            print(f"Set{set_id}: outputs exist → skipping (use --overwrite to rebuild).")
            return counts
        except Exception:
            print(f"Set{set_id}: outputs exist but log unreadable → rebuilding.")

    #Loading metadata...
    print(f"\nLoading labels and set metadata for Set{set_id}...")
    labels = pd.read_csv(P["labels_csv"])
    sets_meta = pd.read_csv(P["sets_csv"])

    labels = labels[labels["Set"] == set_id].copy()
    print(f"Rows for Set{set_id}: {len(labels)}")

    #Normalizing filenames...
    labels["ImageName"] = labels["ImageName"].astype(str).str.strip()
    labels["SensorName"] = labels["SensorName"].astype(str).str.strip()

    #Standardizing image timestamps...
    labels["ImageDateTime"] = _parse_time_naive(labels["ImageDateTime"])
    #Prefer aligning on SensorDateTime if available (same device clock)
    if "SensorDateTime" in labels.columns:
        labels["SensorDateTime"] = _parse_time_naive(labels["SensorDateTime"])

    #Join set parameters (copy columns across all rows)
    if "Set" in sets_meta.columns:
        set_params = sets_meta[sets_meta["Set"] == set_id].copy()
        if len(set_params):
            for col in set_params.columns:
                if col == "Set":
                    continue
                labels[col] = set_params[col].values[0]

    #Processing images...
    print("Processing images ...")
    image_names = labels["ImageName"].tolist()
    img_meta = batch_process_images(
        image_names=image_names,
        src_dir=P["img_dir"],
        dest_dir=P["proc_img_dir"],
        size=IMG_SIZE, mode=IMG_MODE, crop=IMG_CROP,
        keep_name=False,
    )
    img_meta.to_csv(P["proc_root"] / "image_meta.csv", index=False)
    img_lookup = {row["image_name"]: row.to_dict() for _, row in img_meta.iterrows()}

    #Align + window sensors per image -----
    kept_rows = []
    drop_counts = {
        "image_missing_or_corrupt": 0,
        "sensor_file_missing": 0,
        "sensor_empty_or_bad": 0,
        "insufficient_window_coverage": 0,
        "timestamp_missing": 0,
    }

    print("Aligning sensor windows around anchor timestamps ...")
    # robust probe: first non-null, non-empty SensorName that exists on disk
    valid_sens = (
        labels["SensorName"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    sample_sen = None
    for cand in valid_sens[:20]:
        p = P["sensor_dir"] / cand
        if p.is_file():
            sample_sen = cand
            break

    if sample_sen is not None:
        sdf_probe = load_sensor_csv(P["sensor_dir"] / sample_sen)
        print("Probe parsed rows:", len(sdf_probe), "ts nulls:", sdf_probe["timestamp"].isna().sum())
    else:
        print("Probe skipped: no valid SensorName found on disk for this set.")

    for _, row in labels.iterrows():
        imgname = row["ImageName"]
        senname = str(row["SensorName"]).strip()

        #image must have processed OK
        meta = img_lookup.get(imgname)
        if (meta is None) or (meta.get("status") != "ok") or (not meta.get("saved_path")):
            drop_counts["image_missing_or_corrupt"] += 1
            continue

        #Need timestamp (SensorDateTime is preferred for all sets, ImageDateTime is preferred for Set 1 - Surgical fix to get the results)...
        if set_id == 1:
            anchor_time = row["ImageDateTime"]
        else:
            anchor_time = row["SensorDateTime"] if ("SensorDateTime" in row and pd.notna(row["SensorDateTime"])) else row["ImageDateTime"]

        # anchor_time = row["SensorDateTime"] if "SensorDateTime" in row and pd.notna(row["SensorDateTime"]) else row["ImageDateTime"]
        if pd.isna(anchor_time):
            drop_counts["timestamp_missing"] += 1
            continue


        #sensor file path
        if pd.isna(row["SensorName"]) or not str(row["SensorName"]).strip():
            drop_counts["sensor_file_missing"] += 1
            continue

        sen_path = P["sensor_dir"] / senname
        if not sen_path.is_file():
            drop_counts["sensor_file_missing"] += 1
            continue

        #load/clean/resample
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

        #extract centered window
        win = extract_window(sdf, anchor_time=anchor_time, window_seconds=WINDOW_SECONDS)
        if win is None or len(win) == 0:
            drop_counts["insufficient_window_coverage"] += 1
            continue

        #save window as npz (one file per sample), use processed image_id as base
        image_id = meta["image_id"]
        sensor_npz = P["proc_sensor_dir"] / f"{image_id}.npz"
        np.savez_compressed(
            sensor_npz,
            timestamp=win["timestamp"].astype("datetime64[ns]").values,
            accel=win["accel"].values,
            acoustic=win["acoustic"].values,
            force_x=win["force_x"].values,
            force_y=win["force_y"].values,
            force_z=win["force_z"].values,
            rate_hz=SENSOR_RATE_HZ,
        )

        kept_rows.append({
            "set": set_id,
            "image_name": imgname,
            "image_id": image_id,
            "image_path": meta["saved_path"],
            "sensor_name": senname,
            "sensor_window_path": str(sensor_npz),
            "anchor_time": anchor_time.isoformat(),
            "sensor_window_start": win["timestamp"].iloc[0].isoformat(),
            "sensor_window_end":   win["timestamp"].iloc[-1].isoformat(),
            "wear": row.get("wear", np.nan),
            "type": row.get("type", None),
        })

    #Outputs...
    merged_df = pd.DataFrame(kept_rows)
    merged_df.to_csv(P["proc_root"] / "merged.csv", index=False)

    counts = {
        "labels_rows": int(len(labels)),
        "images_processed_ok": int((img_meta["status"] == "ok").sum()),
        "samples_kept": int(len(merged_df)),
        **drop_counts
    }
    log = {
        "set": set_id,
        "params": {
            "img_size": IMG_SIZE,
            "img_mode": IMG_MODE,
            "img_crop": IMG_CROP,
            "sensor_rate_hz": SENSOR_RATE_HZ,
            "window_seconds": WINDOW_SECONDS,
        },
        "counts": counts,
    }
    with open(P["proc_root"] / "preprocessing_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n=== Merging Summary (Set{set_id}) ===")
    print(json.dumps(counts, indent=2))
    print(f"\nSaved:\n  {P['proc_root']/'merged.csv'}\n  {P['proc_root']/'preprocessing_log.json'}")
    return counts


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Align, merge, and export MATWI sets.")
    parser.add_argument("--set", type=int, help="Process a single Set ID (1..17)")
    parser.add_argument("--all", action="store_true", help="Process all sets 1..17")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild outputs even if they exist")
    args = parser.parse_args()

    if not args.set and not args.all:
        print("Nothing to do. Use --set N or --all.")
        return

    set_ids = [args.set] if args.set else list(range(1, 18))
    summary = {}
    for sid in set_ids:
        counts = process_set(sid, overwrite=args.overwrite)
        summary[sid] = counts

    print("\n=== All Sets Summary ===")
    for sid in set_ids:
        c = summary[sid]
        print(f"Set{sid:02d}: kept={c.get('samples_kept',0)} "
              f"img_bad={c.get('image_missing_or_corrupt',0)} "
              f"sens_bad={c.get('sensor_empty_or_bad',0)} "
              f"window_fail={c.get('insufficient_window_coverage',0)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # default action when we click Run
        # sys.argv += ["--set", "2"]   #Processes only Specific set...
        sys.argv += ["--all"]   #Processes all the sets...
        sys.argv += ["--overwrite"]   #Enables this if the sets are already processed and needs to be overwrite...
    main()