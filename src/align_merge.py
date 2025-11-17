#This module is used for defining alignment logic between sensor & image, merging, and exporting the processed final dataset...

from __future__ import annotations
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np

from sensor_processing import load_sensor_csv, clean_sensor_df, resample_sensor, extract_window
from image_processing import batch_process_images, load_resnet_encoder, encode_image
import re

_NAME_TS = re.compile(r"(\d{4}-\d{2}-\d{2})[T_](\d{2})[:_](\d{2})[:_](\d{2})(?:[._](\d+))?")

#Common factors in all sets
IMG_SIZE = 224
IMG_MODE = "RGB"
IMG_CROP = "center"

SENSOR_RATE_HZ = 1000        # resampled rate
WINDOW_SECONDS = 1.0         # symmetric window around image time (±0.5s)

def _ts_from_sensor_name(name: str) -> pd.Timestamp | pd.NaT:
    """
    Parse e.g. '...2022-11-23T10_14_39.119958.csv' -> naive pandas Timestamp.
    """
    m = _NAME_TS.search(name)
    if not m:
        return pd.NaT
    date, hh, mm, ss, frac = m.groups()
    frac = frac or "0"
    # build string 'YYYY-MM-DD HH:MM:SS.frac'
    s = f"{date} {hh}:{mm}:{ss}.{frac}"
    ts = pd.to_datetime(s, errors="coerce", utc=False)
    if isinstance(ts, pd.Timestamp) and getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return ts

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

# def _parse_crop_box_str(s: str) -> tuple[int,int,int,int] | None:
#     if not isinstance(s, str):
#         return None
#     # grab first four integers in order (robust to spaces/commas)
#     nums = re.findall(r"-?\d+", s)
#     if len(nums) < 4:
#         return None
#     L, T, R, B = map(int, nums[:4])
#     # optional sanity checks
#     if R <= L or B <= T:
#         return None
#     return (L, T, R, B)

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

    # #Join set parameters (copy columns across all rows)
    # set_crop = None
    # if "Set" in sets_meta.columns:
    #     set_params = sets_meta[sets_meta["Set"] == set_id].copy()
    #     if len(set_params):
    #         for col in set_params.columns:
    #             if col == "Set":
    #                 continue
    #             labels[col] = set_params[col].values[0]
    #         if "crop" in set_params.columns:
    #             raw_crop = str(set_params.iloc[0] ["crop"])
    #             set_crop = _parse_crop_box_str(raw_crop)

    #Processing images...
    print("Processing images ...")
    image_names = labels["ImageName"].tolist()
    img_meta = batch_process_images(
        image_names=image_names,
        src_dir=P["img_dir"],
        dest_dir=P["proc_img_dir"],
        img_size=IMG_SIZE,
        keep_name=False,
        save_png=True,
    )

    device = "cpu"
    encoder = load_resnet_encoder(device=device)

    embeddings = []
    for _, row in img_meta.iterrows():
        if row.get("status") != "ok" or row.get("tensor") is None:
            embeddings.append([np.nan] * 2048)
            continue

        tensor = row["tensor"] #Processed image tensor
        feat = encode_image(tensor, encoder, device=device) #torch tensor [2048]
        embeddings.append(feat.numpy())


    if len(embeddings) == 0:
        emb_df = pd.DataFrame(columns=[f"emb_{i}" for i in range(2048)])
    else:
        #Stack into 2D array [num_images, 2048]
        emb_arr = np.vstack(embeddings) #shape (N, 2048)
        #Creating column names emb_0... emb_2047
        emb_col = [f"emb_{i}" for i in range(emb_arr.shape[1])]
        emb_df = pd.DataFrame(emb_arr, columns=emb_col, index=img_meta.index)

    #Dropping the tensor column before saving to CSV...
    if "tensor" in img_meta.columns:
        img_meta_no_tensor = img_meta.drop(columns=["tensor"])
    else:
        img_meta_no_tensor = img_meta

    #Saving plain image metadata without embeddings.
    img_meta_no_tensor.to_csv(P["proc_root"] / "image_meta.csv", index=False)

    #Concatenate image details (image_name, image_id) + embeddings into another CSV...
    emb_with_keys = pd.concat([img_meta_no_tensor[["image_name", "image_id"]], emb_df], axis=1)
    print(f"[Set{set_id}] img_meta rows: {len(img_meta)}")
    print(f"[Set{set_id}] embeddings rows: {len(emb_df)}")
    print(f"[Set{set_id}] emb_with_keys shape: {emb_with_keys.shape}")

    emb_with_keys.to_csv(P["proc_root"] / "image_embeddings.csv", index=False)

    img_lookup = {row["image_name"]: row.to_dict() for _, row in img_meta_no_tensor.iterrows()}

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

    #Probe block...
    if sample_sen is not None:
        sdf_probe = load_sensor_csv(P["sensor_dir"] / sample_sen)
        print("Probe parsed rows:", len(sdf_probe), "ts nulls:", sdf_probe["timestamp"].isna().sum())
    else:
        print("Probe skipped: no valid SensorName found on disk for this set.")

    # ---- Set1: estimate a single clock offset (median) between ImageDateTime and sensor-name timestamp
    set1_offset = pd.Timedelta(0)
    if set_id == 1:
        deltas = []
        for _, r in labels.iterrows():
            img_ts = r.get("ImageDateTime", pd.NaT)
            name_ts = _ts_from_sensor_name(str(r.get("SensorName", "")))
            if pd.notna(img_ts) and pd.notna(name_ts):
                deltas.append(img_ts - name_ts)  # positive if image clock is ahead
        if deltas:
            # median is robust against outliers
            set1_offset = pd.Series(deltas).median()
            # sanity cap: ignore absurd offsets (> 6 hours)
            if abs(set1_offset.total_seconds()) > 6 * 3600:
                set1_offset = pd.Timedelta(0)
        print("Set1 estimated offset (image - sensor_name_ts):", set1_offset)

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
        if set_id == 1:
            win = None
            anchor_used = None
            ws_used = WINDOW_SECONDS

            img_ts = row["ImageDateTime"]
            sen_ts = row["SensorDateTime"] if ("SensorDateTime" in row and pd.notna(row["SensorDateTime"])) else pd.NaT
            name_ts = _ts_from_sensor_name(senname)

            strategies = []

            # A) Image time (as recorded)
            if pd.notna(img_ts):
                strategies.append(("image", img_ts, WINDOW_SECONDS))

            # B) SensorDateTime (if present)
            if pd.notna(sen_ts):
                strategies.append(("sensor", sen_ts, WINDOW_SECONDS))

            # C) Image shifted by per-set median offset (robust calibration)
            if pd.notna(img_ts) and set1_offset != pd.Timedelta(0):
                strategies.append(("image_set_offset", img_ts - set1_offset, WINDOW_SECONDS))

            # D) Timestamp parsed from sensor filename (often the true capture time)
            if pd.notna(name_ts):
                strategies.append(("sensor_name_ts", name_ts, WINDOW_SECONDS))

            # E) Wider window around best image guess (edge saver)
            if pd.notna(img_ts):
                strategies.append(("image_wide", img_ts, 1.5))

            # Try in order; accept partial overlap rule from extract_window
            for tag, at, ws in strategies:
                w = extract_window(sdf, anchor_time=at, window_seconds=ws)
                if w is not None and len(w) > 0:
                    win = w
                    anchor_used = tag
                    ws_used = ws
                    break

            # After trying all strategies above, try a nearest-sensor rescue
            if win is None:
                # build candidate anchors we tried (ignore NaT)
                cands = []
                if pd.notna(img_ts):
                    cands.append(img_ts)
                if pd.notna(sen_ts):
                    cands.append(sen_ts)
                if set1_offset != pd.Timedelta(0) and pd.notna(img_ts):
                    cands.append(img_ts - set1_offset)
                if pd.notna(name_ts):
                    cands.append(name_ts)

                if cands:
                    ts_series = sdf["timestamp"]
                    best_anchor = None
                    best_gap = None
                    for t in cands:
                        idx_near = (ts_series - t).abs().idxmin()
                        near_ts = ts_series.loc[idx_near]
                        gap = abs((near_ts - t).total_seconds())
                        if (best_gap is None) or (gap < best_gap):
                            best_gap = gap
                            best_anchor = near_ts

                    # if the nearest sample is reasonably close, center a slightly wider window there
                    if (best_anchor is not None) and (best_gap is not None) and best_gap <= 2.0:
                        w = extract_window(sdf, anchor_time=best_anchor, window_seconds=2.0)
                        if w is not None and len(w) > 0:
                            win = w
                            anchor_used = "nearest_sensor_time"
                            ws_used = 2.0

        else:
            # (unchanged for other sets)
            win = extract_window(sdf, anchor_time=anchor_time, window_seconds=WINDOW_SECONDS)
            anchor_used = "sensor" if ("SensorDateTime" in row and pd.notna(row["SensorDateTime"])) else "image"
            ws_used = WINDOW_SECONDS

        if win is None or len(win) == 0:
            if set_id == 1 and drop_counts["insufficient_window_coverage"] < 5:
                smin, smax = sdf["timestamp"].iloc[0], sdf["timestamp"].iloc[-1]
                print(f"[Set1 FAIL] {senname}  sensor=[{smin} .. {smax}]  "
                      f"img_ts={row.get('ImageDateTime')}  "
                      f"sen_ts={row.get('SensorDateTime')}  "
                      f"name_ts={_ts_from_sensor_name(senname)}  "
                      f"offset={set1_offset}")
            drop_counts["insufficient_window_coverage"] += 1
            continue

    # if win is None or len(win) == 0:
        #             drop_counts["insufficient_window_coverage"] += 1
        #             continue

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
            "anchor_used": anchor_used,
            "window_seconds_used": ws_used,
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
            # "img_crop": IMG_CROP,
            # "img_roi_box": set_crop,
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