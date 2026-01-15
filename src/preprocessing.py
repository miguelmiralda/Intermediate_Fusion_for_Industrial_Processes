from __future__ import annotations

import argparse
import json
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import transforms
from typing import Dict, Iterable, List
from PIL import Image, UnidentifiedImageError
from torchvision.models import ResNet101_Weights, resnet101

_NAME_TS = re.compile(r"(\d{4}-\d{2}-\d{2})[T_](\d{2})[:_](\d{2})[:_](\d{2})(?:[._](\d+))?")

IMG_SIZE = 224
IMG_MODE = "RGB"
IMG_CROP = "center"

SENSOR_RATE_HZ = 1625      # change here later (e.g., 1625)
WINDOW_SECONDS = 1.0       # symmetric window around image time (±0.5s)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def verify_rawset_files(*, folder_set_id: int = 2, labels_set_value: int = 1,) -> Dict[str, object]:
    base_dir = _project_root()
    data_dir = base_dir / "data" / "rawsets" / f"Set{folder_set_id}"
    img_dir = data_dir / "images"
    sensor_dir = data_dir / "sensordata"
    labels_csv = base_dir / "data" / "rawsets" / "labels.csv"
    sets_csv = base_dir / "data" / "rawsets" / "sets.csv"

    # load metadata
    labels_df = pd.read_csv(labels_csv)
    sets_df = pd.read_csv(sets_csv)

    labels_sel = labels_df[labels_df["Set"] == labels_set_value].copy()

    missing_images: List[str] = []
    missing_sensors: List[object] = []

    for idx, row in labels_sel.iterrows():
        img_path = img_dir / str(row["ImageName"])

        name = row["SensorName"]
        if pd.isna(name):
            continue

        if isinstance(name, float) and name.is_integer():
            name_str = str(int(name))
        else:
            name_str = str(name)

        sen_path = sensor_dir / name_str

        if not img_path.is_file():
            missing_images.append(row["ImageName"])
        if not sen_path.is_file():
            missing_sensors.append(row["SensorName"])

    dup_images = int(labels_sel["ImageName"].duplicated().sum())
    dup_sensors = int(labels_sel["SensorName"].duplicated().sum())

    return {
        "folder_set_id": folder_set_id,
        "labels_set_value": labels_set_value,
        "labels_rows": int(len(labels_sel)),
        "unique_images": int(labels_sel["ImageName"].nunique()),
        "unique_sensors": int(labels_sel["SensorName"].nunique()),
        "missing_images": missing_images,
        "missing_sensors": missing_sensors,
        "dup_images": dup_images,
        "dup_sensors": dup_sensors,
    }


#Sensor Processing...
_TS_VALUE_REGEX = re.compile(r"\d{4}-\d{2}-\d{2}[_ T]\d{2}:\d{2}:\d{2}[.,]\d+")

def _parse_dt_series(raw: pd.Series) -> pd.Series:
    # Parse date-times from strings that may use '_' between date and time or commas as decimal...
    s = raw.astype(str).str.strip()
    # replace only the first '_' (between date and time) with a space
    s = s.str.replace("_", " ", n=1, regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_datetime(s, errors="coerce", utc=True)

def load_sensor_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep=";")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    cols = set(df.columns)

    ts_candidates = [c for c in ("timestamp", "time", "date_time", "datetime", "dateandtime", "ts", "t") if c in cols]
    ts = None

    if ts_candidates:
        ts_col = ts_candidates[0]
        s_num = pd.to_numeric(df[ts_col], errors="coerce")
        if s_num.notna().mean() > 0.9:
            m = float(s_num.dropna().median())
            if m > 1e12:
                ts = pd.to_datetime(s_num, unit="ns", utc=True)
            elif m > 1e10:
                ts = pd.to_datetime(s_num, unit="ms", utc=True)
            else:
                ts = pd.to_datetime(s_num, unit="s", utc=True)
        else:
            ts = _parse_dt_series(df[ts_col])

        if ts.notna().any():
            df = df.assign(timestamp=ts)
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
            have_header = True
        else:
            have_header = False
    else:
        have_header = False

    if not have_header:
        # Re-read as headerless
        try:
            df = pd.read_csv(path, header=None)
        except Exception:
            df = pd.read_csv(path, sep=";", header=None)

        # Detect which column looks like timestamps by scanning values with regex
        ts_idx = None
        for i in range(min(10, df.shape[1])):  # inspect first N columns
            col = df.iloc[:, i].astype(str).str.strip()
            hit_rate = col.str.match(_TS_VALUE_REGEX).mean()
            if hit_rate > 0.8:
                ts_idx = i
                break

        if ts_idx is None:
            for i in range(df.shape[1]):
                s_num = pd.to_numeric(df.iloc[:, i], errors="coerce")
                if s_num.notna().mean() > 0.9:
                    ts_idx = i
                    break

        if ts_idx is None:
            raise ValueError(f"Could not find a timestamp column in headerless file: {path.name}")

        # Parse timestamp column
        ts = _parse_dt_series(df.iloc[:, ts_idx])
        if ts.notna().sum() == 0:
            # try numeric epoch
            s_num = pd.to_numeric(df.iloc[:, ts_idx], errors="coerce")
            m = float(s_num.dropna().median()) if s_num.notna().any() else 0.0
            if m > 1e12:
                ts = pd.to_datetime(s_num, unit="ns", utc=True)
            elif m > 1e10:
                ts = pd.to_datetime(s_num, unit="ms", utc=True)
            else:
                ts = pd.to_datetime(s_num, unit="s", utc=True)

        # Remaining columns are signals; preserve their order
        sig_cols = [j for j in range(df.shape[1]) if j != ts_idx]
        sig_df = df.iloc[:, sig_cols].apply(pd.to_numeric, errors="coerce")

        # Heuristic positional mapping (common 5 channels): [accel, acoustic, fx, fy, fz,]
        accel = acoustic = fx = fy = fz = np.nan
        if sig_df.shape[1] >= 5:
            accel, acoustic, fx, fy, fz  = [sig_df.iloc[:, k] for k in range(5)]
        elif sig_df.shape[1] == 4:
            accel, acoustic, fx, fy = [sig_df.iloc[:, k] for k in range(4)]
            fz = np.nan
        elif sig_df.shape[1] == 3:
            accel, acoustic, fx = [sig_df.iloc[:, k] for k in range(3)]
            fy = np.nan
            fz = np.nan

        df = pd.DataFrame(
            {
                "accel": accel,
                "acoustic": acoustic,
                "force_x": fx,
                "force_y": fy,
                "force_z": fz,
                "timestamp": ts,
            }
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
        if getattr(df["timestamp"].dt, "tz", None) is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    # Ensure expected columns exist
    expected = ["accel", "acoustic", "force_x", "force_y", "force_z", "timestamp"]
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan

    return df[expected]

def clean_sensor_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"])

    if df.empty:
        return df

    numeric_cols = [c for c in ["accel", "acoustic", "force_x", "force_y", "force_z"] if c in df.columns]

    for c in numeric_cols:
        series = df[c]
        if series.notna().sum() > 0:
            q1, q99 = series.quantile([0.01, 0.99])
            if pd.notna(q1) and pd.notna(q99) and q1 < q99:
                df[c] = series.clip(q1, q99)

    for c in ["force_x", "force_y", "force_z"]:
        if c in df.columns:
            df[c] = df[c].rolling(window=5, min_periods=1, center=True).median()

    # Interpolate gaps
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].interpolate(limit_direction="both")
    return df


def resample_sensor(df: pd.DataFrame, rate_hz: int = SENSOR_RATE_HZ) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    # Robust frequency: works for 1000, 1625, 2000, etc.
    period_ns = int(round(1e9 / rate_hz))  # nanoseconds per sample
    rule = f"{period_ns}ns"
    out = df.resample(rule).mean()
    out = out.interpolate(limit_direction="both").reset_index()
    return out


def extract_window(
        df_rs: pd.DataFrame,
        anchor_time: pd.Timestamp,
        window_seconds: float = 1.0,) -> pd.DataFrame | None:
    if df_rs.empty:
        return None

    # Ensure comparable (naive) times
    if isinstance(anchor_time, pd.Timestamp):
        if getattr(anchor_time, "tzinfo", None) is not None:
            anchor_time = anchor_time.tz_localize(None)

    ts = df_rs["timestamp"]
    if getattr(ts.dt, "tz", None) is not None:
        # make sensor timestamps naive as well
        ts = ts.dt.tz_localize(None)

    df_rs = df_rs.assign(timestamp=ts).sort_values("timestamp")

    half = pd.to_timedelta(window_seconds / 2.0, unit="s")
    t0, t1 = anchor_time - half, anchor_time + half

    smin, smax = df_rs["timestamp"].iloc[0], df_rs["timestamp"].iloc[-1]

    # No overlap at all → reject
    if t1 < smin or t0 > smax:
        return None

    # Clip to the available sensor range...
    t0c, t1c = max(t0, smin), min(t1, smax)
    out = df_rs[(df_rs["timestamp"] >= t0c) & (df_rs["timestamp"] <= t1c)]

    # If the actual overlap is tiny (e.g., <50% of desired window), reject
    overlap = (t1c - t0c).total_seconds()
    if overlap < 0.3 * window_seconds:
        return None

    return out.reset_index(drop=True)

#Image Processing Related...
def load_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as im:
            return im.copy()
    except FileNotFoundError:
        raise
    except UnidentifiedImageError as e:
        raise RuntimeError(f"Unreadable/corrupt image: {path}") from e


def build_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

def process_image_file(
        src_path: Path,
        dest_root: Path,
        img_size: int = 224,
        keep_name: bool = False,
        save_png: bool = False,
) -> Dict:
    """
    Process a single image: load, transform to tensor (normalized for ResNet),
    and optionally save a PNG for visualization.
    Returns metadata including the tensor and saved path.
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
        vis_transform = transforms.Compose(
            [
                transforms.Resize(int(img_size * 1.15)),
                transforms.CenterCrop(img_size),
            ]
        )
        vis_img = vis_transform(img)
        vis_img.save(out_path, format="PNG", optimize=True)
        meta["saved_path"] = str(out_path)

    return meta


def batch_process_images(
        image_names: Iterable[str],
        src_dir: Path,
        dest_dir: Path,
        img_size: int = 224,
        keep_name: bool = False,
        save_png: bool = True,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for name in image_names:
        meta = process_image_file(
            Path(src_dir) / name,
            Path(dest_dir),
            img_size=img_size,
            keep_name=keep_name,
            save_png=save_png,
            )
        meta["image_name"] = name
        meta["image_id"] = Path(meta["saved_path"]).stem if meta["saved_path"] else Path(name).stem
        rows.append(meta)
    return pd.DataFrame(rows)


def load_resnet_encoder(device="cpu"):
    weights = ResNet101_Weights.DEFAULT
    model = resnet101(weights=weights)

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


#Aligning and merging....
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
    # Parse date-times and ensuring they are naive...
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


def process_set(set_id: int, *, overwrite: bool = False) -> dict:
    # Process a single set; returns the counts dict...
    base_dir = Path(__file__).resolve().parents[1]  # project root
    P = _paths_for_set(base_dir, set_id)

    # Skip if outputs exist and no overwriting...
    merged_csv = P["proc_root"] / "merged.csv"
    log_json = P["proc_root"] / "preprocessing_log.json"
    if (not overwrite) and merged_csv.exists() and log_json.exists():
        # Read counts if present...
        try:
            counts = json.loads(log_json.read_text())["counts"]
            print(f"Set{set_id}: outputs exist → skipping (use --overwrite to rebuild).")
            return counts
        except Exception:
            print(f"Set{set_id}: outputs exist but log unreadable → rebuilding.")

    # Loading metadata...
    print(f"\nLoading labels and set metadata for Set{set_id}...")
    labels = pd.read_csv(P["labels_csv"])
    sets_meta = pd.read_csv(P["sets_csv"])

    labels = labels[labels["Set"] == set_id].copy()
    print(f"Rows for Set{set_id}: {len(labels)}")

    # Normalizing filenames...
    labels["ImageName"] = labels["ImageName"].astype(str).str.strip()
    labels["SensorName"] = labels["SensorName"].astype(str).str.strip()

    # Standardizing image timestamps...
    labels["ImageDateTime"] = _parse_time_naive(labels["ImageDateTime"])
    # Prefer aligning on SensorDateTime if available (same device clock)
    if "SensorDateTime" in labels.columns:
        labels["SensorDateTime"] = _parse_time_naive(labels["SensorDateTime"])

    # Processing images...
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

        tensor = row["tensor"]  # Processed image tensor
        feat = encode_image(tensor, encoder, device=device)  # torch tensor [2048]
        embeddings.append(feat.numpy())

    if len(embeddings) == 0:
        emb_df = pd.DataFrame(columns=[f"emb_{i}" for i in range(2048)])
    else:
        # Stack into 2D array [num_images, 2048]
        emb_arr = np.vstack(embeddings)  # shape (N, 2048)
        # Creating column names emb_0... emb_2047
        emb_col = [f"emb_{i}" for i in range(emb_arr.shape[1])]
        emb_df = pd.DataFrame(emb_arr, columns=emb_col, index=img_meta.index)

    # Free memory (tensors are large; no longer needed after embedding extraction)
    if "tensor" in img_meta.columns:
        img_meta["tensor"] = None

    # Dropping the tensor column before saving to CSV...
    if "tensor" in img_meta.columns:
        img_meta_no_tensor = img_meta.drop(columns=["tensor"])
    else:
        img_meta_no_tensor = img_meta

    # Saving plain image metadata without embeddings.
    img_meta_no_tensor.to_csv(P["proc_root"] / "image_meta.csv", index=False)

    # Concatenate image details (image_name, image_id) + embeddings into another CSV...
    emb_with_keys = pd.concat([img_meta_no_tensor[["image_name", "image_id"]], emb_df], axis=1)
    print(f"[Set{set_id}] img_meta rows: {len(img_meta)}")
    print(f"[Set{set_id}] embeddings rows: {len(emb_df)}")
    print(f"[Set{set_id}] emb_with_keys shape: {emb_with_keys.shape}")

    emb_with_keys.to_csv(P["proc_root"] / "image_embeddings.csv", index=False)

    emb_npz_path = P["proc_root"] / "image_embeddings.npz"

    np.savez_compressed(
        emb_npz_path,
        embeddings=emb_df.to_numpy(dtype=np.float32),   # (N, 2048)
        image_id=img_meta_no_tensor["image_id"].to_numpy(),
        image_name=img_meta_no_tensor["image_name"].to_numpy(),
    )

    img_lookup = {row["image_name"]: row.to_dict() for _, row in img_meta_no_tensor.iterrows()}

    # Align + window sensors per image -----
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
    valid_sens = labels["SensorName"].dropna().astype(str).str.strip()
    sample_sen = None
    for cand in valid_sens[:20]:
        p = P["sensor_dir"] / cand
        if p.is_file():
            sample_sen = cand
            break

    # Probe block...
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

        # image must have processed OK
        meta = img_lookup.get(imgname)
        if (meta is None) or (meta.get("status") != "ok") or (not meta.get("saved_path")):
            drop_counts["image_missing_or_corrupt"] += 1
            continue

        # Need timestamp (SensorDateTime is preferred for all sets, ImageDateTime is preferred for Set 1 - Surgical fix to get the results)...
        if set_id == 1:
            anchor_time = row["ImageDateTime"]
        else:
            anchor_time = (
                row["SensorDateTime"]
                if ("SensorDateTime" in row and pd.notna(row["SensorDateTime"]))
                else row["ImageDateTime"]
            )

        if pd.isna(anchor_time):
            drop_counts["timestamp_missing"] += 1
            continue

        # sensor file path
        if pd.isna(row["SensorName"]) or not str(row["SensorName"]).strip():
            drop_counts["sensor_file_missing"] += 1
            continue

        sen_path = P["sensor_dir"] / senname
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
                print(
                    f"[Set1 FAIL] {senname}  sensor=[{smin} .. {smax}]  "
                    f"img_ts={row.get('ImageDateTime')}  "
                    f"sen_ts={row.get('SensorDateTime')}  "
                    f"name_ts={_ts_from_sensor_name(senname)}  "
                    f"offset={set1_offset}"
                )
            drop_counts["insufficient_window_coverage"] += 1
            continue

        # save window as npz (one file per sample), use processed image_id as base
        image_id = meta["image_id"]
        sensor_base = Path(senname).stem
        sensor_npz = P["proc_sensor_dir"] / f"{sensor_base}.npz"
        sensor_csv_dir = P["proc_sensor_dir"] / "sensorcsvs"
        sensor_csv_dir.mkdir(parents=True, exist_ok=True)
        sensor_csv = sensor_csv_dir / f"{sensor_base}.csv"
        np.savez_compressed(
            sensor_npz,
            accel=win["accel"].values,
            acoustic=win["acoustic"].values,
            force_x=win["force_x"].values,
            force_y=win["force_y"].values,
            force_z=win["force_z"].values,
            timestamp=win["timestamp"].astype("datetime64[ns]").values
        )

        kept_rows.append(
            {
                "set": set_id,
                "image_name": imgname,
                "image_id": image_id,
                "sensor_name": senname,
                "sensor_window_path": str(sensor_npz),
                "anchor_time": anchor_time.isoformat(),
                "anchor_used": anchor_used,
                "window_seconds_used": ws_used,
                "sensor_window_start": win["timestamp"].iloc[0].isoformat(),
                "sensor_window_end": win["timestamp"].iloc[-1].isoformat(),
                "wear": row.get("wear", np.nan),
                "type": row.get("type", None),
            }
        )
        win.to_csv(sensor_csv, index=False)

    # Outputs...
    merged_df = pd.DataFrame(kept_rows)
    merged_df.to_csv(P["proc_root"] / "merged.csv", index=False)

    counts = {
        "labels_rows": int(len(labels)),
        "images_processed_ok": int((img_meta["status"] == "ok").sum()),
        "samples_kept": int(len(merged_df)),
        **drop_counts,
    }
    log = {
        "set": set_id,
        "params": {
            "img_size": IMG_SIZE,
            "img_mode": IMG_MODE,
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
        print(
            f"Set{sid:02d}: kept={c.get('samples_kept',0)} "
            f"img_bad={c.get('image_missing_or_corrupt',0)} "
            f"sens_bad={c.get('sensor_empty_or_bad',0)} "
            f"window_fail={c.get('insufficient_window_coverage',0)}"
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # default action when we click Run
        # sys.argv += ["--set", "2"]   # Processes only Specific set...
        sys.argv += ["--all"]         # Processes all the sets...
        sys.argv += ["--overwrite"]   # Overwrite existing processed outputs...
    main()