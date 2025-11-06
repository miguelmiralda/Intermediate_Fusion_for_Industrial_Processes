#Sensor cleaning, resampling, and feature extractions are handled here...

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import re

_TS_VALUE_REGEX = re.compile(r"\d{4}-\d{2}-\d{2}[_ T]\d{2}:\d{2}:\d{2}[.,]\d+")

def _parse_dt_series(raw: pd.Series) -> pd.Series:
    """Parse datetimes from strings that may use '_' between date and time or commas as decimal sep."""
    s = raw.astype(str).str.strip()
    # replace only the *first* '_' (between date and time) with a space
    s = s.str.replace("_", " ", n=1, regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_datetime(s, errors="coerce", utc=True)

#Load & Read the Sensor related content (CSV) and Standardizing the columns...
def load_sensor_csv(path: Path) -> pd.DataFrame:
    # ---------- pass 1: try normal read with headers ----------
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep=";")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    cols = set(df.columns)

    ts_candidates = [c for c in ("timestamp","time","date_time","datetime","dateandtime","ts","t") if c in cols]
    ts = None

    if ts_candidates:
        # Try numeric epochs first, else string datetimes
        ts_col = ts_candidates[0]
        s_num = pd.to_numeric(df[ts_col], errors="coerce")
        if s_num.notna().mean() > 0.9:
            m = float(s_num.dropna().median())
            if m > 1e12:
                ts = pd.to_datetime(s_num, unit="ns", utc=True)
            elif m > 1e10:
                ts = pd.to_datetime(s_num, unit="ms", utc=True)
            else:
                ts = pd.to_datetime(s_num, unit="s",  utc=True)
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

    # ---------- pass 2: headerless fallback ----------
    if not have_header:
        # Re-read as headerless
        try:
            df = pd.read_csv(path, header=None)
        except Exception:
            df = pd.read_csv(path, sep=";", header=None)

        # Detect which column looks like timestamps by scanning values with regex
        ts_idx = None
        for i in range(min(10, df.shape[1])):  # inspect first N columns quickly
            col = df.iloc[:, i].astype(str).str.strip()
            hit_rate = col.str.match(_TS_VALUE_REGEX).mean()
            if hit_rate > 0.8:  # strong signal this column is datetime strings
                ts_idx = i
                break

        if ts_idx is None:
            # Try numeric epoch heuristic
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
                ts = pd.to_datetime(s_num, unit="s",  utc=True)

        # Remaining columns are signals; preserve their order
        sig_cols = [j for j in range(df.shape[1]) if j != ts_idx]
        sig_df = df.iloc[:, sig_cols].apply(pd.to_numeric, errors="coerce")

        # Heuristic positional mapping (common 5 channels): [fx, fy, fz, acoustic, accel]
        fx = fy = fz = acoustic = accel = np.nan
        if sig_df.shape[1] >= 5:
            fx, fy, fz, acoustic, accel = [sig_df.iloc[:, k] for k in range(5)]
        elif sig_df.shape[1] == 4:
            fx, fy, fz, acoustic = [sig_df.iloc[:, k] for k in range(4)]
        elif sig_df.shape[1] == 3:
            fx, fy, fz = [sig_df.iloc[:, k] for k in range(3)]
        # else: leave missing as NaN

        df = pd.DataFrame({
            "timestamp": ts,
            "force_x": fx,
            "force_y": fy,
            "force_z": fz,
            "acoustic": acoustic,
            "accel": accel,
        })

    # Ensure expected columns exist
    expected = ["timestamp","accel","acoustic","force_x","force_y","force_z"]
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan

    return df[expected]

def clean_sensor_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"])

    if df.empty:
        return df

    numeric_cols = [c for c in ["accel","acoustic","force_x","force_y","force_z"] if c in df.columns]

    for c in numeric_cols:
        series = df[c]
        if series.notna().sum() > 0:
            q1, q99 = series.quantile([0.01, 0.99])
            if pd.notna(q1) and pd.notna(q99) and q1 < q99:
                df[c] = series.clip(q1, q99)

    for c in ["force_x","force_y","force_z"]:
        if c in df.columns:
            df[c] = df[c].rolling(window=5, min_periods=1, center=True).median()

    # Interpolate gaps
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].interpolate(limit_direction="both")
    return df


def resample_sensor(df: pd.DataFrame, rate_hz: int = 1000) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    rule = f"{int(round(1000 / rate_hz))}ms"  # 'L' = milliseconds
    out = df.resample(rule).mean()
    out = out.interpolate(limit_direction="both").reset_index()
    return out

#Extracting the time window around an anchor
def extract_window(df_rs: pd.DataFrame,
                   anchor_time: pd.Timestamp,
                   window_seconds: float = 1.0) -> pd.DataFrame | None:
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

    # Clip to the available sensor range (accept partial coverage)
    t0c, t1c = max(t0, smin), min(t1, smax)
    out = df_rs[(df_rs["timestamp"] >= t0c) & (df_rs["timestamp"] <= t1c)]

    # If the actual overlap is tiny (e.g., <50% of desired window), reject
    overlap = (t1c - t0c).total_seconds()
    if overlap < 0.5 * window_seconds:
        return None

    return out.reset_index(drop=True)