"""
MATWI sensor wear LSTM (simple baseline).

This script:
- Matches NPZ sensor files to wear labels using each set's merged.csv
- Keeps only wear types: flank_wear and flank_wear+adhesion (ignores adhesion)
- Splits by sets (train/val/test) exactly as requested
- Turns each NPZ into a sequence of window features:
    window = 1000 rows
    features per window = mean, std, rms of 5 channels
      (accel, acoustic, force_x, force_y, force_z)
- Trains one LSTM regression model
- Reports MSE/RMSE/MAE, and test MAE separately for the two types

You can paste this file directly into VS Code and run it.
"""

import argparse
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


# ---------- SETTINGS YOU REQUESTED ----------
FEATURE_KEYS = ["accel", "acoustic", "force_x", "force_y", "force_z"]
KEEP_TYPES = {"flank_wear", "flank_wear+adhesion"}

TRAIN_SETS = [1, 2, 5, 7, 8, 10, 11, 16, 17]
VAL_SETS   = [3, 6, 12, 14]
TEST_SETS  = [4, 9, 13, 15]


def set_seed(seed: int) -> None:
    """Make results more repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def npz_basename_from_row(sensor_window_path: str, sensor_name: str) -> str:
    """
    Get the expected .npz filename from merged.csv.

    merged.csv often stores a full absolute path in sensor_window_path.
    We only need the filename (basename), then we look for it locally.

    If sensor_window_path is empty/weird, we fall back to sensor_name and convert
    .csv -> .npz.
    """
    if isinstance(sensor_window_path, str) and sensor_window_path.strip():
        base = os.path.basename(sensor_window_path)
        if base.lower().endswith(".npz"):
            return base

    base = os.path.basename(str(sensor_name))
    if base.lower().endswith(".csv"):
        base = base[:-4] + ".npz"
    elif not base.lower().endswith(".npz"):
        base = base + ".npz"
    return base


def build_sensor_wear_matches(processed_root: Path, out_csv: Path) -> pd.DataFrame:
    """
    Create a clean table: one row per matched sensor NPZ file.

    Output columns:
      set, sensor_npz, wear, type
    """
    rows = []
    skipped_missing_npz = 0

    for s in range(1, 18):
        set_dir = processed_root / f"set{s}"
        merged_path = set_dir / "merged.csv"
        sens_dir = set_dir / "sensordata"

        if not merged_path.exists() or not sens_dir.exists():
            continue

        df = pd.read_csv(merged_path)

        # Keep only the two requested types, drop rows without wear.
        df = df[df["type"].isin(KEEP_TYPES)].dropna(subset=["wear"])

        for _, r in df.iterrows():
            base = npz_basename_from_row(
                sensor_window_path=str(r.get("sensor_window_path", "")),
                sensor_name=str(r.get("sensor_name", "")),
            )
            npz_path = sens_dir / base

            if not npz_path.exists():
                skipped_missing_npz += 1
                continue

            rows.append(
                {
                    "set": int(s),
                    "sensor_npz": str(npz_path),
                    "wear": float(r["wear"]),
                    "type": str(r["type"]),
                }
            )

    match_df = pd.DataFrame(rows).drop_duplicates(subset=["sensor_npz"]).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    match_df.to_csv(out_csv, index=False)

    print("\n=== MATCHING SUMMARY ===")
    print(f"Kept types: {sorted(KEEP_TYPES)}")
    print(f"Matched NPZ files: {len(match_df)}")
    print(f"Skipped (missing NPZ): {skipped_missing_npz}")

    return match_df


def window_stats_from_npz(npz_path: Path, window_size: int) -> np.ndarray:
    """
    Convert one NPZ file into a sequence of (mean, std, rms) features.

    Returns:
      feats: shape (T, 15) where 15 = 5 channels * 3 stats

    Last window:
      If it has < window_size rows, we "pad with zeros" conceptually,
      BUT we compute stats using only the real rows (so padding zeros do not affect stats).
    """
    data = np.load(npz_path)

    # Load and stack channels into X: (N, 5)
    cols = []
    for k in FEATURE_KEYS:
        if k not in data:
            raise KeyError(f"{npz_path} is missing key '{k}'. Keys found: {data.files}")
        arr = np.asarray(data[k], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(arr)

    n = len(cols[0])
    if any(len(c) != n for c in cols):
        raise ValueError(f"Channel length mismatch in {npz_path}")
    if n == 0:
        return np.zeros((1, 15), dtype=np.float32)

    X = np.stack(cols, axis=1)  # (N, 5)

    num_windows = int(math.ceil(n / window_size))
    feats = []

    for w in range(num_windows):
        start = w * window_size
        end = min((w + 1) * window_size, n)
        chunk = X[start:end]  # (valid_len, 5)

        # Stats computed only on real rows (no padded zeros included)
        mean = chunk.mean(axis=0)
        std = chunk.std(axis=0)  # population std (ddof=0)
        rms = np.sqrt((chunk ** 2).mean(axis=0))

        feats.append(np.concatenate([mean, std, rms], axis=0).astype(np.float32))

    return np.stack(feats, axis=0)  # (T, 15)


def fit_feature_scaler(train_df: pd.DataFrame, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean/std of the 15-dim window features using ONLY training data.
    """
    total = 0
    s1 = np.zeros(15, dtype=np.float64)
    s2 = np.zeros(15, dtype=np.float64)

    for p in train_df["sensor_npz"].tolist():
        feats = window_stats_from_npz(Path(p), window_size=window_size)  # (T, 15)
        total += feats.shape[0]
        s1 += feats.sum(axis=0)
        s2 += (feats.astype(np.float64) ** 2).sum(axis=0)

    if total == 0:
        raise ValueError("No training windows found. Check splits / matches.")

    mean = s1 / total
    var = (s2 / total) - (mean ** 2)
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)

    return mean.astype(np.float32), std.astype(np.float32)


def split_by_sets(match_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = match_df[match_df["set"].isin(TRAIN_SETS)].reset_index(drop=True)
    val_df   = match_df[match_df["set"].isin(VAL_SETS)].reset_index(drop=True)
    test_df  = match_df[match_df["set"].isin(TEST_SETS)].reset_index(drop=True)

    print("\n=== SPLIT SUMMARY ===")
    print(f"Train files: {len(train_df)}")
    print(f"Val files  : {len(val_df)}")
    print(f"Test files : {len(test_df)}")

    return train_df, val_df, test_df


class SensorWearDataset(Dataset):
    """
    One sample = one sensor NPZ file.

    Returns:
      x: (T, 15) normalized window features
      length: T (number of windows)
      y: wear label (float)
      t: type string
    """

    def __init__(self, df: pd.DataFrame, window_size: int, mean: np.ndarray, std: np.ndarray):
        self.df = df
        self.window_size = window_size
        self.mean = mean
        self.std = std
        self._cache = {}  # path -> np.ndarray (T, 15)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        npz_path = Path(row["sensor_npz"])

        if npz_path not in self._cache:
            self._cache[npz_path] = window_stats_from_npz(npz_path, self.window_size)

        x = self._cache[npz_path]  # (T, 15)
        x = (x - self.mean) / self.std  # normalize using training-set stats

        x = torch.from_numpy(x).float()
        y = torch.tensor(float(row["wear"]), dtype=torch.float32)
        t = str(row["type"])
        return x, x.shape[0], y, t


def collate_batch(batch):
    """
    Pad sequences so we can train with batches.
    """
    xs, lens, ys, ts = zip(*batch)

    # Sort by length (required for pack_padded_sequence with enforce_sorted=True)
    order = np.argsort(lens)[::-1]
    xs = [xs[i] for i in order]
    lens = [lens[i] for i in order]
    ys = torch.stack([ys[i] for i in order], dim=0)
    ts = [ts[i] for i in order]

    x_pad = pad_sequence(xs, batch_first=True)  # (B, Tmax, 15)
    lens_t = torch.tensor(lens, dtype=torch.long)

    return x_pad, lens_t, ys, ts


class LSTMWearRegressor(nn.Module):
    """
    LSTM -> last hidden state -> linear -> wear prediction (scalar)
    """
    def __init__(self, input_size=15, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0.0),
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x_pad: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x_pad, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (h_n, _) = self.lstm(packed)

        # h_n shape: (num_layers, B, hidden). Take last layer.
        last = h_n[-1]  # (B, hidden)
        out = self.head(last).squeeze(1)  # (B,)
        return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets, types = [], [], []

    for x_pad, lengths, y, t in loader:
        x_pad = x_pad.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        p = model(x_pad, lengths)

        preds.append(p.cpu().numpy())
        targets.append(y.cpu().numpy())
        types.extend(list(t))

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    mse = float(np.mean((preds - targets) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - targets)))

    return {"mse": mse, "rmse": rmse, "mae": mae, "preds": preds, "targets": targets, "types": types}


def train_one_epoch(model, loader, opt, device):
    model.train()
    loss_fn = nn.MSELoss()

    total_loss = 0.0
    n = 0

    for x_pad, lengths, y, _t in loader:
        x_pad = x_pad.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)
        pred = model(x_pad, lengths)
        loss = loss_fn(pred, y)
        loss.backward()

        # Helps stability (prevents exploding gradients)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        bs = y.shape[0]
        total_loss += loss.item() * bs
        n += bs

    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", type=str, default="processed",
                        help="Path to processed/ (contains set1..set17).")
    parser.add_argument("--out_dir", type=str, default="lstm_windowstats_runs",
                        help="Folder to save matches, scaler, and best_model.pt")
    parser.add_argument("--window_size", type=int, default=813)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild_matches", action="store_true")
    parser.add_argument("--refit_scaler", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    processed_root = Path(args.processed_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    matches_csv = out_dir / "sensor_wear_matches.csv"
    scaler_npz  = out_dir / "window_feature_scaler.npz"
    best_model  = out_dir / "best_model.pt"

    # 1) Build or load the matches table.
    if matches_csv.exists() and not args.rebuild_matches:
        match_df = pd.read_csv(matches_csv)
        print(f"[OK] Loaded matches: {matches_csv} (rows={len(match_df)})")
    else:
        match_df = build_sensor_wear_matches(processed_root, matches_csv)
        print(f"[OK] Saved matches to: {matches_csv}")

    # 2) Split by set IDs.
    train_df, val_df, test_df = split_by_sets(match_df)

    # 3) Fit or load feature scaler (mean/std) using ONLY training set.
    if scaler_npz.exists() and not args.refit_scaler:
        z = np.load(scaler_npz)
        mean, std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
        print(f"[OK] Loaded feature scaler: {scaler_npz}")
    else:
        print("\nComputing feature scaler from training set (one-time cost)...")
        mean, std = fit_feature_scaler(train_df, window_size=args.window_size)
        np.savez(scaler_npz, mean=mean, std=std)
        print(f"[OK] Saved feature scaler to: {scaler_npz}")

    # 4) Build PyTorch datasets/loaders.
    train_ds = SensorWearDataset(train_df, args.window_size, mean, std)
    val_ds   = SensorWearDataset(val_df, args.window_size, mean, std)
    test_ds  = SensorWearDataset(test_df, args.window_size, mean, std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = (device.type == "cuda")

    #Training, testing, validation....
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_batch, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_batch, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_batch, num_workers=0, pin_memory=pin)

    # 5) Model + optimizer.
    model = LSTMWearRegressor(hidden_size=args.hidden_size,
                              num_layers=args.num_layers,
                              dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 6) Train, selecting the best model by validation RMSE.
    best_val_rmse = float("inf")

    print("\n=== TRAINING ===")
    for epoch in range(1, args.epochs + 1):
        train_mse = train_one_epoch(model, train_loader, opt, device)
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_MSE={train_mse:.6f} | "
            f"val_MSE={val_metrics['mse']:.6f} "
            f"val_RMSE={val_metrics['rmse']:.6f} "
            f"val_MAE={val_metrics['mae']:.6f}"
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            torch.save(model.state_dict(), best_model)

    print(f"\n[OK] Best validation RMSE: {best_val_rmse:.6f}")
    print(f"[OK] Best model saved to: {best_model}")

    # 7) Test evaluation (load best model first).
    # This avoids a PyTorch warning on newer versions, while still being compatible with older ones.
    try:
        state = torch.load(best_model, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(best_model, map_location=device)
    model.load_state_dict(state)

    test_metrics = evaluate(model, test_loader, device)

    print("\n=== TEST EVALUATION ===")
    print(f"Test MSE : {test_metrics['mse']:.6f}")
    print(f"Test RMSE: {test_metrics['rmse']:.6f}")
    print(f"Test MAE : {test_metrics['mae']:.6f}")

    # MAE by type (requested): one model, but report per-type MAE.
    preds = test_metrics["preds"]
    targets = test_metrics["targets"]
    types = test_metrics["types"]

    # ===== PLOT: Predicted vs Actual wear (Test) =====
    import matplotlib.pyplot as plt

    unit = "µm"  # change to "mm" only if you later divide wear by 1000 in the script

    # Convert to numpy arrays (types is list of strings)
    preds_arr = np.asarray(preds, dtype=np.float64)
    targets_arr = np.asarray(targets, dtype=np.float64)
    types_arr = np.asarray(types, dtype=object)

    plt.figure(figsize=(7, 6))

    # Plot each type separately (different colors automatically)
    for wear_type in ["flank_wear", "flank_wear+adhesion"]:
        m = (types_arr == wear_type)
        if m.sum() == 0:
            continue
        plt.scatter(
            targets_arr[m], preds_arr[m],
            s=20, alpha=0.8,
            label=f"{wear_type} (n={m.sum()})"
        )

    # y = x line (ideal prediction)
    lo = float(min(targets_arr.min(), preds_arr.min()))
    hi = float(max(targets_arr.max(), preds_arr.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="gray", label="y = x")

    plt.xlabel(f"Actual wear ({unit})")
    plt.ylabel(f"Predicted wear ({unit})")
    plt.title("Test: Predicted vs Actual wear (LSTM)")
    plt.legend()
    plt.tight_layout()

    plot_path = out_dir / "test_pred_vs_actual.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"[OK] Saved plot -> {plot_path}")
    # ===============================================


    print("\n=== TEST MAE BY TYPE ===")
    for type_name in ["flank_wear", "flank_wear+adhesion"]:
        mask = np.array([t == type_name for t in types], dtype=bool)
        if mask.sum() == 0:
            print(f"{type_name:18s}  (no samples in test)")
            continue
        mae_t = float(np.mean(np.abs(preds[mask] - targets[mask])))
        print(f"{type_name:18s}  MAE {mae_t:.6f}  N={int(mask.sum())}")


if __name__ == "__main__":
    main()
