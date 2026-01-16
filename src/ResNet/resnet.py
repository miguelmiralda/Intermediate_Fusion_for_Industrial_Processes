# ============================================================
#  MULTI-HEAD TOOL WEAR TRAINING (USING PRECOMPUTED EMBEDDINGS)
#  + Hyperparameter sweep over LR and head embedding dim
#
#  Model A: 2-head (flank_wear, adhesion) on TRAIN_SETS
#  Model B: 1-head (flank_wear+adhesion) on FWAD_SETS (separate experiment)
#
#  Training pairing:
#    - sorted by time within type
#    - (i,j) for all i,j INCLUDING self-pairs (i,i)
#    - optional cap: exclude near pairs (keep only j==i OR |j-i| > K)
#
#  Testing pairing ("ref" mode):
#    - first image is reference; pairs (ref, j) for all j incl (ref,ref)
#
#  Inputs expected per set:
#    data/processed/set{sid}/merged.csv
#    data/processed/set{sid}/image_embeddings.npz
#
#  image_embeddings.npz must contain:
#    embeddings: (N, 2048) float
#    image_id:   (N,)      stringable
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Config
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Model A split (MATWI paper split)
TRAIN_SETS = [1, 2, 5, 7, 8, 10, 11]
VAL_SETS   = [3, 6, 12]
TEST_SETS  = [4, 9, 13]

# Model B split (sets containing flank_wear+adhesion) — separate experiment
FWAD_TRAIN_SETS = [3, 13, 16]
FWAD_VAL_SETS   = [12]
FWAD_TEST_SETS  = [17]

PAIR_BY_TIME_COL = "anchor_time"

# Cap for training (paper-style): exclude near/consecutive pairs.
# Keep (i,i) always, and keep (i,j) only if |j-i| > K after sorting by time.
# Example: K=1 => 1st pairs with 3rd; K=2 => 1st pairs with 4th.
# Set to None to disable.
MAX_INDEX_GAP_K = None

WEAR_COL = "wear"
TYPE_COL = "type"
IMAGE_ID_COL = "image_id"

BATCH_SIZE = 16
EPOCHS = 20
EMBED_IN_DIM = 2048

TYPE_FW = "flank_wear"
TYPE_AD = "adhesion"
TYPE_FWAD = "flank_wear+adhesion"

# ----------------------------
# Hyperparameter sweep config
# ----------------------------

LR_GRID = [1e-3, 5e-4, 1e-4]
HEAD_DIM_GRID = [8, 16, 32, 64, 128]

# Set False if you don’t want to export head_embeddings csvs for every sweep run
EXPORT_EMBEDDINGS_EACH_RUN = False


# ----------------------------
# Utilities: IO
# ----------------------------

def load_embeddings_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)

    if "embeddings" not in data or "image_id" not in data:
        raise KeyError(
            f"{npz_path} must contain keys: 'embeddings' and 'image_id' "
            f"(found: {list(data.keys())})"
        )

    out = {
        "embeddings": data["embeddings"].astype(np.float32),
        "image_id": data["image_id"].astype(str),
    }
    if "image_name" in data:
        out["image_name"] = data["image_name"].astype(str)

    return out


def build_imageid_to_embedding(emb_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    image_ids = emb_data["image_id"]
    emb = emb_data["embeddings"]

    out: Dict[str, np.ndarray] = {}
    for i in range(len(image_ids)):
        vec = emb[i]
        if np.isnan(vec).all():
            continue
        out[str(image_ids[i])] = vec
    return out


def build_imageid_to_imagename(emb_data: Dict[str, np.ndarray]) -> Dict[str, str]:
    if "image_name" not in emb_data:
        return {}
    ids = emb_data["image_id"]
    names = emb_data["image_name"]
    return {str(ids[i]): str(names[i]) for i in range(min(len(ids), len(names)))}


def load_merged_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if PAIR_BY_TIME_COL in df.columns:
        df[PAIR_BY_TIME_COL] = pd.to_datetime(df[PAIR_BY_TIME_COL], errors="coerce")

    for col in (IMAGE_ID_COL, WEAR_COL, TYPE_COL):
        if col not in df.columns:
            raise KeyError(f"{path} missing required column '{col}'")

    df[TYPE_COL] = df[TYPE_COL].astype(str).str.strip()
    df[IMAGE_ID_COL] = df[IMAGE_ID_COL].astype(str).str.strip()
    return df


# ----------------------------
# Pair building
# ----------------------------

@dataclass(frozen=True)
class Pair:
    E_ref: np.ndarray
    E_cur: np.ndarray
    d_wear: float
    head_idx: int


def _dense_pairs_from_df_for_type(
        df: pd.DataFrame,
        id2emb: Dict[str, np.ndarray],
        wear_type: str,
        head_idx: int,
) -> List[Pair]:
    """
    Training pairing:
      - Sorted by time within wear_type
      - Build (i,j) for all i and all j INCLUDING (i,i)
      - If MAX_INDEX_GAP_K = K, exclude near pairs: keep (i,j) only if (j==i) or |j-i| > K
    """
    sub = df[df[TYPE_COL] == wear_type].copy()
    if sub.empty:
        return []

    sub = sub[sub[IMAGE_ID_COL].isin(id2emb.keys())].copy()
    if sub.empty:
        return []

    if PAIR_BY_TIME_COL in sub.columns:
        sub = sub.sort_values(PAIR_BY_TIME_COL)
    sub = sub.drop_duplicates(subset=[IMAGE_ID_COL], keep="first").reset_index(drop=True)

    def _to_float(x) -> Optional[float]:
        try:
            return float(x)
        except Exception:
            return None

    wears: List[float] = []
    embs: List[np.ndarray] = []

    for _, r in sub.iterrows():
        w = _to_float(r[WEAR_COL])
        if w is None or pd.isna(w):
            continue
        iid = str(r[IMAGE_ID_COL])
        if iid not in id2emb:
            continue
        wears.append(float(w))
        embs.append(id2emb[iid])

    n = len(wears)
    if n < 1:
        return []

    K = MAX_INDEX_GAP_K
    if K is not None and K < 0:
        return []

    pairs: List[Pair] = []

    for i in range(n):
        Ei = embs[i]
        wi = wears[i]

        if K is None:
            for j in range(n):
                pairs.append(Pair(E_ref=Ei, E_cur=embs[j], d_wear=abs(wears[j] - wi), head_idx=head_idx))
        else:
            # keep self
            pairs.append(Pair(E_ref=Ei, E_cur=Ei, d_wear=0.0, head_idx=head_idx))

            # far-left: j <= i-K-1  <=> j < i-K
            left_stop = max(0, i - K)
            for j in range(0, left_stop):
                if j == i:
                    continue
                if abs(j - i) <= K:
                    continue
                pairs.append(Pair(E_ref=Ei, E_cur=embs[j], d_wear=abs(wears[j] - wi), head_idx=head_idx))

            # far-right: j >= i+K+1
            right_start = min(n, i + K + 1)
            for j in range(right_start, n):
                if j == i:
                    continue
                if abs(j - i) <= K:
                    continue
                pairs.append(Pair(E_ref=Ei, E_cur=embs[j], d_wear=abs(wears[j] - wi), head_idx=head_idx))

    return pairs


def _reference_pairs_from_df_for_type(
        df: pd.DataFrame,
        id2emb: Dict[str, np.ndarray],
        wear_type: str,
        head_idx: int,
) -> List[Pair]:
    """
    Testing pairing (reference mode):
      - Sorted by time within wear_type
      - Use FIRST image as reference
      - Build (ref, j) for all j INCLUDING (ref, ref)
    """
    sub = df[df[TYPE_COL] == wear_type].copy()
    if sub.empty:
        return []

    sub = sub[sub[IMAGE_ID_COL].isin(id2emb.keys())].copy()
    if sub.empty:
        return []

    if PAIR_BY_TIME_COL in sub.columns:
        sub = sub.sort_values(PAIR_BY_TIME_COL)
    sub = sub.drop_duplicates(subset=[IMAGE_ID_COL], keep="first").reset_index(drop=True)

    def _to_float(x) -> Optional[float]:
        try:
            return float(x)
        except Exception:
            return None

    wears: List[float] = []
    embs: List[np.ndarray] = []

    for _, r in sub.iterrows():
        w = _to_float(r[WEAR_COL])
        if w is None or pd.isna(w):
            continue
        iid = str(r[IMAGE_ID_COL])
        if iid not in id2emb:
            continue
        wears.append(float(w))
        embs.append(id2emb[iid])

    n = len(wears)
    if n < 1:
        return []

    Eref = embs[0]
    wref = wears[0]

    return [
        Pair(E_ref=Eref, E_cur=embs[j], d_wear=abs(wears[j] - wref), head_idx=head_idx)
        for j in range(n)
    ]


def build_pairs_for_set_multitype(
        set_id: int,
        type_to_head: Dict[str, int],
        *,
        mode: str,  # "train" or "ref"
) -> List[Pair]:
    set_dir = PROCESSED_DIR / f"set{set_id}"
    merged_path = set_dir / "merged.csv"
    emb_path = set_dir / "image_embeddings.npz"

    if not merged_path.is_file():
        raise FileNotFoundError(f"Missing {merged_path}")
    if not emb_path.is_file():
        raise FileNotFoundError(f"Missing {emb_path}")

    df = load_merged_csv(merged_path)
    emb_data = load_embeddings_npz(emb_path)
    id2emb = build_imageid_to_embedding(emb_data)

    pairs: List[Pair] = []
    rows_used = 0

    for wear_type, head_idx in type_to_head.items():
        sub = df[df[TYPE_COL] == wear_type]
        rows_used += len(sub)

        if mode == "train":
            pairs.extend(_dense_pairs_from_df_for_type(df, id2emb, wear_type, head_idx))
        elif mode == "ref":
            pairs.extend(_reference_pairs_from_df_for_type(df, id2emb, wear_type, head_idx))
        else:
            raise ValueError(f"Unknown pairing mode: {mode}")

    print(f"[Set{set_id}] rows_used={rows_used} pairs_built={len(pairs)} types={list(type_to_head.keys())}")
    return pairs


def build_pairs_over_sets(
        set_ids: List[int],
        type_to_head: Dict[str, int],
        *,
        mode: str,
) -> List[Pair]:
    all_pairs: List[Pair] = []
    for sid in set_ids:
        all_pairs.extend(build_pairs_for_set_multitype(sid, type_to_head, mode=mode))
    print(f"Total pairs: {len(all_pairs)}")
    return all_pairs


# ----------------------------
# Dataset / Dataloader
# ----------------------------

class WearPairDataset(Dataset):
    def __init__(self, pairs: List[Pair]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p = self.pairs[idx]
        return (
            torch.tensor(p.E_ref, dtype=torch.float32),
            torch.tensor(p.E_cur, dtype=torch.float32),
            torch.tensor([p.d_wear], dtype=torch.float32),
            torch.tensor(p.head_idx, dtype=torch.long),
        )


# ----------------------------
# Model: Heads only
# ----------------------------

class WearNetHead(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Linear(in_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class WearDistanceLoss(nn.Module):
    def forward(self, Z_ref: torch.Tensor, Z_cur: torch.Tensor, d_wear: torch.Tensor) -> torch.Tensor:
        d_embed = torch.norm(Z_cur - Z_ref, dim=1, keepdim=True)
        return ((d_embed - d_wear) ** 2).mean()


# ----------------------------
# Eval
# ----------------------------

@torch.no_grad()
def evaluate_heads_on_pairs(
        heads: nn.ModuleList,
        pairs: List[Pair],
        num_heads: int,
        batch_size: int = 256,
) -> Dict[str, Dict[str, float]]:
    device = next(heads[0].parameters()).device
    ds = WearPairDataset(pairs)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    sum_se = [0.0 for _ in range(num_heads)]
    sum_ae = [0.0 for _ in range(num_heads)]
    cnt    = [0   for _ in range(num_heads)]

    for E_ref, E_cur, d_wear, head_idx in dl:
        E_ref = E_ref.to(device)
        E_cur = E_cur.to(device)
        d_wear = d_wear.to(device)
        head_idx = head_idx.to(device)

        for h in range(num_heads):
            mask = (head_idx == h)
            m = int(mask.sum().item())
            if m == 0:
                continue

            Er = E_ref[mask]
            Ec = E_cur[mask]
            dw = d_wear[mask]

            Zr = heads[h](Er)
            Zc = heads[h](Ec)

            pred = torch.norm(Zc - Zr, dim=1, keepdim=True)
            err = pred - dw

            sum_se[h] += float((err ** 2).sum().item())
            sum_ae[h] += float(err.abs().sum().item())
            cnt[h] += m

    out: Dict[str, Dict[str, float]] = {}
    for h in range(num_heads):
        if cnt[h] == 0:
            out[f"H{h}"] = {"mse": float("nan"), "mae": float("nan"), "n": 0}
        else:
            out[f"H{h}"] = {
                "mse": sum_se[h] / cnt[h],
                "mae": sum_ae[h] / cnt[h],
                "n": float(cnt[h]),
            }
    return out


def load_heads_from_ckpt(ckpt_path: Path) -> Tuple[nn.ModuleList, Dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)

    num_heads = int(ckpt["num_heads"])
    in_dim = int(ckpt["in_dim"])
    embed_dim = int(ckpt["embed_dim"])

    heads = nn.ModuleList([WearNetHead(in_dim=in_dim, embed_dim=embed_dim).to(device) for _ in range(num_heads)])
    for h in range(num_heads):
        heads[h].load_state_dict(ckpt["state_dicts"][h])
    return heads, ckpt


# ----------------------------
# Export head embeddings to CSV (after training)
# ----------------------------

@torch.no_grad()
def export_head_embeddings_csv(
        *,
        ckpt_path: Path,
        exp_name: str,
        set_ids: List[int],
        type_to_head: Dict[str, int],
) -> None:
    heads, ckpt = load_heads_from_ckpt(ckpt_path)
    num_heads = int(ckpt["num_heads"])
    embed_dim = int(ckpt["embed_dim"])
    device = next(heads[0].parameters()).device

    for sid in set_ids:
        set_dir = PROCESSED_DIR / f"set{sid}"
        merged_path = set_dir / "merged.csv"
        emb_path = set_dir / "image_embeddings.npz"

        if not merged_path.is_file() or not emb_path.is_file():
            print(f"[Export] Set{sid}: missing merged.csv or image_embeddings.npz, skipping")
            continue

        df = load_merged_csv(merged_path)
        emb_data = load_embeddings_npz(emb_path)
        id2emb = build_imageid_to_embedding(emb_data)
        id2name = build_imageid_to_imagename(emb_data)

        df = df[df[TYPE_COL].isin(type_to_head.keys())].copy()
        if df.empty:
            print(f"[Export] Set{sid}: no rows for requested types, skipping")
            continue

        df = df[df[IMAGE_ID_COL].isin(id2emb.keys())].copy()

        if PAIR_BY_TIME_COL in df.columns:
            df = df.sort_values(PAIR_BY_TIME_COL)
        df = df.drop_duplicates(subset=[IMAGE_ID_COL], keep="first").reset_index(drop=True)

        if df.empty:
            print(f"[Export] Set{sid}: no rows with embeddings, skipping")
            continue

        df["head_idx"] = df[TYPE_COL].map(type_to_head).astype(int)

        rows_out = []
        for h in range(num_heads):
            sub = df[df["head_idx"] == h]
            if sub.empty:
                continue

            X = np.stack([id2emb[str(iid)] for iid in sub[IMAGE_ID_COL].astype(str).tolist()], axis=0).astype(np.float32)
            Xt = torch.tensor(X, dtype=torch.float32, device=device)

            bs = 512
            Z_all = []
            for i in range(0, Xt.shape[0], bs):
                Z_all.append(heads[h](Xt[i:i+bs]).detach().cpu().numpy())
            Z = np.concatenate(Z_all, axis=0)

            sub2 = sub.reset_index(drop=True)
            for k in range(len(sub2)):
                r = sub2.iloc[k]
                row = {
                    "set_id": sid,
                    "image_id": str(r[IMAGE_ID_COL]),
                    "image_name": id2name.get(str(r[IMAGE_ID_COL]), ""),
                    "type": str(r[TYPE_COL]),
                    "head_idx": int(h),
                }
                for d in range(embed_dim):
                    row[f"z{d}"] = float(Z[k, d])
                rows_out.append(row)

        if not rows_out:
            print(f"[Export] Set{sid}: nothing exported")
            continue

        out_df = pd.DataFrame(rows_out)
        out_path = set_dir / f"head_embeddings_{exp_name}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"[Export] Saved: {out_path}")


# ----------------------------
# Training
# ----------------------------

def train_multhead(
        *,
        exp_name: str,
        pairs: List[Pair],
        num_heads: int,
        in_dim: int,
        embed_dim: int,
        epochs: int,
        batch_size: int,
        lr: float,
        save_dir: Path,
) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n==============================")
    print(f"Experiment: {exp_name}")
    print("Device:", device)
    print("Heads:", num_heads)
    print(f"LR: {lr} | head_dim: {embed_dim}")
    print("==============================")

    if len(pairs) == 0:
        raise RuntimeError(f"[{exp_name}] No pairs built. Check types present, merged.csv, and embeddings.npz.")

    ds = WearPairDataset(pairs)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    heads = nn.ModuleList([WearNetHead(in_dim=in_dim, embed_dim=embed_dim).to(device) for _ in range(num_heads)])
    opts = [torch.optim.Adam(head.parameters(), lr=lr) for head in heads]
    criterion = WearDistanceLoss()

    for epoch in range(1, epochs + 1):
        heads.train()
        total_loss = [0.0 for _ in range(num_heads)]
        total_batches = [0 for _ in range(num_heads)]

        for E_ref, E_cur, d_wear, head_idx in dl:
            E_ref = E_ref.to(device)
            E_cur = E_cur.to(device)
            d_wear = d_wear.to(device)
            head_idx = head_idx.to(device)

            for h in range(num_heads):
                mask = (head_idx == h)
                if mask.sum().item() == 0:
                    continue

                Er = E_ref[mask]
                Ec = E_cur[mask]
                dw = d_wear[mask]

                Zr = heads[h](Er)
                Zc = heads[h](Ec)

                loss = criterion(Zr, Zc, dw)

                opts[h].zero_grad()
                loss.backward()
                opts[h].step()

                total_loss[h] += float(loss.item())
                total_batches[h] += 1

        parts = []
        for h in range(num_heads):
            if total_batches[h] == 0:
                parts.append(f"H{h}: n/a")
            else:
                parts.append(f"H{h}: {total_loss[h]/total_batches[h]:.4f}")
        print(f"Epoch {epoch:03d} | " + " | ".join(parts))

    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "exp_name": exp_name,
        "in_dim": in_dim,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "state_dicts": [h.state_dict() for h in heads],
    }
    out_path = save_dir / f"{exp_name}.pkl"
    torch.save(ckpt, out_path)
    print("Saved model to:", out_path)
    return out_path


def _fmt_lr(lr: float) -> str:
    # safe filename-friendly lr string
    s = f"{lr:.0e}" if lr < 1e-2 else f"{lr:g}"
    return s.replace("+", "").replace(".", "p")


def _print_metrics(tag: str, metrics: Dict[str, Dict[str, float]]) -> None:
    parts = []
    for hk, m in metrics.items():
        parts.append(f"{hk}: mse={m['mse']:.6f} mae={m['mae']:.6f} n={int(m['n'])}")
    print(f"[{tag}] " + " | ".join(parts))


# ----------------------------
# Main: build pairs once, then sweep
# ----------------------------

def main():
    # Reproducibility (optional)
    torch.manual_seed(0)
    np.random.seed(0)

    type_to_head_A = {TYPE_FW: 0, TYPE_AD: 1}
    type_to_head_B = {TYPE_FWAD: 0}

    # Build pairs ONCE (pairing does not depend on lr/head_dim)
    print("\n=== Building pairs for Model A ===")
    pairs_A_train = build_pairs_over_sets(TRAIN_SETS, type_to_head_A, mode="train")
    pairs_A_val   = build_pairs_over_sets(VAL_SETS,   type_to_head_A, mode="ref")
    pairs_A_test  = build_pairs_over_sets(TEST_SETS,  type_to_head_A, mode="ref")

    print("\n=== Building pairs for Model B ===")
    pairs_B_train = build_pairs_over_sets(FWAD_TRAIN_SETS, type_to_head_B, mode="train")
    pairs_B_val   = build_pairs_over_sets(FWAD_VAL_SETS,   type_to_head_B, mode="ref")
    pairs_B_test  = build_pairs_over_sets(FWAD_TEST_SETS,  type_to_head_B, mode="ref")

    sweep_rows = []

    for head_dim in HEAD_DIM_GRID:
        for lr in LR_GRID:
            lr_tag = _fmt_lr(lr)
            expA = f"wear_heads_FW_AD_dim{head_dim}_lr{lr_tag}"
            expB = f"wear_head_FWAD_dim{head_dim}_lr{lr_tag}"

            # ---------
            # Train A
            # ---------
            ckpt_A = train_multhead(
                exp_name=expA,
                pairs=pairs_A_train,
                num_heads=2,
                in_dim=EMBED_IN_DIM,
                embed_dim=head_dim,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                lr=lr,
                save_dir=MODELS_DIR,
            )
            heads_A, _ = load_heads_from_ckpt(ckpt_A)

            mA_val  = evaluate_heads_on_pairs(heads_A, pairs_A_val,  num_heads=2)
            mA_test = evaluate_heads_on_pairs(heads_A, pairs_A_test, num_heads=2)

            _print_metrics(f"ModelA VAL  ({expA})", mA_val)
            _print_metrics(f"ModelA TEST ({expA})", mA_test)

            if EXPORT_EMBEDDINGS_EACH_RUN:
                export_head_embeddings_csv(
                    ckpt_path=ckpt_A,
                    exp_name=expA,
                    set_ids=(TRAIN_SETS + VAL_SETS + TEST_SETS),
                    type_to_head=type_to_head_A,
                )

            # ---------
            # Train B
            # ---------
            ckpt_B = train_multhead(
                exp_name=expB,
                pairs=pairs_B_train,
                num_heads=1,
                in_dim=EMBED_IN_DIM,
                embed_dim=head_dim,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                lr=lr,
                save_dir=MODELS_DIR,
            )
            heads_B, _ = load_heads_from_ckpt(ckpt_B)

            mB_val  = evaluate_heads_on_pairs(heads_B, pairs_B_val,  num_heads=1)
            mB_test = evaluate_heads_on_pairs(heads_B, pairs_B_test, num_heads=1)

            _print_metrics(f"ModelB VAL  ({expB})", mB_val)
            _print_metrics(f"ModelB TEST ({expB})", mB_test)

            if EXPORT_EMBEDDINGS_EACH_RUN:
                export_head_embeddings_csv(
                    ckpt_path=ckpt_B,
                    exp_name=expB,
                    set_ids=(FWAD_TRAIN_SETS + FWAD_VAL_SETS + FWAD_TEST_SETS),
                    type_to_head=type_to_head_B,
                )

            # Collect summary row (TEST MAE only, as you requested)
            row = {
                "lr": lr,
                "head_dim": head_dim,
                "ModelA_Test_MAE_H0": float(mA_test["H0"]["mae"]),
                "ModelA_Test_MAE_H1": float(mA_test["H1"]["mae"]),
                "ModelB_Test_MAE_H0": float(mB_test["H0"]["mae"]),
                "ModelA_Test_N_H0": int(mA_test["H0"]["n"]),
                "ModelA_Test_N_H1": int(mA_test["H1"]["n"]),
                "ModelB_Test_N_H0": int(mB_test["H0"]["n"]),
                "expA": expA,
                "expB": expB,
            }
            sweep_rows.append(row)

    # Final summary table
    summary_df = pd.DataFrame(sweep_rows).sort_values(["head_dim", "lr"]).reset_index(drop=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = MODELS_DIR / "hparam_sweep_summary.csv"
    summary_df.to_csv(out_csv, index=False)

    print("\n==============================")
    print("SUMMARY (TEST MAE)")
    print("==============================")
    # Short, readable print
    for _, r in summary_df.iterrows():
        print(
            f"head_dim={int(r['head_dim']):3d} | lr={r['lr']:.6g} | "
            f"A: H0_MAE={r['ModelA_Test_MAE_H0']:.6f}  H1_MAE={r['ModelA_Test_MAE_H1']:.6f} | "
            f"B: H0_MAE={r['ModelB_Test_MAE_H0']:.6f}"
        )
    print(f"\nSaved summary CSV to: {out_csv}")


if __name__ == "__main__":
    main()