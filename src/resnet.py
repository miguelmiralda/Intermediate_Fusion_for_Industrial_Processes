# ============================================================
#  MULTI-HEAD TOOL WEAR TRAINING (USING PRECOMPUTED EMBEDDINGS)
#  Model A: 2-head (flank_wear, adhesion) on TRAIN_SETS
#  Model B: 1-head (flank_wear+adhesion) on FWAD_SETS (separate experiment)
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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Config
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# MATWI paper split
TRAIN_SETS = [1, 2, 5, 7, 8, 10, 11]

# Sets containing flank_wear+adhesion
FWAD_TRAIN_SETS = [3, 13, 16]
FWAD_VAL_SETS   = [17]
FWAD_TEST_SETS  = [12]

# Pairing logic: pair consecutive samples sorted by time *within the same wear type*
PAIR_BY_TIME_COL = "anchor_time"

# Columns in merged.csv
WEAR_COL = "wear"
TYPE_COL = "type"
IMAGE_ID_COL = "image_id"

# Training hyperparams
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
EMBED_IN_DIM = 2048
HEAD_EMBED_DIM = 128

# Wear types (must match values in merged.csv "type" column)
TYPE_FW = "flank_wear"
TYPE_AD = "adhesion"
TYPE_FWAD = "flank_wear+adhesion"


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

    emb = data["embeddings"].astype(np.float32)  # (N, 2048)
    image_id = data["image_id"].astype(str)      # (N,)
    return {"embeddings": emb, "image_id": image_id}


def build_imageid_to_embedding(emb_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    image_ids = emb_data["image_id"]
    emb = emb_data["embeddings"]

    out: Dict[str, np.ndarray] = {}
    for i in range(len(image_ids)):
        vec = emb[i]
        # Skip missing/corrupt rows saved as all-NaN
        if np.isnan(vec).all():
            continue
        out[str(image_ids[i])] = vec
    return out


def load_merged_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Make anchor_time parseable
    if PAIR_BY_TIME_COL in df.columns:
        df[PAIR_BY_TIME_COL] = pd.to_datetime(df[PAIR_BY_TIME_COL], errors="coerce")

    # Ensure required columns exist
    for col in (IMAGE_ID_COL, WEAR_COL, TYPE_COL):
        if col not in df.columns:
            raise KeyError(f"{path} missing required column '{col}'")

    # Normalize types to string
    df[TYPE_COL] = df[TYPE_COL].astype(str).str.strip()
    df[IMAGE_ID_COL] = df[IMAGE_ID_COL].astype(str).str.strip()

    return df


# ----------------------------
# Pair building
# ----------------------------

@dataclass(frozen=True)
class Pair:
    E_ref: np.ndarray   # (2048,)
    E_cur: np.ndarray   # (2048,)
    d_wear: float       # scalar >= 0
    head_idx: int       # which head to train (0 or 1), or 0 for single-head experiments


def _pairs_from_df_for_type(
        df: pd.DataFrame,
        id2emb: Dict[str, np.ndarray],
        wear_type: str,
        head_idx: int,
) -> List[Pair]:
    """
    Build consecutive (ref->cur) pairs *only within the same wear_type*.
    This is important because your 'wear' column mixes types; you must not
    subtract wear values from different types.
    """
    sub = df[df[TYPE_COL] == wear_type].copy()
    if sub.empty:
        return []

    # Keep only rows with valid embeddings
    sub = sub[sub[IMAGE_ID_COL].isin(id2emb.keys())].copy()
    if sub.empty:
        return []

    # Sort by time for "consecutive" pairing
    if PAIR_BY_TIME_COL in sub.columns:
        sub = sub.sort_values(PAIR_BY_TIME_COL)
    sub = sub.reset_index(drop=True)

    pairs: List[Pair] = []
    for i in range(1, len(sub)):
        r0 = sub.iloc[i - 1]
        r1 = sub.iloc[i]

        w0 = r0[WEAR_COL]
        w1 = r1[WEAR_COL]
        if pd.isna(w0) or pd.isna(w1):
            continue

        # IMPORTANT: wear values are numeric, but can be stringy; force float
        try:
            w0f = float(w0)
            w1f = float(w1)
        except Exception:
            continue

        E0 = id2emb[str(r0[IMAGE_ID_COL])]
        E1 = id2emb[str(r1[IMAGE_ID_COL])]

        pairs.append(Pair(E_ref=E0, E_cur=E1, d_wear=abs(w1f - w0f), head_idx=head_idx))

    return pairs


def build_pairs_for_set_multitype(
        set_id: int,
        type_to_head: Dict[str, int],
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
        pairs.extend(_pairs_from_df_for_type(df, id2emb, wear_type, head_idx))

    print(f"[Set{set_id}] rows_used={rows_used} pairs_built={len(pairs)} types={list(type_to_head.keys())}")
    return pairs


def build_pairs_over_sets(
        set_ids: List[int],
        type_to_head: Dict[str, int],
) -> List[Pair]:
    all_pairs: List[Pair] = []
    for sid in set_ids:
        all_pairs.extend(build_pairs_for_set_multitype(sid, type_to_head))
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
            torch.tensor([p.d_wear], dtype=torch.float32),   # shape (1,)
            torch.tensor(p.head_idx, dtype=torch.long),
        )


# ----------------------------
# Model: Heads only (uses precomputed 2048-D embeddings)
# ----------------------------

class WearNetHead(nn.Module):
    def __init__(self, in_dim: int = EMBED_IN_DIM, embed_dim: int = HEAD_EMBED_DIM):
        super().__init__()
        self.embedding = nn.Linear(in_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class WearDistanceLoss(nn.Module):
    """
    For a given head:
        ||Z_cur - Z_ref||  ≈  d_wear
    """
    def forward(self, Z_ref: torch.Tensor, Z_cur: torch.Tensor, d_wear: torch.Tensor) -> torch.Tensor:
        # Z_ref, Z_cur: (B, D)
        # d_wear:      (B, 1)
        d_embed = torch.norm(Z_cur - Z_ref, dim=1, keepdim=True)  # (B,1)
        return ((d_embed - d_wear) ** 2).mean()


# ----------------------------
# Training
# ----------------------------

def train_multhead(
        *,
        exp_name: str,
        pairs: List[Pair],
        num_heads: int,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr: float = LR,
        save_dir: Path = MODELS_DIR,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n==============================")
    print(f"Experiment: {exp_name}")
    print("Device:", device)
    print("Heads:", num_heads)
    print("==============================")

    if len(pairs) == 0:
        raise RuntimeError(f"[{exp_name}] No pairs built. Check types present, merged.csv, and embeddings.npz.")

    ds = WearPairDataset(pairs)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    heads = nn.ModuleList([WearNetHead().to(device) for _ in range(num_heads)])
    opts = [torch.optim.Adam(head.parameters(), lr=lr) for head in heads]
    criterion = WearDistanceLoss()

    for epoch in range(1, epochs + 1):
        heads.train()
        total_loss = [0.0 for _ in range(num_heads)]
        total_batches = [0 for _ in range(num_heads)]

        for E_ref, E_cur, d_wear, head_idx in dl:
            E_ref = E_ref.to(device)
            E_cur = E_cur.to(device)
            d_wear = d_wear.to(device)      # (B,1)
            head_idx = head_idx.to(device)  # (B,)

            # For each head, only train on samples belonging to that head
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

                total_loss[h] += loss.item()
                total_batches[h] += 1

        # Print losses
        parts = []
        for h in range(num_heads):
            if total_batches[h] == 0:
                parts.append(f"H{h}: n/a")
            else:
                parts.append(f"H{h}: {total_loss[h]/total_batches[h]:.4f}")
        print(f"Epoch {epoch:03d} | " + " | ".join(parts))

    # Save (torch.save to .pkl is totally fine; it’s a pickle under the hood)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "exp_name": exp_name,
        "in_dim": EMBED_IN_DIM,
        "embed_dim": HEAD_EMBED_DIM,
        "num_heads": num_heads,
        "state_dicts": [h.state_dict() for h in heads],
    }
    out_path = save_dir / f"{exp_name}.pkl"
    torch.save(ckpt, out_path)
    print("Saved model to:", out_path)


# ----------------------------
# Main: run both experiments
# ----------------------------

def main():
    # -------------------------
    # Model A: 2-head (FW + AD) on TRAIN_SETS
    # -------------------------
    type_to_head_A = {
        TYPE_FW: 0,
        TYPE_AD: 1,
    }
    pairs_A = build_pairs_over_sets(TRAIN_SETS, type_to_head_A)
    train_multhead(
        exp_name="wear_heads_FW_AD",
        pairs=pairs_A,
        num_heads=2,
    )

    # -------------------------
    # Model B: 1-head (FW+AD) on FWAD_SETS (separate experiment)
    # -------------------------
    type_to_head_B = {
        TYPE_FWAD: 0,
    }
    pairs_B_train = build_pairs_over_sets(FWAD_TRAIN_SETS, type_to_head_B)
    pairs_B_val   = build_pairs_over_sets(FWAD_VAL_SETS, type_to_head_B)
    pairs_B_test  = build_pairs_over_sets(FWAD_TEST_SETS, type_to_head_B)

    train_multhead(exp_name="wear_head_FWAD", pairs=pairs_B_train, num_heads=1)


if __name__ == "__main__":
    main()