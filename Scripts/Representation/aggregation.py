from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


DEFAULT_POSITION_ORDER: Tuple[str, ...] = ("A", "E", "M", "P", "T")
REQUIRED_WINDOW_META_COLUMNS = ("patient_id", "position", "window_id", "window_idx")
REQUIRED_POSITION_META_COLUMNS = ("patient_id", "position")


def aggregate_windows_to_position(window_embeddings: np.ndarray) -> np.ndarray:
    """Average-pool all window embeddings from one (patient_id, position).

    Parameters
    ----------
    window_embeddings:
        Array with shape [n_windows, D].

    Returns
    -------
    np.ndarray
        Position-level embedding with shape [D].
    """
    x = np.asarray(window_embeddings, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"window_embeddings must have shape [n_windows, D], but got {x.shape}")
    if x.shape[0] == 0:
        raise ValueError("window_embeddings must contain at least one window")
    return np.asarray(x.mean(axis=0), dtype=np.float32)


def build_position_embeddings(
    window_meta_df: pd.DataFrame,
    window_embeddings: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Aggregate window-level embeddings into position-level embeddings.

    Parameters
    ----------
    window_meta_df:
        One row per window. Must contain at least:
        patient_id, position, window_id, window_idx.
    window_embeddings:
        Window-level embeddings with shape [N, D].

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        position_meta_df: one row per (patient_id, position)
        position_embeddings: [M, D]
    """
    _validate_required_columns(window_meta_df, REQUIRED_WINDOW_META_COLUMNS, "window_meta_df")

    x = np.asarray(window_embeddings, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"window_embeddings must have shape [N, D], but got {x.shape}")
    if len(window_meta_df) != x.shape[0]:
        raise ValueError(
            f"window_meta_df and window_embeddings size mismatch: {len(window_meta_df)} vs {x.shape[0]}"
        )

    df = window_meta_df.copy().reset_index(drop=True)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["position"] = df["position"].astype(str).str.upper().str.strip()
    df["_row_idx"] = np.arange(len(df), dtype=np.int64)

    group_cols = ["patient_id", "position"]
    pos_meta_rows = []
    pos_embs = []

    for (patient_id, position), group in df.groupby(group_cols, sort=True):
        row_indices = group["_row_idx"].to_numpy(dtype=np.int64)
        emb = aggregate_windows_to_position(x[row_indices])
        pos_embs.append(emb)
        pos_meta_rows.append(
            {
                "patient_id": patient_id,
                "position": position,
                "n_windows": int(len(group)),
            }
        )

    position_meta_df = pd.DataFrame(pos_meta_rows)
    position_embeddings = np.stack(pos_embs, axis=0).astype(np.float32, copy=False)
    return position_meta_df, position_embeddings


def aggregate_positions_to_patient(
    position_meta_df: pd.DataFrame,
    position_embeddings: np.ndarray,
    position_order: Tuple[str, ...] = DEFAULT_POSITION_ORDER,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Concatenate position embeddings into one patient embedding.

    Protocol fixed here
    -------------------
    1) Each available position already has one embedding [D].
    2) Positions are concatenated in fixed order A/E/M/P/T.
    3) If a position is missing, use a zero vector [D].

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        patient_meta_df: one row per patient, with has_A ... has_T columns
        patient_embeddings: [P, 5*D]
    """
    _validate_required_columns(position_meta_df, REQUIRED_POSITION_META_COLUMNS, "position_meta_df")

    x = np.asarray(position_embeddings, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"position_embeddings must have shape [M, D], but got {x.shape}")
    if len(position_meta_df) != x.shape[0]:
        raise ValueError(
            f"position_meta_df and position_embeddings size mismatch: {len(position_meta_df)} vs {x.shape[0]}"
        )

    df = position_meta_df.copy().reset_index(drop=True)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["position"] = df["position"].astype(str).str.upper().str.strip()
    df["_row_idx"] = np.arange(len(df), dtype=np.int64)

    emb_dim = int(x.shape[1])
    zero_vec = np.zeros((emb_dim,), dtype=np.float32)

    patient_meta_rows = []
    patient_embs = []

    for patient_id, group in df.groupby("patient_id", sort=True):
        pos_to_idx = dict(zip(group["position"], group["_row_idx"]))
        parts = []
        meta = {"patient_id": patient_id}

        for pos in position_order:
            has_pos = pos in pos_to_idx
            meta[f"has_{pos}"] = bool(has_pos)
            if has_pos:
                parts.append(x[int(pos_to_idx[pos])])
            else:
                parts.append(zero_vec)

        patient_meta_rows.append(meta)
        patient_embs.append(np.concatenate(parts, axis=0))

    patient_meta_df = pd.DataFrame(patient_meta_rows)
    patient_embeddings = np.stack(patient_embs, axis=0).astype(np.float32, copy=False)
    return patient_meta_df, patient_embeddings


def _validate_required_columns(df: pd.DataFrame, required: Tuple[str, ...], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}. Available columns: {list(df.columns)}")
