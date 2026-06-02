#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract external-validation patient-level BEATs embeddings.

This script is intentionally BEATs-only for external validation. It follows the
same representation-learning protocol used by Scripts/Representation/extract_embeddings.py:

    prepared window_library
        -> WindowDataset
        -> frozen BEATsAdapter
        -> window-level embeddings
        -> window-to-position mean pooling
        -> fixed A/E/M/P/T position concatenation into patient-level embeddings

No route selection, no model comparison, and no window-parameter tuning are exposed here.
The external validation should use the fixed BEATs route and the fixed 4 s / 1 s
window library prepared in Outputs/validation/preprocessing/Data_windows.

Python usage
------------
from Scripts.Validation.representation.extract_external_beats_embeddings import main
main([])

Command line
------------
python Scripts/Validation/representation/extract_external_beats_embeddings.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


POSITION_ORDER: Tuple[str, ...] = ("A", "E", "M", "P", "T")


# ---------------------------------------------------------------------
# Import existing repository implementation
# ---------------------------------------------------------------------
def _add_representation_dir_to_path() -> Path:
    """Make Scripts/Representation importable when this file is run directly.

    Expected layout:
        Scripts/
          Representation/
            aggregation.py
            beats_adapter.py
            window_dataset.py
          Validation/
            representation/
              extract_external_beats_embeddings.py
    """
    this_file = Path(__file__).resolve()
    scripts_dir = this_file.parents[2]
    representation_dir = scripts_dir / "Representation"
    if not representation_dir.exists():
        raise FileNotFoundError(f"Cannot find Scripts/Representation directory: {representation_dir}")
    representation_dir_str = str(representation_dir)
    if representation_dir_str not in sys.path:
        sys.path.insert(0, representation_dir_str)
    return representation_dir


_add_representation_dir_to_path()

from Scripts.Representation.aggregation import aggregate_positions_to_patient, build_position_embeddings  # noqa: E402
from Scripts.Representation.beats_adapter import BEATsAdapter  # noqa: E402
from Scripts.Representation.window_dataset import WindowDataset, load_patient_ids  # noqa: E402


# ---------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------
def print_banner(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def log_info(msg: str) -> None:
    print(f"[info] {msg}")


def log_warn(msg: str) -> None:
    print(f"[warn] {msg}")


def log_done(msg: str) -> None:
    print(f"[done] {msg}")


# ---------------------------------------------------------------------
# BEATs extraction helpers
# ---------------------------------------------------------------------
def _collate_waveform_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep waveforms as a list and metadata as parallel Python lists.

    This mirrors the collate logic in Scripts/Representation/extract_embeddings.py.
    BEATsAdapter handles resampling and padding internally.
    """
    return {
        "patient_id": [item["patient_id"] for item in batch],
        "position": [item["position"] for item in batch],
        "window_id": [item["window_id"] for item in batch],
        "window_idx": [item["window_idx"] for item in batch],
        "window_library_path": [item["window_library_path"] for item in batch],
        "waveforms": [item["waveform"] for item in batch],
        "src_sr": int(batch[0]["src_sr"]),
    }


def run_beats_window_embedding_extraction(
    dataset: WindowDataset,
    adapter: BEATsAdapter,
    batch_size: int,
    num_workers: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract one frozen BEATs embedding vector for every window in the dataset."""
    if len(dataset) == 0:
        raise ValueError("dataset is empty")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_waveform_batch,
    )

    meta_rows: List[Dict[str, Any]] = []
    emb_batches: List[np.ndarray] = []

    progress = tqdm(loader, desc="Extracting BEATs window embeddings", leave=False)
    for batch in progress:
        emb = adapter.extract_embeddings(waveforms=batch["waveforms"], src_sr=int(batch["src_sr"]))
        emb = np.asarray(emb, dtype=np.float32)

        if emb.ndim != 2:
            raise ValueError(f"BEATsAdapter.extract_embeddings() must return [B, D], but got {emb.shape}")
        if emb.shape[0] != len(batch["waveforms"]):
            raise ValueError(f"Embedding batch size mismatch: got {emb.shape[0]} embeddings for {len(batch['waveforms'])} inputs")

        for i in range(emb.shape[0]):
            meta_rows.append(
                {
                    "patient_id": str(batch["patient_id"][i]),
                    "position": str(batch["position"][i]),
                    "window_id": str(batch["window_id"][i]),
                    "window_idx": int(batch["window_idx"][i]),
                    "window_library_path": str(batch["window_library_path"][i]),
                }
            )
        emb_batches.append(emb)

    if not meta_rows or not emb_batches:
        raise ValueError("No embeddings were extracted")

    window_meta_df = pd.DataFrame(meta_rows)
    window_embeddings = np.concatenate(emb_batches, axis=0).astype(np.float32, copy=False)

    if len(window_meta_df) != window_embeddings.shape[0]:
        raise ValueError(f"window_meta_df and window_embeddings size mismatch: {len(window_meta_df)} vs {window_embeddings.shape[0]}")

    return window_meta_df, window_embeddings


def save_embedding_artifacts(
    out_dir: Path,
    window_meta_df: pd.DataFrame,
    window_embeddings: np.ndarray,
    position_meta_df: pd.DataFrame,
    position_embeddings: np.ndarray,
    patient_meta_df: pd.DataFrame,
    patient_embeddings: np.ndarray,
) -> None:
    """Save window/position/patient metadata and embeddings using the existing layout."""
    out_dir.mkdir(parents=True, exist_ok=True)

    window_meta_df.to_csv(out_dir / "window_meta.csv", index=False, encoding="utf-8-sig")
    np.save(out_dir / "window_embeddings.npy", np.asarray(window_embeddings, dtype=np.float32))

    position_meta_df.to_csv(out_dir / "position_meta.csv", index=False, encoding="utf-8-sig")
    np.save(out_dir / "position_embeddings.npy", np.asarray(position_embeddings, dtype=np.float32))

    patient_meta_df.to_csv(out_dir / "patient_meta.csv", index=False, encoding="utf-8-sig")
    np.save(out_dir / "patient_embeddings.npy", np.asarray(patient_embeddings, dtype=np.float32))


def write_representation_qc_summary(
    out_dir: Path,
    patient_ids: List[str],
    window_meta_df: pd.DataFrame,
    window_embeddings: np.ndarray,
    position_meta_df: pd.DataFrame,
    position_embeddings: np.ndarray,
    patient_meta_df: pd.DataFrame,
    patient_embeddings: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Write a compact QC summary for the external BEATs representation step."""
    out_dir.mkdir(parents=True, exist_ok=True)

    n_complete_patients = 0
    if all(f"has_{pos}" in patient_meta_df.columns for pos in POSITION_ORDER):
        has_cols = [f"has_{pos}" for pos in POSITION_ORDER]
        n_complete_patients = int(patient_meta_df[has_cols].all(axis=1).sum())

    rows = [
        {"item": "route", "value": "beats"},
        {"item": "n_patient_ids_requested", "value": int(len(patient_ids))},
        {"item": "n_windows_embedded", "value": int(window_embeddings.shape[0])},
        {"item": "window_embedding_dim", "value": int(window_embeddings.shape[1])},
        {"item": "n_position_embeddings", "value": int(position_embeddings.shape[0])},
        {"item": "position_embedding_dim", "value": int(position_embeddings.shape[1])},
        {"item": "n_patient_embeddings", "value": int(patient_embeddings.shape[0])},
        {"item": "patient_embedding_dim", "value": int(patient_embeddings.shape[1])},
        {"item": "n_patients_with_all_AEMPT_positions", "value": int(n_complete_patients)},
        {"item": "window_lib_path", "value": str(args.window_lib_path)},
        {"item": "patient_list", "value": str(args.patient_list)},
        {"item": "checkpoint_path", "value": str(args.checkpoint_path)},
        {"item": "device", "value": str(args.device)},
        {"item": "batch_size", "value": int(args.batch_size)},
        {"item": "num_workers", "value": int(args.num_workers)},
        {"item": "out_dir", "value": str(args.out_dir)},
    ]

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "representation_qc_summary.csv", index=False, encoding="utf-8-sig")

    windows_per_patient = (
        window_meta_df.groupby("patient_id", sort=True)
        .size()
        .reset_index(name="n_windows")
    )
    windows_per_patient.to_csv(out_dir / "windows_per_patient.csv", index=False, encoding="utf-8-sig")

    config = {
        "route": "beats",
        "window_lib_path": str(args.window_lib_path),
        "patient_list": str(args.patient_list),
        "checkpoint_path": str(args.checkpoint_path),
        "device": str(args.device),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "out_dir": str(args.out_dir),
        "position_order": list(POSITION_ORDER),
    }
    with (out_dir / "external_beats_embedding_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return summary_df


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen BEATs embeddings for the external-validation window library."
    )
    parser.add_argument("--window-lib-path", type=Path, default=Path("Outputs/validation/preprocessing/Data_windows/windows_4_5_4.0_1.0"))
    parser.add_argument("--patient-list", type=Path, default=Path("Outputs/validation/preprocessing/Data_screened/passed_patients.csv"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("Representation_learning/checkpoints/BEATs_iter3_plus_AS2M.pt"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=Path, default=Path("Outputs/validation/representation/beats_4s_1s"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print_banner("External validation: frozen BEATs representation extraction")
    log_info("route             : beats")
    log_info(f"window_lib_path   : {args.window_lib_path}")
    log_info(f"patient_list      : {args.patient_list}")
    log_info(f"checkpoint_path   : {args.checkpoint_path}")
    log_info(f"device            : {args.device}")
    log_info(f"batch_size        : {args.batch_size}")
    log_info(f"num_workers       : {args.num_workers}")
    log_info(f"out_dir           : {args.out_dir}")

    window_index_csv = args.window_lib_path / "window_index.csv"
    recording_manifest_csv = args.window_lib_path / "recording_manifest.csv"

    if not window_index_csv.exists():
        raise FileNotFoundError(f"window_index.csv not found: {window_index_csv}")
    if not recording_manifest_csv.exists():
        raise FileNotFoundError(f"recording_manifest.csv not found: {recording_manifest_csv}")
    if not args.patient_list.exists():
        raise FileNotFoundError(f"patient list not found: {args.patient_list}")
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"BEATs checkpoint not found: {args.checkpoint_path}")

    patient_ids = load_patient_ids(str(args.patient_list))
    log_done(f"Loaded external patient IDs | n={len(patient_ids)}")

    dataset = WindowDataset(
        window_index_csv=str(window_index_csv),
        recording_manifest_csv=str(recording_manifest_csv),
        patient_ids=patient_ids,
        positions=POSITION_ORDER,
    )
    log_done(f"Constructed external WindowDataset | n_windows={len(dataset)}")

    adapter = BEATsAdapter(checkpoint_path=str(args.checkpoint_path), device=str(args.device), trainable=False)
    log_done("Built frozen BEATs adapter")

    window_meta_df, window_embeddings = run_beats_window_embedding_extraction(
        dataset=dataset,
        adapter=adapter,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    log_done(f"Window embeddings extracted | shape={tuple(window_embeddings.shape)}")

    print_banner("Aggregating window -> position -> patient")
    position_meta_df, position_embeddings = build_position_embeddings(
        window_meta_df=window_meta_df,
        window_embeddings=window_embeddings,
    )
    patient_meta_df, patient_embeddings = aggregate_positions_to_patient(
        position_meta_df=position_meta_df,
        position_embeddings=position_embeddings,
        position_order=POSITION_ORDER,
    )

    log_done(f"Position embeddings built | shape={tuple(position_embeddings.shape)}")
    log_done(f"Patient embeddings built | shape={tuple(patient_embeddings.shape)}")

    save_embedding_artifacts(
        out_dir=args.out_dir,
        window_meta_df=window_meta_df,
        window_embeddings=window_embeddings,
        position_meta_df=position_meta_df,
        position_embeddings=position_embeddings,
        patient_meta_df=patient_meta_df,
        patient_embeddings=patient_embeddings,
    )

    summary_df = write_representation_qc_summary(
        out_dir=args.out_dir,
        patient_ids=patient_ids,
        window_meta_df=window_meta_df,
        window_embeddings=window_embeddings,
        position_meta_df=position_meta_df,
        position_embeddings=position_embeddings,
        patient_meta_df=patient_meta_df,
        patient_embeddings=patient_embeddings,
        args=args,
    )

    print_banner("Done")
    print(f"Saved external BEATs embedding artifacts to: {args.out_dir}")
    print(f"window_embeddings shape  : {tuple(window_embeddings.shape)}")
    print(f"position_embeddings shape: {tuple(position_embeddings.shape)}")
    print(f"patient_embeddings shape : {tuple(patient_embeddings.shape)}")

    return {
        "window_meta_df": window_meta_df,
        "window_embeddings": window_embeddings,
        "position_meta_df": position_meta_df,
        "position_embeddings": position_embeddings,
        "patient_meta_df": patient_meta_df,
        "patient_embeddings": patient_embeddings,
        "summary_df": summary_df,
    }


if __name__ == "__main__":
    import constants
    window_lib_path = Path(r"D:\PycharmProjects\HSCA\Outputs\validation\preprocessing\Data_windows\windows_4_5_4.0_1.0")
    patient_list = Path(r"D:\PycharmProjects\HSCA\Outputs\validation\preprocessing\Data_screened\passed_patients.csv")
    checkpoint_path = constants.CHECKPOINT_BEATS
    out_dir = Path(r"D:\PycharmProjects\HSCA\Outputs\validation\representation/beats_4s_1s")

    main_args = [
        "--window-lib-path", str(window_lib_path),
        "--patient-list", str(patient_list),
        "--checkpoint-path", str(checkpoint_path),
        "--batch-size", "64",
        "--num-workers", "4",
        "--device", "cuda",
        "--out-dir", str(out_dir),
    ]
    main(main_args)
