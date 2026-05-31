
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from aggregation import aggregate_positions_to_patient, build_position_embeddings
from ast_adapter import ASTAdapter
from beats_adapter import BEATsAdapter
from byola_adapter import BYOLAAdapter
from ead_adapter import EngineeredAcousticDescriptorAdapter
from panns_adapter import PANNsAdapter
from routes import SUPPORTED_ROUTE_NAMES, get_route_config
from window_dataset import WindowDataset, load_patient_ids


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


def _collate_waveform_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep waveforms as a list and metadata as parallel Python lists."""
    return {
        "patient_id": [item["patient_id"] for item in batch],
        "position": [item["position"] for item in batch],
        "window_id": [item["window_id"] for item in batch],
        "window_idx": [item["window_idx"] for item in batch],
        "window_library_path": [item["window_library_path"] for item in batch],
        "waveforms": [item["waveform"] for item in batch],
        "src_sr": int(batch[0]["src_sr"]),
    }


def build_adapter(
    route_name: str,
    checkpoint_path: Optional[str],
    stats_path: Optional[str],
    device: str,
):
    """Build the adapter object for one fixed route."""
    route_name = str(route_name).strip().lower()
    cfg = get_route_config(route_name)

    ckpt = checkpoint_path if checkpoint_path not in (None, "") else cfg.default_checkpoint_path
    stats = stats_path if stats_path not in (None, "") else cfg.default_stats_path

    if route_name == "panns":
        if not ckpt:
            raise ValueError("checkpoint_path is required for route='panns'")
        return PANNsAdapter(checkpoint_path=ckpt, device=device)

    if route_name == "ast":
        if not ckpt:
            raise ValueError("checkpoint_path is required for route='ast'")
        if not stats:
            raise ValueError("stats_path is required for route='ast'")
        return ASTAdapter(checkpoint_path=ckpt, stats_path=stats, device=device)

    if route_name == "beats":
        if not ckpt:
            raise ValueError("checkpoint_path is required for route='beats'")
        return BEATsAdapter(checkpoint_path=ckpt, device=device, trainable=False)

    if route_name == "beats_adapt":
        if not ckpt:
            raise ValueError("checkpoint_path is required for route='beats_adapt'")
        return BEATsAdapter(checkpoint_path=ckpt, device=device, trainable=False)

    if route_name == "byola":
        if not ckpt:
            raise ValueError("checkpoint_path is required for route='byola'")
        if not stats:
            raise ValueError("stats_path is required for route='byola'")
        return BYOLAAdapter(checkpoint_path=ckpt, stats_path=stats, device=device)

    if route_name == "ead":
        return EngineeredAcousticDescriptorAdapter(device=device)

    raise ValueError(
        f"Unsupported route_name={route_name!r}. Expected one of: {', '.join(SUPPORTED_ROUTE_NAMES)}"
    )


def run_window_embedding_extraction(
    route_name: str,
    dataset: WindowDataset,
    adapter,
    batch_size: int,
    num_workers: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Extract one embedding vector for every window in the dataset."""
    if len(dataset) == 0:
        raise ValueError("dataset is empty")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")

    _ = route_name  # stable public signature; route-specific logic lives in the adapter

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_waveform_batch,
    )

    meta_rows: List[Dict[str, Any]] = []
    emb_batches: List[np.ndarray] = []

    progress = tqdm(loader, desc=f"Extracting {route_name} window embeddings", leave=False)
    for batch in progress:
        emb = adapter.extract_embeddings(
            waveforms=batch["waveforms"],
            src_sr=int(batch["src_sr"]),
        )
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError(f"adapter.extract_embeddings() must return [B, D], but got {emb.shape}")
        if emb.shape[0] != len(batch["waveforms"]):
            raise ValueError(
                f"Embedding batch size mismatch: got {emb.shape[0]} embeddings for {len(batch['waveforms'])} inputs"
            )

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

    if len(meta_rows) == 0 or len(emb_batches) == 0:
        raise ValueError("No embeddings were extracted")

    window_meta_df = pd.DataFrame(meta_rows)
    window_embeddings = np.concatenate(emb_batches, axis=0).astype(np.float32, copy=False)

    if len(window_meta_df) != window_embeddings.shape[0]:
        raise ValueError(
            f"window_meta_df and window_embeddings size mismatch: {len(window_meta_df)} vs {window_embeddings.shape[0]}"
        )

    return window_meta_df, window_embeddings


def save_embedding_artifacts(
    out_dir: str,
    window_meta_df: pd.DataFrame,
    window_embeddings: np.ndarray,
    position_meta_df: pd.DataFrame,
    position_embeddings: np.ndarray,
    patient_meta_df: pd.DataFrame,
    patient_embeddings: np.ndarray,
) -> None:
    """Save window/position/patient metadata and embeddings using the fixed layout."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    window_meta_df.to_csv(out_path / "window_meta.csv", index=False, encoding="utf-8-sig")
    np.save(out_path / "window_embeddings.npy", np.asarray(window_embeddings, dtype=np.float32))

    position_meta_df.to_csv(out_path / "position_meta.csv", index=False, encoding="utf-8-sig")
    np.save(out_path / "position_embeddings.npy", np.asarray(position_embeddings, dtype=np.float32))

    patient_meta_df.to_csv(out_path / "patient_meta.csv", index=False, encoding="utf-8-sig")
    np.save(out_path / "patient_embeddings.npy", np.asarray(patient_embeddings, dtype=np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract window/position/patient embeddings for one representation-learning route."
    )
    parser.add_argument("--route", type=str, required=True, choices=SUPPORTED_ROUTE_NAMES)
    parser.add_argument("--window-lib-path", type=str, required=True)
    parser.add_argument("--patient-list", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--stats-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    print_banner("Representation extraction")
    log_info(f"route            : {args.route}")
    log_info(f"window_lib_path  : {args.window_lib_path}")
    log_info(f"patient_list     : {args.patient_list}")
    log_info(f"device           : {args.device}")
    log_info(f"batch_size       : {args.batch_size}")
    log_info(f"num_workers      : {args.num_workers}")
    log_info(f"out_dir          : {args.out_dir}")

    patient_ids = load_patient_ids(args.patient_list)
    log_done(f"Loaded patient IDs | n={len(patient_ids)}")

    dataset = WindowDataset(
        window_index_csv=f"{args.window_lib_path}/window_index.csv",
        recording_manifest_csv=f"{args.window_lib_path}/recording_manifest.csv",
        patient_ids=patient_ids,
    )
    log_done(f"Constructed WindowDataset | n_windows={len(dataset)}")

    adapter = build_adapter(
        route_name=args.route,
        checkpoint_path=args.checkpoint_path,
        stats_path=args.stats_path,
        device=args.device,
    )
    log_done("Built adapter")

    window_meta_df, window_embeddings = run_window_embedding_extraction(
        route_name=args.route,
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
    )
    log_done(f"Position embeddings built | shape={tuple(position_embeddings.shape)}")
    log_done(f"Patient embeddings built  | shape={tuple(patient_embeddings.shape)}")

    save_embedding_artifacts(
        out_dir=args.out_dir,
        window_meta_df=window_meta_df,
        window_embeddings=window_embeddings,
        position_meta_df=position_meta_df,
        position_embeddings=position_embeddings,
        patient_meta_df=patient_meta_df,
        patient_embeddings=patient_embeddings,
    )

    print_banner("Done")
    print(f"Saved embedding artifacts to: {args.out_dir}")
    print(f"window_embeddings shape:   {tuple(window_embeddings.shape)}")
    print(f"position_embeddings shape: {tuple(position_embeddings.shape)}")
    print(f"patient_embeddings shape:  {tuple(patient_embeddings.shape)}")


if __name__ == "__main__":
    main()
