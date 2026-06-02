from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from torch.utils.data import DataLoader

from audio_utils import resample_waveform, right_pad_waveforms, waveforms_to_logmel
from routes import get_route_config
from window_dataset import WindowDataset, load_patient_ids
import constants


ALLOWED_ROUTES = ("ast", )


def _collate_waveform_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep waveform arrays as a plain Python list for later audio processing."""
    return {
        "patient_id": [item["patient_id"] for item in batch],
        "position": [item["position"] for item in batch],
        "window_id": [item["window_id"] for item in batch],
        "window_idx": [item["window_idx"] for item in batch],
        "window_library_path": [item["window_library_path"] for item in batch],
        "waveforms": [item["waveform"] for item in batch],
        "src_sr": int(batch[0]["src_sr"]),
    }


def compute_global_stats(
    route_name: str,
    dataset: WindowDataset,
    batch_size: int,
    device: str,
) -> Dict[str, Any]:
    """Compute global scalar mean/std on training-set log-Mel features.

    Parameters
    ----------
    route_name:
        Only "ast" or "byola" are allowed.
    dataset:
        Training-window dataset.
    batch_size:
        Batch size used for feature computation.
    device:
        Included for interface consistency. This script does not run a model,
        so the current implementation does not need to move data to the device.

    Returns
    -------
    dict
        {
            "route_name": str,
            "n_windows": int,
            "mean": float,
            "std": float,
            "target_sr": int,
            "n_mels": int,
            "frame_ms": float,
            "hop_ms": float,
            "fmin": float,
            "fmax": float,
        }
    """
    route_name = str(route_name).strip().lower()
    if route_name not in ALLOWED_ROUTES:
        raise ValueError(f"route_name must be one of {ALLOWED_ROUTES}, but got: {route_name!r}")
    if len(dataset) == 0:
        raise ValueError("dataset is empty")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    _ = device  # kept only for a stable public function signature
    cfg = get_route_config(route_name)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_waveform_batch,
    )

    total_count = 0
    total_sum = 0.0
    total_sq_sum = 0.0
    total_windows = 0

    for batch in loader:
        waveforms_rs = [
            resample_waveform(w, orig_sr=int(batch["src_sr"]), target_sr=int(cfg.target_sr))
            for w in batch["waveforms"]
        ]
        padded_waveforms, _ = right_pad_waveforms(waveforms_rs, pad_value=0.0)

        feats = waveforms_to_logmel(
            waveforms=padded_waveforms,
            sample_rate=int(cfg.target_sr),
            n_mels=int(cfg.n_mels),
            frame_ms=float(cfg.frame_ms),
            hop_ms=float(cfg.hop_ms),
            fmin=float(cfg.fmin),
            fmax=float(cfg.fmax),
        )
        feats64 = np.asarray(feats, dtype=np.float64)

        total_windows += int(feats64.shape[0])
        total_count += int(feats64.size)
        total_sum += float(feats64.sum())
        total_sq_sum += float(np.square(feats64).sum())

    if total_count == 0:
        raise ValueError("No feature values were accumulated when computing global stats")

    mean = total_sum / total_count
    var = max(total_sq_sum / total_count - mean * mean, 0.0)
    std = float(np.sqrt(var))

    return {
        "route_name": route_name,
        "n_windows": int(total_windows),
        "mean": float(mean),
        "std": float(std),
        "target_sr": int(cfg.target_sr),
        "n_mels": int(cfg.n_mels),
        "frame_ms": float(cfg.frame_ms),
        "hop_ms": float(cfg.hop_ms),
        "fmin": float(cfg.fmin),
        "fmax": float(cfg.fmax),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute global log-Mel mean/std for AST on the training window set."
    )
    parser.add_argument("--route", type=str, required=True, choices=ALLOWED_ROUTES)
    parser.add_argument("--window-lib-path", type=str, required=True)
    # parser.add_argument("--window-index-csv", type=str, required=True)
    # parser.add_argument("--recording-manifest-csv", type=str, required=True)
    parser.add_argument("--train-patient-list", type=str, required=True)
    parser.add_argument("--out-json", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    patient_ids = load_patient_ids(args.train_patient_list)
    window_lib_path = Path(args.window_lib_path)

    dataset = WindowDataset(
        window_index_csv=window_lib_path / "window_index.csv",
        recording_manifest_csv=window_lib_path / "recording_manifest.csv",
        patient_ids=patient_ids,
    )

    stats = compute_global_stats(
        route_name=args.route,
        dataset=dataset,
        batch_size=int(args.batch_size),
        device=args.device,
    )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Saved stats json to: {out_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    min_windows_per_position = constants.MIN_WINDOWN_PER_POSTISIONS
    patient_pass_min_positions = constants.PATIENT_PASS_MIN_POSTISIONS
    window_sec = constants.WINDOW_SEC
    stride_sec = constants.STRIDE_SEC
    route = "ast"
    window_lib_path = (constants.OUTPUT_FOLDER / "preprocessing" / "Data_windows" /
                       f"windows_{min_windows_per_position}_{patient_pass_min_positions}_{window_sec}_{stride_sec}")
    train_patient_list = (constants.OUTPUT_FOLDER / "representation"/"Data_split"/
                          f"splits_{min_windows_per_position}_{patient_pass_min_positions}_{window_sec}_{stride_sec}" /
                          "train_patients.txt")
    out_json = (constants.OUTPUT_FOLDER / "representation" / "Stats" /
                                                f"{route}_global_logmel_stats_"
                                                f"{min_windows_per_position}_"
                                                f"{patient_pass_min_positions}_"
                                                f"{window_sec}_{stride_sec}.json")

    main([
        "--route", route,
        "--window-lib-path", str(window_lib_path),
        "--train-patient-list", str(train_patient_list),
        "--out-json", str(out_json),
        "--batch-size", "64",
        "--device", "cuda",
    ])
