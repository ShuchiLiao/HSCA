#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate patient-level acoustic descriptor profile from preprocessed heart-sound windows.

Input
-----
Outputs/preprocessing/Data_windows/windows_<min_windows>_<min_positions>_<window_sec>_<stride_sec>/window_library/
or any folder with the structure:
    window_library/<patient_id>/A_windows.npy
    window_library/<patient_id>/E_windows.npy
    window_library/<patient_id>/M_windows.npy
    window_library/<patient_id>/P_windows.npy
    window_library/<patient_id>/T_windows.npy

Output
------
patient_acoustic_descriptor_profile.csv

"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


POSITION_ORDER = ["A", "E", "M", "P", "T"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate patient-level acoustic descriptor profile CSV.")
    parser.add_argument("--window-library-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--fs", type=float, default=8000.0)
    parser.add_argument("--max-windows-per-position", type=int, default=40)
    return parser.parse_args(argv)


def read_csv_safely(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def _safe_float_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    x = x.astype(np.float32, copy=False)
    max_abs = np.nanmax(np.abs(x)) if x.size else 1.0
    if max_abs > 10:
        x = x / 32768.0
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def autocorr_periodicity(x: np.ndarray, fs: float, lag_min_sec: float = 0.30, lag_max_sec: float = 2.00) -> float:
    if x.size < int(lag_min_sec * fs) + 2:
        return np.nan
    x = x - np.mean(x)
    denom = np.sum(x ** 2) + 1e-12
    ac = np.correlate(x, x, mode="full")[x.size - 1:]
    ac = ac / denom
    lo = int(lag_min_sec * fs)
    hi = min(int(lag_max_sec * fs), len(ac))
    if hi <= lo:
        return np.nan
    return float(np.nanmax(ac[lo:hi]))


def compute_window_features(windows: np.ndarray, fs: float) -> pd.DataFrame:
    w = _safe_float_array(windows)
    rows = []

    for x in w:
        if x.size < 8:
            continue

        x = x - np.mean(x)
        rms = float(np.sqrt(np.mean(x ** 2)))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(x).astype(int))))) if x.size > 1 else np.nan

        spec = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
        total = float(np.sum(spec) + 1e-12)

        centroid = float(np.sum(freqs * spec) / total)
        p = spec / total
        entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p) + 1e-12))

        def band(lo: float, hi: float) -> float:
            mask = (freqs >= lo) & (freqs < hi)
            return float(np.sum(spec[mask]) / total) if mask.any() else np.nan

        rows.append({
            "RMS energy": rms,
            "Zero-crossing rate": zcr,
            "Spectral centroid": centroid,
            "Low-frequency energy ratio": band(20, 100),
            "Mid-frequency energy ratio": band(100, 250),
            "High-frequency energy ratio": band(250, 600),
            "Murmur-band energy ratio": band(150, 600),
            "Spectral entropy": entropy,
            "S1/S2 periodicity proxy": autocorr_periodicity(x, fs),
        })

    return pd.DataFrame(rows)


def list_patient_ids(window_library_dir: Path) -> list[str]:
    if not window_library_dir.exists():
        raise FileNotFoundError(f"window_library_dir does not exist: {window_library_dir}")
    patient_ids = sorted([p.name for p in window_library_dir.iterdir() if p.is_dir()])
    if not patient_ids:
        raise FileNotFoundError(f"No patient folders found under: {window_library_dir}")
    return patient_ids


def compute_patient_acoustic_features(
    window_library_dir: Path,
    patient_ids: Sequence[str],
    fs: float,
    max_windows_per_position: int,
) -> pd.DataFrame:
    rows = []

    for i, pid in enumerate(patient_ids):
        pdir = window_library_dir / str(pid)
        pos_features = []

        for pos in POSITION_ORDER:
            npy = pdir / f"{pos}_windows.npy"
            if not npy.exists():
                continue

            try:
                arr = np.load(npy, mmap_mode="r")
                if arr.ndim == 1:
                    arr = arr[None, :]

                if arr.shape[0] > max_windows_per_position:
                    idx = np.linspace(0, arr.shape[0] - 1, max_windows_per_position).round().astype(int)
                    arr = np.asarray(arr[idx])
                else:
                    arr = np.asarray(arr)

                feats = compute_window_features(arr, fs)
                if not feats.empty:
                    feats["position"] = pos
                    pos_features.append(feats)

            except Exception as e:
                warnings.warn(f"Could not process {npy}: {e}")

        if not pos_features:
            continue

        all_feat = pd.concat(pos_features, ignore_index=True)
        agg = all_feat.drop(columns=["position"], errors="ignore").mean(numeric_only=True).to_dict()
        agg["patient_id"] = str(pid)
        rows.append(agg)

        if (i + 1) % 100 == 0:
            print(f"[info] Acoustic descriptors processed for {i + 1}/{len(patient_ids)} patients")

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(
            "No acoustic descriptors were generated. Check whether window_library_dir contains "
            "<patient_id>/<position>_windows.npy files."
        )

    cols = ["patient_id"] + [c for c in out.columns if c != "patient_id"]
    return out[cols]


def main(argv=None) -> None:
    args = parse_args(argv)

    patient_ids = list_patient_ids(args.window_library_dir)
    descriptor_df = compute_patient_acoustic_features(
        window_library_dir=args.window_library_dir,
        patient_ids=patient_ids,
        fs=args.fs,
        max_windows_per_position=args.max_windows_per_position,
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    descriptor_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print(f"[done] Saved patient acoustic descriptor profile: {args.out_csv}")
    print(f"[done] Patients: {len(descriptor_df)}")
    print(f"[done] Descriptor columns: {len(descriptor_df.columns) - 1}")


if __name__ == "__main__":
    from pathlib import Path
    import constants

    min_windows_per_position = constants.MIN_WINDOWN_PER_POSTISIONS
    patient_pass_min_positions = constants.PATIENT_PASS_MIN_POSTISIONS
    window_sec = constants.WINDOW_SEC
    stride_sec = constants.STRIDE_SEC

    window_library_dir = (
        constants.OUTPUT_FOLDER / "preprocessing" / "Data_windows" /
        f"windows_{min_windows_per_position}_{patient_pass_min_positions}_{window_sec}_{stride_sec}" /
        "window_library"
    )

    out_csv = (
        constants.OUTPUT_FOLDER / "alignment" / "acoustic_descriptors" /
        "patient_acoustic_descriptor_profile.csv"
    )

    main_args = [
        "--window-library-dir", str(window_library_dir),
        "--out-csv", str(out_csv),
        "--fs", "8000",
        "--max-windows-per-position", "30",
    ]

    main(main_args)