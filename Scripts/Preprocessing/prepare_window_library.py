from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import re

import numpy as np
import pandas as pd


@dataclass
class PrepConfig:
    # Input data
    input_dir: str = "./Data_trimmed"
    passed_patients_csv: str = ".passed_patients.csv"
    output_dir: str = "./Data_representation"

    # Signal assumptions
    raw_sample_rate: int = 8000
    position_pattern: str = r"^([AEMPT])\.npy$"

    # Windowing
    window_sec: float = 4.0
    stride_sec: float = 1.0
    drop_last_short_tail: bool = True

    # Lightweight amplitude normalization
    amp_norm_percentile: float = 99.0
    amp_norm_eps: float = 1e-8

    # Optional value clipping after normalization to prevent a few extreme points
    # from dominating storage / later visualization. Set to None to disable.
    clip_value_after_norm: Optional[float] = 5.0


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def detect_patient_id_col(df: pd.DataFrame) -> str:
    candidates = ["patient_id", "编码", "id", "PID", "pid"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Cannot find patient id column. Available columns: {list(df.columns)}")


def load_passed_patient_ids(csv_path: str | Path) -> List[str]:
    df = pd.read_csv(csv_path, dtype=str)
    if len(df) == 0:
        return []

    patient_col = detect_patient_id_col(df)
    df[patient_col] = df[patient_col].astype(str).str.strip()

    # If the table already contains only passed patients, keep all rows.
    # If a pass flag exists, filter it conservatively.
    possible_flag_cols = ["patient_candidate_pass", "patient_pass", "pass", "is_passed"]
    for col in possible_flag_cols:
        if col in df.columns:
            flag = df[col].astype(str).str.strip().str.lower()
            df = df[flag.isin(["1", "true", "yes", "y", "pass", "passed"])]
            break

    ids = sorted(df[patient_col].dropna().astype(str).str.strip().unique().tolist())
    return ids


def read_npy_signal_as_float32(npy_path: str | Path) -> np.ndarray:
    x = np.load(npy_path, allow_pickle=False)
    if x.ndim != 1:
        raise ValueError("not_mono_1d_signal")
    if x.size == 0:
        raise ValueError("empty_signal")

    if np.issubdtype(x.dtype, np.integer):
        x = np.asarray(x, dtype=np.float32) / 32768.0
    elif np.issubdtype(x.dtype, np.floating):
        x = np.asarray(x, dtype=np.float32)
    else:
        raise ValueError(f"unsupported_dtype:{x.dtype}")

    return x.astype(np.float32, copy=False)


def remove_dc(x: np.ndarray) -> np.ndarray:
    """
    Remove the DC component by subtracting the signal mean.

    Intuition:
        If the whole waveform sits slightly above or below zero, that constant
        offset is called a DC component. It is usually not useful heart-sound
        information. Subtracting the mean recenters the waveform around zero.
    """
    return (x - np.mean(x, dtype=np.float64)).astype(np.float32, copy=False)


def robust_amplitude_normalize(
    x: np.ndarray,
    percentile: float = 99.0,
    eps: float = 1e-8,
    clip_value: Optional[float] = 5.0,
) -> Tuple[np.ndarray, float]:
    abs_x = np.abs(x)
    scale = float(np.percentile(abs_x, percentile)) if x.size > 0 else 0.0
    scale = max(scale, eps)
    x_norm = (x / scale).astype(np.float32, copy=False)
    if clip_value is not None:
        x_norm = np.clip(x_norm, -float(clip_value), float(clip_value)).astype(np.float32, copy=False)
    return x_norm, scale


def generate_fixed_windows(
    n_samples: int,
    sr: int,
    window_sec: float,
    stride_sec: float,
    drop_last_short_tail: bool = True,
) -> List[Tuple[int, int]]:
    win = int(round(window_sec * sr))
    hop = int(round(stride_sec * sr))
    if n_samples < win:
        return []

    last_start = n_samples - win
    starts = list(range(0, last_start + 1, hop))
    windows = [(st, st + win) for st in starts]

    next_start = starts[-1] + hop if starts else 0
    if not drop_last_short_tail and next_start < n_samples and (not starts or starts[-1] != last_start):
        windows.append((last_start, n_samples))
    return windows


def stack_windows(x: np.ndarray, windows: List[Tuple[int, int]]) -> np.ndarray:
    if len(windows) == 0:
        return np.empty((0, 0), dtype=np.float32)
    win_len = windows[0][1] - windows[0][0]
    out = np.empty((len(windows), win_len), dtype=np.float32)
    for i, (st, ed) in enumerate(windows):
        out[i] = x[st:ed]
    return out


def process_one_recording(
    patient_id: str,
    position: str,
    source_npy_path: Path,
    cfg: PrepConfig,
) -> Tuple[dict, List[dict]]:
    x = read_npy_signal_as_float32(source_npy_path)
    x = remove_dc(x)
    x, norm_scale = robust_amplitude_normalize(
        x,
        percentile=cfg.amp_norm_percentile,
        eps=cfg.amp_norm_eps,
        clip_value=cfg.clip_value_after_norm,
    )

    windows = generate_fixed_windows(
        n_samples=len(x),
        sr=cfg.raw_sample_rate,
        window_sec=cfg.window_sec,
        stride_sec=cfg.stride_sec,
        drop_last_short_tail=cfg.drop_last_short_tail,
    )

    patient_out_dir = ensure_dir(Path(cfg.output_dir) / "window_library" / patient_id)
    lib_path = patient_out_dir / f"{position}_windows.npy"
    np.save(lib_path, stack_windows(x, windows))

    record_row = {
        "patient_id": patient_id,
        "position": position,
        "source_npy_path": str(source_npy_path),
        "window_library_path": str(lib_path),
        "raw_num_samples": int(len(x)),
        "raw_duration_sec": float(len(x) / cfg.raw_sample_rate),
        "amp_norm_percentile": float(cfg.amp_norm_percentile),
        "amp_norm_scale": float(norm_scale),
        "window_sec": float(cfg.window_sec),
        "stride_sec": float(cfg.stride_sec),
        "n_windows": int(len(windows)),
        "win_num_samples": int(round(cfg.window_sec * cfg.raw_sample_rate)),
    }

    index_rows = []
    for i, (st, ed) in enumerate(windows):
        index_rows.append({
            "patient_id": patient_id,
            "position": position,
            "window_id": f"{patient_id}_{position}_w{i:04d}",
            "window_idx": int(i),
            "source_npy_path": str(source_npy_path),
            "window_library_path": str(lib_path),
            "start_sample": int(st),
            "end_sample": int(ed),
            "start_sec": float(st / cfg.raw_sample_rate),
            "end_sec": float(ed / cfg.raw_sample_rate),
            "duration_sec": float((ed - st) / cfg.raw_sample_rate),
        })

    return record_row, index_rows


def run(cfg: PrepConfig):
    out_dir = ensure_dir(cfg.output_dir)
    record_rows = []
    index_rows = []
    missing_rows = []

    passed_ids = load_passed_patient_ids(cfg.passed_patients_csv)
    if len(passed_ids) == 0:
        raise ValueError("No passed patients found in passed_patients.csv")

    pos_re = re.compile(cfg.position_pattern, re.IGNORECASE)
    input_root = Path(cfg.input_dir)
    if not input_root.exists():
        raise FileNotFoundError(f"input_dir not found: {input_root}")

    for patient_id in passed_ids:
        patient_dir = input_root / patient_id
        if not patient_dir.exists() or not patient_dir.is_dir():
            missing_rows.append({"patient_id": patient_id, "position": "ALL", "reason": "patient_dir_missing"})
            continue

        files = sorted([p for p in patient_dir.iterdir() if p.is_file() and p.suffix.lower() == ".npy"])
        found_any = False
        for f in files:
            m = pos_re.match(f.name)
            if not m:
                continue
            found_any = True
            position = m.group(1).upper()
            try:
                rec_row, idx_rows = process_one_recording(patient_id, position, f, cfg)
                record_rows.append(rec_row)
                index_rows.extend(idx_rows)
            except Exception as e:
                missing_rows.append({"patient_id": patient_id, "position": position, "reason": str(e)})

        if not found_any:
            missing_rows.append({"patient_id": patient_id, "position": "ALL", "reason": "no_position_npy_found"})

    recordings_df = pd.DataFrame(record_rows)
    if len(recordings_df) > 0:
        recordings_df = recordings_df.sort_values(["patient_id", "position"]).reset_index(drop=True)
    windows_index_df = pd.DataFrame(index_rows)
    if len(windows_index_df) > 0:
        windows_index_df = windows_index_df.sort_values(["patient_id", "position", "window_idx"]).reset_index(drop=True)
    missing_df = pd.DataFrame(missing_rows)

    recordings_df.to_csv(out_dir / "recording_manifest.csv", index=False, encoding="utf-8-sig")
    windows_index_df.to_csv(out_dir / "window_index.csv", index=False, encoding="utf-8-sig")
    missing_df.to_csv(out_dir / "missing_or_failed_records.csv", index=False, encoding="utf-8-sig")

    print(f"Saved to: {out_dir}")
    print("Generated files:")
    print("  - recording_manifest.csv")
    print("  - window_index.csv")
    print("  - missing_or_failed_records.csv")
    print("  - window_library/<patient_id>/<position>_windows.npy")


if __name__ == "__main__":
    window_sec = 4
    stride_sec = 1
    cfg = PrepConfig(
        input_dir="./Data_trimmed",
        passed_patients_csv=f"passed_patients_4_win_5_pos_{window_sec}_{stride_sec}.csv",
        output_dir=f"./window_lib_{window_sec}_{stride_sec}",
        raw_sample_rate=8000,
        window_sec=window_sec,
        stride_sec=stride_sec,
        drop_last_short_tail=True,
        amp_norm_percentile=99.0,
        clip_value_after_norm=5.0,
    )
    run(cfg)
