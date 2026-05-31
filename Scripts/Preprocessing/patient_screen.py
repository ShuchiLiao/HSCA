from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import re

import numpy as np
import pandas as pd


@dataclass
class SimpleScreenConfig:
    input_dir: str = "./Data_trimmed"
    output_dir: str = "Data_screened"

    raw_sample_rate: int = 8000
    window_sec: float = 4.0
    stride_sec: float = 1.0
    drop_last_short_tail: bool = True

    min_windows_per_position: int = 3
    max_windows_per_position: int = 35
    patient_pass_min_positions: int = 4

    position_pattern: str = r"^([AEMPT])\.npy$"
    output_filename: str = f"passed_patients.csv"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def scan_manifest(cfg: SimpleScreenConfig) -> pd.DataFrame:
    root = Path(cfg.input_dir)
    if not root.exists():
        raise FileNotFoundError(f"input_dir not found: {root}")

    pos_re = re.compile(cfg.position_pattern, re.IGNORECASE)
    rows = []
    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir() or patient_dir.name.startswith("_"):
            continue
        patient_id = patient_dir.name.strip()
        for f in sorted(patient_dir.iterdir()):
            if not f.is_file() or f.suffix.lower() != ".npy":
                continue
            m = pos_re.match(f.name)
            if not m:
                continue
            position = m.group(1).upper()
            rows.append({"patient_id": patient_id, "position": position, "npy_path": str(f)})
    return pd.DataFrame(rows)


def read_npy_int16_mono(npy_path: str) -> np.ndarray:
    x = np.load(npy_path, allow_pickle=False)
    if x.ndim != 1:
        raise ValueError("not_mono_1d_signal")
    if x.size == 0:
        raise ValueError("empty_file")

    if np.issubdtype(x.dtype, np.integer):
        return np.asarray(x, dtype=np.int16)

    if np.issubdtype(x.dtype, np.floating):
        xmax = float(np.max(np.abs(x))) if x.size > 0 else 0.0
        if xmax <= 1.5:
            x = np.clip(np.round(x * 32768.0), -32768, 32767)
        else:
            x = np.clip(np.round(x), -32768, 32767)
        return x.astype(np.int16)

    raise ValueError(f"unsupported_dtype:{x.dtype}")


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


def evaluate_one_position(row: dict, cfg: SimpleScreenConfig) -> dict:
    patient_id = str(row["patient_id"])
    position = str(row["position"])
    npy_path = str(row["npy_path"])

    file_fail_reason = ""
    raw_num_samples = 0
    raw_duration_sec = 0.0
    n_candidate_windows = 0

    try:
        x_raw_int16 = read_npy_int16_mono(npy_path)
        raw_num_samples = int(len(x_raw_int16))
        raw_duration_sec = raw_num_samples / float(cfg.raw_sample_rate)
        windows = generate_fixed_windows(
            n_samples=raw_num_samples,
            sr=cfg.raw_sample_rate,
            window_sec=cfg.window_sec,
            stride_sec=cfg.stride_sec,
            drop_last_short_tail=cfg.drop_last_short_tail,
        )
        n_candidate_windows = len(windows)

        if n_candidate_windows < cfg.min_windows_per_position:
            file_fail_reason = "too_few_windows"
        if n_candidate_windows > cfg.max_windows_per_position:
            file_fail_reason = "too_many_windows"

    except Exception as e:
        file_fail_reason = str(e)

    return {
        "patient_id": patient_id,
        "position": position,
        "npy_path": npy_path,
        "raw_num_samples": raw_num_samples,
        "raw_duration_sec": raw_duration_sec,
        "window_sec": cfg.window_sec,
        "stride_sec": cfg.stride_sec,
        "n_candidate_windows": int(n_candidate_windows),
        "min_windows_required": int(cfg.min_windows_per_position),
        "position_pass": int(file_fail_reason == ""),
        "file_fail_reason": file_fail_reason,
    }


def summarize_patients(position_df: pd.DataFrame, cfg: SimpleScreenConfig) -> pd.DataFrame:
    if len(position_df) == 0:
        return pd.DataFrame(columns=[
            "patient_id", "n_positions_found", "n_positions_pass",
            "positions_found", "positions_pass", "patient_pass"
        ])

    rows = []
    for patient_id, g in position_df.groupby("patient_id", dropna=False):
        positions_found = sorted(g["position"].astype(str).tolist())
        positions_pass = sorted(g.loc[g["position_pass"] == 1, "position"].astype(str).tolist())
        patient_pass = int(len(positions_pass) >= cfg.patient_pass_min_positions)
        rows.append({
            "patient_id": patient_id,
            "n_positions_found": int(len(positions_found)),
            "n_positions_pass": int(len(positions_pass)),
            "positions_found": "|".join(positions_found),
            "positions_pass": "|".join(positions_pass),
            "patient_pass": patient_pass,
        })
    return pd.DataFrame(rows).sort_values(["patient_id"]).reset_index(drop=True)


def run_simple_screen(cfg: SimpleScreenConfig):
    out_dir = ensure_dir(cfg.output_dir)
    manifest = scan_manifest(cfg)
    if len(manifest) == 0:
        print("manifest is empty, nothing to process.")
        return

    position_rows = [evaluate_one_position(row, cfg) for row in manifest.to_dict("records")]
    position_df = pd.DataFrame(position_rows).sort_values(["patient_id", "position"]).reset_index(drop=True)
    patient_df = summarize_patients(position_df, cfg)
    passed_patient_df = patient_df.loc[patient_df["patient_pass"] == 1].copy().reset_index(drop=True)

    passed_patient_df.to_csv(out_dir / cfg.output_filename, index=False, encoding="utf-8-sig")

    print(f"Saved to: {out_dir / cfg.output_filename}")
    print(f"Passed patients: {len(passed_patient_df)} / {len(patient_df)}")


if __name__ == "__main__":
    min_windows_per_position = 4
    patient_pass_min_positions = 5
    window_sec = 4
    stride_sec = 1
    cfg = SimpleScreenConfig(
        input_dir="./Data_trimmed",
        output_dir="./",
        raw_sample_rate=8000,
        window_sec=window_sec,
        stride_sec=stride_sec,
        drop_last_short_tail=True,
        min_windows_per_position=min_windows_per_position,
        patient_pass_min_positions=patient_pass_min_positions,
        output_filename=f"passed_patients_{min_windows_per_position}_win_{patient_pass_min_positions}_pos"
                        f"_{window_sec}_{stride_sec}.csv"
    )
    run_simple_screen(cfg)
