#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare external Zhongshan First Hospital heart-sound WAV data for validation.

This script keeps the same preprocessing logic as the internal HSCA workflow:
1) build valid_recordings.csv from external patient folders and the clinical table;
2) apply edge trimming and save int16 npy files to Data_trimmed;
3) run patient_screen on trimmed npy files and save passed_patients.csv;
4) run prepare_window_library and save fixed-window arrays to Data_windows.

External naming rule
--------------------
External WAV files are named Point A.wav ... Point E.wav. They are mapped to
internal A/E/M/P/T positions as follows:
    Point B -> A_path  aortic area
    Point C -> E_path  second aortic / Erb area
    Point E -> M_path  mitral area
    Point A -> P_path  pulmonary area
    Point D -> T_path  tricuspid area

Usage
-----
Command line:
    python Scripts/Validation/preprocessing/prepare_external_heart_sound.py

PyCharm / Python:
    from Scripts.Validation.preprocessing.prepare_external_heart_sound import main
    outputs = main([])
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly


POSITION_ORDER = ["A", "E", "M", "P", "T"]
POINT_TO_INTERNAL = {
    "Point B.wav": "A",  # aortic area
    "Point C.wav": "E",  # second aortic / Erb area
    "Point E.wav": "M",  # mitral area
    "Point A.wav": "P",  # pulmonary area
    "Point D.wav": "T",  # tricuspid area
}
INTERNAL_TO_POINT = {v: k for k, v in POINT_TO_INTERNAL.items()}
NAME_COL_CANDIDATES = ["姓名", "患者姓名", "病人姓名", "name", "patient_name"]

_EDGE_CFG = None


def _project_root_from_this_file() -> Path:
    """Return project root when this file is placed under Scripts/Validation/preprocessing."""
    return Path(__file__).resolve().parents[3]


def _add_preprocessing_to_path(project_root: Path) -> None:
    """Allow importing Scripts/Preprocessing modules when running this file directly."""
    preprocessing_dir = project_root / "Scripts" / "Preprocessing"
    for p in [project_root, preprocessing_dir]:
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_existing_clinical_path(path: str | Path) -> Path:
    """Resolve common .xslx typo to .xlsx if needed."""
    path = Path(path)
    if path.exists():
        return path
    if path.suffix.lower() == ".xslx":
        candidate = path.with_suffix(".xlsx")
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"clinical table not found: {path}")


def normalize_text(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def detect_name_col(df: pd.DataFrame, user_col: Optional[str] = None) -> str:
    if user_col and user_col in df.columns:
        return user_col
    for col in NAME_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"Cannot find patient name column. Available columns: {list(df.columns)}")


def build_name_to_code_map(clinical_table: Path, patient_code_col: str, patient_name_col: Optional[str]) -> Dict[str, str]:
    df = pd.read_excel(clinical_table, dtype={patient_code_col: str})
    if patient_code_col not in df.columns:
        raise ValueError(f"Cannot find patient code column '{patient_code_col}'. Available columns: {list(df.columns)}")
    name_col = detect_name_col(df, patient_name_col)
    df[name_col] = df[name_col].map(normalize_text)
    df[patient_code_col] = df[patient_code_col].map(normalize_text)
    df = df[(df[name_col] != "") & (df[patient_code_col] != "")].copy()
    name_to_code = {}
    duplicated_names = set(df.loc[df[name_col].duplicated(keep=False), name_col].tolist())
    if duplicated_names:
        raise ValueError(f"Duplicated patient names in clinical table: {sorted(duplicated_names)[:10]}")
    for row in df.itertuples(index=False):
        name_to_code[getattr(row, name_col)] = getattr(row, patient_code_col)
    return name_to_code


def find_wav_case_insensitive(patient_dir: Path, filename: str) -> Optional[Path]:
    direct = patient_dir / filename
    if direct.exists():
        return direct
    target = filename.lower().replace(" ", "")
    for f in patient_dir.iterdir():
        if f.is_file() and f.name.lower().replace(" ", "") == target:
            return f
    return None


def build_valid_recordings(
    external_root: Path,
    clinical_table: Path,
    patient_code_col: str,
    patient_name_col: Optional[str],
    out_dir: Path,
) -> pd.DataFrame:
    """Build one-row-per-patient valid_recordings.csv required by the validation workflow."""
    external_root = Path(external_root)
    if not external_root.exists():
        raise FileNotFoundError(f"external wav root not found: {external_root}")

    name_to_code = build_name_to_code_map(clinical_table, patient_code_col, patient_name_col)
    valid_rows, excluded_rows = [], []

    for patient_dir in sorted(external_root.iterdir()):
        if not patient_dir.is_dir() or patient_dir.name.startswith("_"):
            continue
        patient_name = patient_dir.name.strip()
        patient_id = name_to_code.get(patient_name, "")
        if patient_id == "":
            excluded_rows.append({"patient_name": patient_name, "patient_id": "", "recording_dir": str(patient_dir), "reason": "name_not_found_in_clinical_table"})
            continue

        pos_paths = {}
        missing = []
        for internal_pos in POSITION_ORDER:
            point_file = INTERNAL_TO_POINT[internal_pos]
            wav_path = find_wav_case_insensitive(patient_dir, point_file)
            if wav_path is None:
                missing.append(point_file)
                pos_paths[f"{internal_pos}_path"] = ""
            else:
                pos_paths[f"{internal_pos}_path"] = str(wav_path)

        if missing:
            excluded_rows.append({"patient_name": patient_name, "patient_id": patient_id, "recording_dir": str(patient_dir), "reason": "missing_wav:" + "|".join(missing)})
            continue

        valid_rows.append({"病人姓名": patient_name, "病人编码": patient_id, "听诊录音地址": str(patient_dir), **pos_paths})

    valid_df = pd.DataFrame(valid_rows)
    excluded_df = pd.DataFrame(excluded_rows)
    ensure_dir(out_dir)
    valid_df.to_csv(out_dir / "valid_recordings.csv", index=False, encoding="utf-8-sig")
    excluded_df.to_csv(out_dir / "excluded_recordings.csv", index=False, encoding="utf-8-sig")
    return valid_df


def expand_valid_recordings_to_long(valid_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Convert valid_recordings.csv to the long manifest shape used by edge_trim.py."""
    rows = []
    for row in valid_df.itertuples(index=False):
        patient_id = str(getattr(row, "病人编码")).strip()
        patient_name = str(getattr(row, "病人姓名")).strip()
        for position in POSITION_ORDER:
            wav_path = str(getattr(row, f"{position}_path")).strip()
            rows.append({"patient_id": patient_id, "patient_name": patient_name, "position": position, "wav_path": wav_path})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest_long_raw.csv", index=False, encoding="utf-8-sig")
    return manifest


def read_wav_as_int16_mono(wav_path: str | Path, target_sr: int) -> np.ndarray:
    """Read WAV as mono int16. Resampling is only used when WAV fs differs from target_sr."""
    sr, x = wavfile.read(str(wav_path))
    if x.size == 0:
        raise ValueError("empty_file")
    if x.ndim == 2:
        x = x.astype(np.float32).mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        x_float = x.astype(np.float32) / float(max(abs(info.min), info.max))
    elif np.issubdtype(x.dtype, np.floating):
        x_float = x.astype(np.float32)
    else:
        raise ValueError(f"unsupported_wav_dtype:{x.dtype}")

    if sr != target_sr:
        gcd = int(np.gcd(sr, target_sr))
        x_float = resample_poly(x_float, target_sr // gcd, sr // gcd).astype(np.float32, copy=False)

    x_float = np.nan_to_num(x_float, nan=0.0, posinf=0.0, neginf=0.0)
    x_float = np.clip(x_float, -1.0, 1.0)
    return np.asarray(np.round(x_float * 32767.0), dtype=np.int16)


def _empty_edge_info(cfg) -> Dict[str, object]:
    return {
        "edge_trim_applied": 0, "keep_start_sample": 0, "keep_end_sample": 0, "keep_start_ratio": 0.0, "keep_end_ratio": 0.0,
        "head_trim_applied": 0, "tail_trim_applied": 0, "head_trim_ratio": 0.0, "tail_trim_ratio": 0.0,
        "head_trim_sec": 0.0, "tail_trim_sec": 0.0, "trim_total_sec": 0.0, "trim_total_ratio": 0.0,
        "head_reason": "", "tail_reason": "", "ref_window_sec": np.nan, "slide_sec": np.nan,
        "n_head_windows": 0, "n_tail_windows": 0, "head_first_bad_similarity": np.nan, "tail_first_bad_similarity": np.nan,
        "head_similarity_median": np.nan, "tail_similarity_median": np.nan, "n_samples_full": 0, "n_samples_after_dynamic_trim": 0,
        "post_trim_head_ratio": float(cfg.edge_post_trim_head_ratio), "post_trim_tail_ratio": float(cfg.edge_post_trim_tail_ratio),
        "post_trim_head_samples": 0, "post_trim_tail_samples": 0, "post_trim_head_sec": 0.0, "post_trim_tail_sec": 0.0,
        "post_trim_total_sec": 0.0, "post_trim_total_ratio_full": 0.0, "dynamic_keep_start_sample_full": 0,
        "dynamic_keep_end_sample_full": 0, "final_keep_start_sample": 0, "final_keep_end_sample": 0, "final_duration_sec": 0.0,
        "overall_trim_total_sec": 0.0, "overall_trim_total_ratio": 0.0,
    }


def process_one_wav_recording(row: Dict[str, str], cfg) -> Tuple[Dict[str, object], Dict[str, object]]:
    """WAV adapter for edge_trim.py. The trimming algorithm itself is unchanged."""
    from Scripts.Preprocessing import edge_trim as et

    patient_id = str(row["patient_id"])
    patient_name = str(row.get("patient_name", ""))
    position = str(row["position"])
    wav_path = str(row["wav_path"])
    out_npy_path = str(Path(cfg.output_dir) / patient_id / f"{position}.npy")
    file_fail_reason = ""

    try:
        x_raw_int16_full = read_wav_as_int16_mono(wav_path, cfg.raw_sample_rate)
        x_filt_full = et.bandpass_filter_with_sos(et.remove_dc(et.pcm_int16_to_float(x_raw_int16_full)), cfg.filter_sos)
        edge_info = et.detect_edge_truncation_with_initial_crop(x_filt_full, cfg)
        keep_start = max(0, min(int(edge_info["final_keep_start_sample"]), len(x_raw_int16_full)))
        keep_end = max(keep_start, min(int(edge_info["final_keep_end_sample"]), len(x_raw_int16_full)))
        x_raw_int16 = x_raw_int16_full[keep_start:keep_end]
        if x_raw_int16.size == 0:
            raise ValueError("no_samples_after_edge_trimming")
        et.ensure_dir(Path(cfg.output_dir) / patient_id)
        et.save_trimmed_signal_npy(out_npy_path, x_raw_int16)
    except Exception as e:
        x_raw_int16_full = np.array([], dtype=np.int16)
        x_raw_int16 = np.array([], dtype=np.int16)
        file_fail_reason = str(e)
        edge_info = _empty_edge_info(cfg)

    recording_row = {
        "patient_id": patient_id, "patient_name": patient_name, "position": position,
        "input_wav_path": wav_path, "output_npy_path": out_npy_path,
        "raw_num_samples_original": int(len(x_raw_int16_full)),
        "raw_duration_sec_original": float(len(x_raw_int16_full) / cfg.raw_sample_rate if len(x_raw_int16_full) > 0 else 0.0),
        "raw_num_samples_after_edge_trim": int(len(x_raw_int16)),
        "raw_duration_sec_after_edge_trim": float(len(x_raw_int16) / cfg.raw_sample_rate if len(x_raw_int16) > 0 else 0.0),
        "file_fail_reason": file_fail_reason,
        "recording_edge_pass": int((file_fail_reason == "") and (len(x_raw_int16) > 0)),
        **edge_info,
    }
    edge_row = {
        "patient_id": patient_id, "patient_name": patient_name, "position": position,
        "input_wav_path": wav_path, "output_npy_path": out_npy_path,
        "raw_duration_sec_original": recording_row["raw_duration_sec_original"],
        "raw_duration_sec_after_edge_trim": recording_row["raw_duration_sec_after_edge_trim"],
        **edge_info,
        "file_fail_reason": file_fail_reason,
    }
    return recording_row, edge_row


def _init_edge_worker(cfg) -> None:
    global _EDGE_CFG
    _EDGE_CFG = cfg


def _process_edge_worker(row: Dict[str, str]) -> Tuple[Dict[str, object], Dict[str, object]]:
    return process_one_wav_recording(row, _EDGE_CFG)


def run_edge_trim_for_external_wav(manifest: pd.DataFrame, cfg) -> None:
    """Run the repository edge-trim algorithm on external WAV files."""
    from Scripts.Preprocessing import edge_trim as et

    out_dir = ensure_dir(cfg.output_dir)
    rows = manifest.to_dict("records")
    if len(rows) == 0:
        print("manifest is empty, nothing to trim.")
        return

    recording_rows, edge_rows = [], []
    max_workers = cfg.num_workers if cfg.num_workers is not None else max(1, min((os.cpu_count() or 1) - 1, 8))
    print(f"Start external WAV edge trimming: {len(rows)} recordings, workers={max_workers}")
    start = time.time()

    if max_workers <= 1:
        for i, row in enumerate(rows, start=1):
            rec_row, edge_row = process_one_wav_recording(row, cfg)
            recording_rows.append(rec_row)
            edge_rows.append(edge_row)
            if i == len(rows) or i % 50 == 0:
                print(f"  processed {i}/{len(rows)} recordings")
    else:
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_edge_worker, initargs=(cfg,)) as ex:
            for i, (rec_row, edge_row) in enumerate(ex.map(_process_edge_worker, rows, chunksize=cfg.chunksize), start=1):
                recording_rows.append(rec_row)
                edge_rows.append(edge_row)
                if i == len(rows) or i % 50 == 0:
                    elapsed = max(time.time() - start, 1e-6)
                    print(f"  processed {i}/{len(rows)} recordings, {i / elapsed:.2f} rec/s")

    recordings_qc = pd.DataFrame(recording_rows).sort_values(["patient_id", "position"]).reset_index(drop=True)
    edge_df = pd.DataFrame(edge_rows).sort_values(["patient_id", "position"]).reset_index(drop=True)
    patients_qc = et.summarize_patient_edge_qc(recordings_qc)
    manifest_preprocessed = recordings_qc.loc[recordings_qc["recording_edge_pass"] == 1, ["patient_id", "position", "output_npy_path"]].copy()

    recordings_qc.to_csv(out_dir / "recordings_qc.csv", index=False, encoding="utf-8-sig")
    edge_df.to_csv(out_dir / "edge_truncation.csv", index=False, encoding="utf-8-sig")
    patients_qc.to_csv(out_dir / "patients_qc.csv", index=False, encoding="utf-8-sig")
    manifest_preprocessed.to_csv(out_dir / "manifest_long_preprocessed.csv", index=False, encoding="utf-8-sig")

    if cfg.plot_patient_figures:
        recordings_qc_for_plot = recordings_qc.copy()
        recordings_qc_for_plot["input_pcm_path"] = recordings_qc_for_plot["input_wav_path"]

        old_loader = et.load_filtered_original_signal_for_plot

        def _load_filtered_original_wav_for_plot(wav_path: str, plot_cfg) -> np.ndarray:
            x_raw_int16 = read_wav_as_int16_mono(wav_path, plot_cfg.raw_sample_rate)
            return et.bandpass_filter_with_sos(et.remove_dc(et.pcm_int16_to_float(x_raw_int16)), plot_cfg.filter_sos)

        et.load_filtered_original_signal_for_plot = _load_filtered_original_wav_for_plot
        try:
            et.save_patient_waveform_figures(recordings_qc_for_plot, cfg)
        finally:
            et.load_filtered_original_signal_for_plot = old_loader

    print(f"Saved trimmed npy files and QC tables to: {out_dir}")


def run_patient_screen(trimmed_dir: Path, screened_dir: Path, args) -> Path:
    from Scripts.Preprocessing import patient_screen as ps

    output_filename = args.passed_patients_filename
    cfg = ps.SimpleScreenConfig(
        input_dir=trimmed_dir,
        output_dir=screened_dir,
        raw_sample_rate=args.fs,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        drop_last_short_tail=True,
        min_windows_per_position=args.min_windows_per_position,
        max_windows_per_position=args.max_windows_per_position,
        patient_pass_min_positions=args.patient_pass_min_positions,
        position_pattern=r"^([AEMPT])\.npy$",
        output_filename=output_filename,
    )
    ps.run_simple_screen(cfg)
    return screened_dir / output_filename


def run_prepare_window_library(trimmed_dir: Path, passed_patients_csv: Path, windows_dir: Path, args) -> None:
    from Scripts.Preprocessing import prepare_window_library as pwl

    cfg = pwl.PrepConfig(
        input_dir=trimmed_dir,
        passed_patients_csv=passed_patients_csv,
        output_dir=windows_dir,
        raw_sample_rate=args.fs,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        drop_last_short_tail=True,
        amp_norm_percentile=args.amp_norm_percentile,
        clip_value_after_norm=args.clip_value_after_norm,
    )
    pwl.run(cfg)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Prepare external heart-sound WAV data for validation.")
    parser.add_argument("--external-wav-root", type=Path, default=Path(r"D:\TongJiPCG\同济心音外部验证\中山一院"))
    parser.add_argument("--clinical-table", type=Path, default=Path(r"D:\TongJiPCG\同济心音外部验证\中山一院听诊队列.xlsx"))
    parser.add_argument("--patient-code-col", type=str, default="序号")
    parser.add_argument("--patient-name-col", type=str, default=None)
    parser.add_argument("--out-root", type=Path, default=Path(r"D:\PycharmProjects\HSCA\Outputs\validation\preprocessing"))
    parser.add_argument("--fs", type=int, default=8000)
    parser.add_argument("--window-sec", type=float, default=4)
    parser.add_argument("--stride-sec", type=float, default=1)
    parser.add_argument("--min-windows-per-position", type=int, default=4)
    parser.add_argument("--max-windows-per-position", type=int, default=35)
    parser.add_argument("--patient-pass-min-positions", type=int, default=5)
    parser.add_argument("--edge-post-trim-head-ratio", type=float, default=0.05)
    parser.add_argument("--edge-post-trim-tail-ratio", type=float, default=0.10)
    parser.add_argument("--edge-sim-thresh", type=float, default=0.8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--chunksize", type=int, default=16)
    parser.add_argument("--no-edge-plots", action="store_true")
    parser.add_argument("--amp-norm-percentile", type=float, default=99.0)
    parser.add_argument("--clip-value-after-norm", type=float, default=5.0)
    parser.add_argument("--passed-patients-filename", type=str, default="passed_patients.csv")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    args = parse_args(argv)
    project_root = _project_root_from_this_file()
    _add_preprocessing_to_path(project_root)

    from Scripts.Preprocessing import edge_trim as et

    clinical_table = resolve_existing_clinical_path(args.clinical_table)
    out_root = ensure_dir(args.out_root)
    trimmed_dir = ensure_dir(out_root / "Data_trimmed")
    screened_dir = ensure_dir(out_root / "Data_screened")
    windows_dir = ensure_dir(out_root / "Data_windows" / f"windows_{args.min_windows_per_position}_{args.patient_pass_min_positions}_{args.window_sec}_{args.stride_sec}")

    print("Step 1/4: build valid_recordings.csv")
    valid_df = build_valid_recordings(args.external_wav_root, clinical_table, args.patient_code_col, args.patient_name_col, out_root)
    manifest = expand_valid_recordings_to_long(valid_df, out_root)
    print(f"Valid patients with five WAV files and clinical code: {len(valid_df)}")

    print("Step 2/4: edge trim external WAV files")
    edge_cfg = et.EdgeTrimConfig(
        excel_path=str(clinical_table),
        output_dir=str(trimmed_dir),
        patient_id_col=args.patient_code_col,
        position_path_cols={"A": "A_path", "E": "E_path", "M": "M_path", "P": "P_path", "T": "T_path"},
        raw_sample_rate=args.fs,
        edge_post_trim_head_ratio=args.edge_post_trim_head_ratio,
        edge_post_trim_tail_ratio=args.edge_post_trim_tail_ratio,
        edge_trim_enabled=True,
        edge_ref_start_ratio=0.40,
        edge_ref_end_ratio=0.60,
        edge_slide_ratio=0.01,
        edge_consecutive_bad_windows=5,
        edge_env_smooth_ms=50.0,
        edge_feature_dim=256,
        edge_band_count=16,
        edge_acf_keep_ratio=0.25,
        edge_band_weight=0.5,
        edge_sim_thresh=args.edge_sim_thresh,
        num_workers=args.num_workers,
        chunksize=args.chunksize,
        manifest_sample_n=None,
        plot_patient_figures=args.no_edge_plots,
        plot_subdir="_plots_edge_trim",
    )
    run_edge_trim_for_external_wav(manifest, edge_cfg)

    print("Step 3/4: patient screen")
    passed_patients_csv = run_patient_screen(trimmed_dir, screened_dir, args)

    print("Step 4/4: prepare window library")
    run_prepare_window_library(trimmed_dir, passed_patients_csv, windows_dir, args)

    outputs = {
        "valid_recordings_csv": out_root / "valid_recordings.csv",
        "excluded_recordings_csv": out_root / "excluded_recordings.csv",
        "trimmed_dir": trimmed_dir,
        "screened_dir": screened_dir,
        "passed_patients_csv": passed_patients_csv,
        "windows_dir": windows_dir,
        "window_index_csv": windows_dir / "window_index.csv",
        "recording_manifest_csv": windows_dir / "recording_manifest.csv",
    }
    print("\nDone. Main outputs:")
    for k, v in outputs.items():
        print(f" - {k}: {v}")
    return outputs


if __name__ == "__main__":
    main()
