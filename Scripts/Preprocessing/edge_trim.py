from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


@dataclass
class EdgeTrimConfig:
    excel_path: str = "../Data/patient_info.xlsx"
    output_dir: str = "./Data_preprocessed"

    patient_id_col: str = "编码"
    position_path_cols: Dict[str, str] = None

    raw_sample_rate: int = 8000
    raw_dtype: np.dtype = np.int16
    endian: str = "little"

    bandpass_low_hz: float = 25.0
    bandpass_high_hz: float = 800.0
    bandpass_order: int = 4

    # After dynamic edge trimming, additionally remove the first 5% and
    # the last 10% of the dynamically retained segment.
    edge_post_trim_head_ratio: float = 0.05
    edge_post_trim_tail_ratio: float = 0.10

    # Dynamic edge trimming by block similarity.
    edge_trim_enabled: bool = True
    edge_ref_start_ratio: float = 0.40
    edge_ref_end_ratio: float = 0.60
    edge_slide_ratio: float = 0.01
    edge_consecutive_bad_windows: int = 5
    edge_env_smooth_ms: float = 50.0
    edge_feature_dim: int = 256
    edge_band_count: int = 16
    edge_acf_keep_ratio: float = 0.25
    edge_band_weight: float = 0.5
    edge_amp_gate_enabled: bool = True
    edge_amp_percentile: float = 95.0
    edge_amp_log_sigma: float = np.log(2) / np.sqrt(-np.log(0.9))
    edge_sim_thresh: float = 0.80

    num_workers: Optional[int] = None
    chunksize: int = 16
    progress_bar_width: int = 28
    progress_refresh_sec: float = 0.2
    manifest_sample_n: Optional[int] = None

    plot_patient_figures: bool = True
    plot_positions: Tuple[str, ...] = ("A", "E", "M", "P", "T")
    plot_subdir: str = "_plots_edge_trim"
    plot_max_points_per_position: int = 12000
    plot_linewidth: float = 0.5

    def __post_init__(self):
        if self.position_path_cols is None:
            self.position_path_cols = {
                "A": "A_path",
                "E": "E_path",
                "P": "P_path",
                "M": "M_path",
                "T": "T_path",
            }

    @property
    def filter_sos(self) -> np.ndarray:
        nyq = self.raw_sample_rate / 2.0
        low = self.bandpass_low_hz / nyq
        high = self.bandpass_high_hz / nyq
        return butter(self.bandpass_order, [low, high], btype="bandpass", output="sos")


_CFG: Optional[EdgeTrimConfig] = None
_FILTER_SOS: Optional[np.ndarray] = None


def _init_worker(cfg: EdgeTrimConfig):
    global _CFG, _FILTER_SOS
    _CFG = cfg
    _FILTER_SOS = cfg.filter_sos


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def expand_manifest_from_excel(cfg: EdgeTrimConfig) -> pd.DataFrame:
    df = pd.read_excel(cfg.excel_path, dtype={cfg.patient_id_col: str})
    df[cfg.patient_id_col] = df[cfg.patient_id_col].astype(str).str.strip()
    if cfg.manifest_sample_n is not None:
        df = df.sample(n=min(cfg.manifest_sample_n, len(df)), random_state=42).reset_index(drop=True)

    rows = []
    for row in df.itertuples(index=False):
        patient_id = str(getattr(row, cfg.patient_id_col)).strip()
        for position, col in cfg.position_path_cols.items():
            pcm_path = getattr(row, col, None)
            if pd.notna(pcm_path) and str(pcm_path).strip() != "":
                rows.append({"patient_id": patient_id, "position": position, "pcm_path": str(pcm_path)})
    return pd.DataFrame(rows)


def read_pcm_int16_mono(pcm_path: str, cfg: EdgeTrimConfig) -> np.ndarray:
    raw = Path(pcm_path).read_bytes()
    if len(raw) == 0:
        raise ValueError("empty_file")
    if len(raw) % 2 != 0:
        raise ValueError("odd_number_of_bytes")
    dtype = "<i2" if cfg.endian == "little" else ">i2"
    x = np.frombuffer(raw, dtype=dtype).copy()
    if x.ndim != 1:
        raise ValueError("not_mono_1d_signal")
    return x


def save_trimmed_signal_npy(npy_path: str | Path, x_int16: np.ndarray):
    np.save(npy_path, np.asarray(x_int16, dtype=np.int16))


def load_trimmed_signal_npy(npy_path: str | Path) -> np.ndarray:
    x = np.load(npy_path)
    return np.asarray(x, dtype=np.int16)


def pcm_int16_to_float(x_int16: np.ndarray) -> np.ndarray:
    return x_int16.astype(np.float32) / 32768.0


def remove_dc(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x, dtype=np.float64)


def bandpass_filter_with_sos(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    return sosfiltfilt(sos, x).astype(np.float32, copy=False)


def _resample_signal_1d(x: np.ndarray, out_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out_len = max(8, int(out_len))
    if x.size == 0:
        return np.zeros(out_len, dtype=np.float32)
    if x.size == out_len:
        return x.astype(np.float32, copy=False)
    xp = np.arange(x.size, dtype=np.float64)
    xq = np.linspace(0, x.size - 1, out_len, dtype=np.float64)
    return np.interp(xq, xp, x).astype(np.float32)


def _normalize_vector(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.mean(x, dtype=np.float64)
    norm = float(np.linalg.norm(x))
    return x / max(norm, eps)


def _moving_average_1d(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    win = max(1, int(win))
    if x.size == 0 or win <= 1:
        return x.astype(np.float32, copy=False)
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, kernel, mode="same").astype(np.float32, copy=False)


def _compute_smoothed_envelope(window: np.ndarray, cfg: EdgeTrimConfig) -> np.ndarray:
    smooth_win = max(1, int(round(cfg.edge_env_smooth_ms * 1e-3 * cfg.raw_sample_rate)))
    return _moving_average_1d(np.abs(window), smooth_win)


def _envelope_amplitude_summary(env: np.ndarray, cfg: EdgeTrimConfig) -> float:
    env = np.asarray(env, dtype=np.float32)
    if env.size == 0:
        return 0.0
    q = float(np.clip(cfg.edge_amp_percentile, 50.0, 100.0))
    return float(np.percentile(env, q))


def _amplitude_gate(ref_amp: float, cand_amp: float, cfg: EdgeTrimConfig, eps: float = 1e-8) -> float:
    if not cfg.edge_amp_gate_enabled:
        return 1.0
    sigma = max(float(cfg.edge_amp_log_sigma), 1e-6)
    d = abs(np.log((float(cand_amp) + eps) / (float(ref_amp) + eps)))
    return float(np.exp(- (d / sigma) ** 2))


def _band_energy_feature(env: np.ndarray, cfg: EdgeTrimConfig) -> np.ndarray:
    env_rs = _resample_signal_1d(env, cfg.edge_feature_dim)
    spec = np.fft.rfft(env_rs)
    power = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32, copy=False)
    power = np.maximum(power, 1e-12)

    n_bins = power.size
    n_bands = max(1, min(int(cfg.edge_band_count), n_bins))
    edges = np.linspace(0, n_bins, n_bands + 1).round().astype(np.int32)
    feats = []
    for i in range(n_bands):
        s, e = int(edges[i]), int(edges[i + 1])
        if e <= s:
            e = min(n_bins, s + 1)
        feats.append(float(np.sum(power[s:e], dtype=np.float64)))
    feats = np.log(np.asarray(feats, dtype=np.float32) + 1e-12)
    return _normalize_vector(feats)


def _autocorr_feature(env: np.ndarray, cfg: EdgeTrimConfig) -> np.ndarray:
    x = _resample_signal_1d(env, cfg.edge_feature_dim).astype(np.float32, copy=False)
    x = x - np.mean(x, dtype=np.float64)
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        keep = max(8, int(round(cfg.edge_feature_dim * cfg.edge_acf_keep_ratio)))
        return np.zeros(keep, dtype=np.float32)

    n = len(x)
    nfft = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(X * np.conj(X), n=nfft)[:n]
    ac = (ac / denom).astype(np.float32, copy=False)

    keep = max(8, min(n - 1, int(round(n * cfg.edge_acf_keep_ratio))))
    feat = ac[1:keep + 1]
    return _normalize_vector(feat)


def _window_similarity_from_features(
    ref_band: np.ndarray,
    ref_acf: np.ndarray,
    ref_amp: float,
    cand_band: np.ndarray,
    cand_acf: np.ndarray,
    cand_amp: float,
    cfg: EdgeTrimConfig,
) -> float:
    sim_band = float(np.dot(ref_band, cand_band))
    sim_acf = float(np.dot(ref_acf, cand_acf))
    sim_struct = float(cfg.edge_band_weight * sim_band + (1.0 - cfg.edge_band_weight) * sim_acf)
    g_amp = _amplitude_gate(ref_amp, cand_amp, cfg)
    return float(sim_struct * g_amp)


def _find_first_consecutive_low(similarity_list: List[float], m: int, thresh: float) -> Optional[int]:
    if len(similarity_list) == 0:
        return None
    m = max(1, int(m))
    arr = np.asarray(similarity_list, dtype=np.float32)
    if arr.size < m:
        return None
    for i in range(0, arr.size - m + 1):
        if np.all(arr[i:i + m] < thresh):
            return i
    return None


def compute_edge_block_similarity(x_filt: np.ndarray, cfg: EdgeTrimConfig) -> Dict[str, object]:
    n = len(x_filt)
    if n == 0:
        return {
            "ref_start": 0,
            "ref_end": 0,
            "ref_width": 0,
            "slide": 0,
            "head_starts": [],
            "tail_starts": [],
            "head_similarity": [],
            "tail_similarity": [],
        }

    ref_start = int(round(cfg.edge_ref_start_ratio * n))
    ref_end = int(round(cfg.edge_ref_end_ratio * n))
    ref_start = max(0, min(ref_start, n - 1))
    ref_end = max(ref_start + 1, min(ref_end, n))
    ref_width = ref_end - ref_start
    slide = max(1, int(round(cfg.edge_slide_ratio * n)))

    ref_window = x_filt[ref_start:ref_end]
    ref_env = _compute_smoothed_envelope(ref_window, cfg)
    ref_band = _band_energy_feature(ref_env, cfg)
    ref_acf = _autocorr_feature(ref_env, cfg)
    ref_amp = _envelope_amplitude_summary(ref_env, cfg)

    head_starts: List[int] = []
    head_similarity: List[float] = []
    s = ref_start - ref_width
    while s >= 0:
        cand = x_filt[s:s + ref_width]
        cand_env = _compute_smoothed_envelope(cand, cfg)
        cand_band = _band_energy_feature(cand_env, cfg)
        cand_acf = _autocorr_feature(cand_env, cfg)
        cand_amp = _envelope_amplitude_summary(cand_env, cfg)
        head_starts.append(int(s))
        head_similarity.append(_window_similarity_from_features(ref_band, ref_acf, ref_amp, cand_band, cand_acf, cand_amp, cfg))
        s -= slide

    tail_starts: List[int] = []
    tail_similarity: List[float] = []
    s = ref_end
    while s + ref_width <= n:
        cand = x_filt[s:s + ref_width]
        cand_env = _compute_smoothed_envelope(cand, cfg)
        cand_band = _band_energy_feature(cand_env, cfg)
        cand_acf = _autocorr_feature(cand_env, cfg)
        cand_amp = _envelope_amplitude_summary(cand_env, cfg)
        tail_starts.append(int(s))
        tail_similarity.append(_window_similarity_from_features(ref_band, ref_acf, ref_amp, cand_band, cand_acf, cand_amp, cfg))
        s += slide

    return {
        "ref_start": int(ref_start),
        "ref_end": int(ref_end),
        "ref_width": int(ref_width),
        "slide": int(slide),
        "head_starts": head_starts,
        "tail_starts": tail_starts,
        "head_similarity": head_similarity,
        "tail_similarity": tail_similarity,
    }


def detect_dynamic_edge_truncation(x_filt: np.ndarray, cfg: EdgeTrimConfig) -> Dict[str, object]:
    n = len(x_filt)
    result = {
        "edge_trim_applied": 0,
        "keep_start_sample": 0,
        "keep_end_sample": int(n),
        "keep_start_ratio": 0.0,
        "keep_end_ratio": 1.0 if n > 0 else 0.0,
        "head_trim_applied": 0,
        "tail_trim_applied": 0,
        "head_trim_ratio": 0.0,
        "tail_trim_ratio": 0.0,
        "head_trim_sec": 0.0,
        "tail_trim_sec": 0.0,
        "trim_total_sec": 0.0,
        "trim_total_ratio": 0.0,
        "head_reason": "",
        "tail_reason": "",
        "ref_window_sec": 0.0,
        "slide_sec": 0.0,
        "n_head_windows": 0,
        "n_tail_windows": 0,
        "head_first_bad_similarity": np.nan,
        "tail_first_bad_similarity": np.nan,
        "head_similarity_median": np.nan,
        "tail_similarity_median": np.nan,
    }
    if (not cfg.edge_trim_enabled) or n == 0:
        return result

    info = compute_edge_block_similarity(x_filt, cfg)
    ref_width = int(info["ref_width"])
    slide = int(info["slide"])
    head_starts = info["head_starts"]
    tail_starts = info["tail_starts"]
    head_similarity = info["head_similarity"]
    tail_similarity = info["tail_similarity"]

    result["ref_window_sec"] = float(ref_width / cfg.raw_sample_rate)
    result["slide_sec"] = float(slide / cfg.raw_sample_rate)
    result["n_head_windows"] = int(len(head_starts))
    result["n_tail_windows"] = int(len(tail_starts))
    if len(head_similarity) > 0:
        result["head_similarity_median"] = float(np.median(head_similarity))
    if len(tail_similarity) > 0:
        result["tail_similarity_median"] = float(np.median(tail_similarity))

    m = max(1, int(cfg.edge_consecutive_bad_windows))

    head_bad_idx = _find_first_consecutive_low(head_similarity, m, cfg.edge_sim_thresh)
    if head_bad_idx is not None:
        bad_start = int(head_starts[head_bad_idx])
        keep_start = bad_start + ref_width
        result["head_first_bad_similarity"] = float(head_similarity[head_bad_idx])
    else:
        keep_start = 0

    tail_bad_idx = _find_first_consecutive_low(tail_similarity, m, cfg.edge_sim_thresh)
    if tail_bad_idx is not None:
        keep_end = int(tail_starts[tail_bad_idx])
        result["tail_first_bad_similarity"] = float(tail_similarity[tail_bad_idx])
    else:
        keep_end = n

    keep_start = max(0, min(keep_start, n))
    keep_end = max(keep_start, min(keep_end, n))

    head_trim_ratio = keep_start / float(max(n, 1))
    tail_trim_ratio = max(0.0, (n - keep_end) / float(max(n, 1)))
    head_applied = int(keep_start > 0)
    tail_applied = int(keep_end < n)

    result.update({
        "edge_trim_applied": int(head_applied or tail_applied),
        "keep_start_sample": int(keep_start),
        "keep_end_sample": int(keep_end),
        "keep_start_ratio": float(head_trim_ratio),
        "keep_end_ratio": float(keep_end / float(max(n, 1))),
        "head_trim_applied": head_applied,
        "tail_trim_applied": tail_applied,
        "head_trim_ratio": float(head_trim_ratio),
        "tail_trim_ratio": float(tail_trim_ratio),
        "head_trim_sec": float(keep_start / cfg.raw_sample_rate),
        "tail_trim_sec": float((n - keep_end) / cfg.raw_sample_rate),
        "trim_total_sec": float((keep_start + (n - keep_end)) / cfg.raw_sample_rate),
        "trim_total_ratio": float((keep_start + (n - keep_end)) / float(max(n, 1))),
        "head_reason": "head_m_consecutive_low_similarity" if head_applied else "",
        "tail_reason": "tail_m_consecutive_low_similarity" if tail_applied else "",
    })
    return result


def detect_edge_truncation_with_initial_crop(x_filt_full: np.ndarray, cfg: EdgeTrimConfig) -> Dict[str, object]:
    n_full = len(x_filt_full)

    # Step 1: dynamic edge truncation directly on the full recording.
    dyn = detect_dynamic_edge_truncation(x_filt_full, cfg)
    dyn_keep_start = int(dyn["keep_start_sample"])
    dyn_keep_end = int(dyn["keep_end_sample"])
    dyn_keep_start = max(0, min(dyn_keep_start, n_full))
    dyn_keep_end = max(dyn_keep_start, min(dyn_keep_end, n_full))

    # Step 2: fixed post-trim on the dynamically retained segment.
    n_after_dynamic = max(0, dyn_keep_end - dyn_keep_start)
    post_head = int(round(cfg.edge_post_trim_head_ratio * n_after_dynamic))
    post_tail = int(round(cfg.edge_post_trim_tail_ratio * n_after_dynamic))
    if post_head + post_tail >= n_after_dynamic:
        overflow = post_head + post_tail - max(n_after_dynamic - 1, 0)
        if overflow > 0:
            reduce_tail = min(post_tail, overflow)
            post_tail -= reduce_tail
            overflow -= reduce_tail
        if overflow > 0:
            post_head = max(0, post_head - overflow)

    final_keep_start = dyn_keep_start + post_head
    final_keep_end = dyn_keep_end - post_tail
    final_keep_start = max(0, min(final_keep_start, n_full))
    final_keep_end = max(final_keep_start, min(final_keep_end, n_full))

    post_head_sec = post_head / float(cfg.raw_sample_rate)
    post_tail_sec = post_tail / float(cfg.raw_sample_rate)
    post_total_sec = (post_head + post_tail) / float(cfg.raw_sample_rate)
    post_total_ratio_full = (post_head + post_tail) / float(max(n_full, 1))
    total_trim_sec = (final_keep_start + (n_full - final_keep_end)) / float(cfg.raw_sample_rate)
    total_trim_ratio = (final_keep_start + (n_full - final_keep_end)) / float(max(n_full, 1))

    return {
        **dyn,
        "n_samples_full": int(n_full),
        "n_samples_after_dynamic_trim": int(n_after_dynamic),
        "post_trim_head_ratio": float(cfg.edge_post_trim_head_ratio),
        "post_trim_tail_ratio": float(cfg.edge_post_trim_tail_ratio),
        "post_trim_head_samples": int(post_head),
        "post_trim_tail_samples": int(post_tail),
        "post_trim_head_sec": float(post_head_sec),
        "post_trim_tail_sec": float(post_tail_sec),
        "post_trim_total_sec": float(post_total_sec),
        "post_trim_total_ratio_full": float(post_total_ratio_full),
        "dynamic_keep_start_sample_full": int(dyn_keep_start),
        "dynamic_keep_end_sample_full": int(dyn_keep_end),
        "final_keep_start_sample": int(final_keep_start),
        "final_keep_end_sample": int(final_keep_end),
        "final_duration_sec": float((final_keep_end - final_keep_start) / cfg.raw_sample_rate),
        "overall_trim_total_sec": float(total_trim_sec),
        "overall_trim_total_ratio": float(total_trim_ratio),
    }


def process_one_recording_local(row: Dict[str, str], cfg: EdgeTrimConfig) -> Tuple[Dict[str, object], Dict[str, object]]:
    patient_id = str(row["patient_id"])
    position = str(row["position"])
    pcm_path = str(row["pcm_path"])
    out_npy_path = str(Path(cfg.output_dir) / patient_id / f"{position}.npy")

    file_fail_reason = ""
    try:
        x_raw_int16_full = read_pcm_int16_mono(pcm_path, cfg)
        x_filt_full = bandpass_filter_with_sos(remove_dc(pcm_int16_to_float(x_raw_int16_full)), cfg.filter_sos)
        edge_info = detect_edge_truncation_with_initial_crop(x_filt_full, cfg)

        keep_start = max(0, min(int(edge_info["final_keep_start_sample"]), len(x_raw_int16_full)))
        keep_end = max(keep_start, min(int(edge_info["final_keep_end_sample"]), len(x_raw_int16_full)))
        x_raw_int16 = x_raw_int16_full[keep_start:keep_end]

        if x_raw_int16.size == 0:
            raise ValueError("no_samples_after_edge_trimming")

        ensure_dir(Path(cfg.output_dir) / patient_id)
        save_trimmed_signal_npy(out_npy_path, x_raw_int16)

    except Exception as e:
        x_raw_int16_full = np.array([], dtype=np.int16)
        x_raw_int16 = np.array([], dtype=np.int16)
        file_fail_reason = str(e)
        edge_info = {
            "edge_trim_applied": 0,
            "keep_start_sample": 0,
            "keep_end_sample": 0,
            "keep_start_ratio": 0.0,
            "keep_end_ratio": 0.0,
            "head_trim_applied": 0,
            "tail_trim_applied": 0,
            "head_trim_ratio": 0.0,
            "tail_trim_ratio": 0.0,
            "head_trim_sec": 0.0,
            "tail_trim_sec": 0.0,
            "trim_total_sec": 0.0,
            "trim_total_ratio": 0.0,
            "head_reason": "",
            "tail_reason": "",
            "ref_window_sec": np.nan,
            "slide_sec": np.nan,
            "n_head_windows": 0,
            "n_tail_windows": 0,
            "head_first_bad_similarity": np.nan,
            "tail_first_bad_similarity": np.nan,
            "head_similarity_median": np.nan,
            "tail_similarity_median": np.nan,
            "n_samples_full": 0,
            "n_samples_after_dynamic_trim": 0,
            "post_trim_head_ratio": float(cfg.edge_post_trim_head_ratio),
            "post_trim_tail_ratio": float(cfg.edge_post_trim_tail_ratio),
            "post_trim_head_samples": 0,
            "post_trim_tail_samples": 0,
            "post_trim_head_sec": 0.0,
            "post_trim_tail_sec": 0.0,
            "post_trim_total_sec": 0.0,
            "post_trim_total_ratio_full": 0.0,
            "dynamic_keep_start_sample_full": 0,
            "dynamic_keep_end_sample_full": 0,
            "final_keep_start_sample": 0,
            "final_keep_end_sample": 0,
            "final_duration_sec": 0.0,
            "overall_trim_total_sec": 0.0,
            "overall_trim_total_ratio": 0.0,
        }

    recording_row = {
        "patient_id": patient_id,
        "position": position,
        "input_pcm_path": pcm_path,
        "output_npy_path": out_npy_path,
        "raw_num_samples_original": int(len(x_raw_int16_full)),
        "raw_duration_sec_original": float(len(x_raw_int16_full) / cfg.raw_sample_rate if len(x_raw_int16_full) > 0 else 0.0),
        "raw_num_samples_after_edge_trim": int(len(x_raw_int16)),
        "raw_duration_sec_after_edge_trim": float(len(x_raw_int16) / cfg.raw_sample_rate if len(x_raw_int16) > 0 else 0.0),
        "file_fail_reason": file_fail_reason,
        "recording_edge_pass": int((file_fail_reason == "") and (len(x_raw_int16) > 0)),
        **edge_info,
    }

    edge_row = {
        "patient_id": patient_id,
        "position": position,
        "input_pcm_path": pcm_path,
        "output_npy_path": out_npy_path,
        "raw_duration_sec_original": float(len(x_raw_int16_full) / cfg.raw_sample_rate if len(x_raw_int16_full) > 0 else 0.0),
        "raw_duration_sec_after_edge_trim": float(len(x_raw_int16) / cfg.raw_sample_rate if len(x_raw_int16) > 0 else 0.0),
        **edge_info,
        "file_fail_reason": file_fail_reason,
    }
    return recording_row, edge_row


def _process_one_recording_worker(row: Dict[str, str]):
    return process_one_recording_local(row, _CFG)


def summarize_patient_edge_qc(recordings_qc: pd.DataFrame) -> pd.DataFrame:
    if len(recordings_qc) == 0:
        return pd.DataFrame()
    rows = []
    for patient_id, g in recordings_qc.groupby("patient_id", dropna=False):
        positions_found = sorted(g["position"].astype(str).tolist())
        positions_pass = sorted(g.loc[g["recording_edge_pass"] == 1, "position"].astype(str).tolist())
        rows.append({
            "patient_id": patient_id,
            "n_positions_found": int(len(positions_found)),
            "n_positions_pass": int(len(positions_pass)),
            "positions_found": "|".join(positions_found),
            "positions_pass": "|".join(positions_pass),
            "n_positions_dynamic_edge_trimmed": int(g.get("edge_trim_applied", pd.Series(dtype=int)).fillna(0).sum()) if "edge_trim_applied" in g.columns else 0,
            "total_overall_trim_sec": float(g.get("overall_trim_total_sec", pd.Series(dtype=float)).fillna(0).sum()) if "overall_trim_total_sec" in g.columns else 0.0,
        })
    return pd.DataFrame(rows)


def load_filtered_signal_for_plot(npy_path: str, cfg: EdgeTrimConfig) -> np.ndarray:
    x_raw_int16 = load_trimmed_signal_npy(npy_path)
    return bandpass_filter_with_sos(remove_dc(pcm_int16_to_float(x_raw_int16)), cfg.filter_sos)


def load_filtered_original_signal_for_plot(pcm_path: str, cfg: EdgeTrimConfig) -> np.ndarray:
    x_raw_int16 = read_pcm_int16_mono(pcm_path, cfg)
    return bandpass_filter_with_sos(remove_dc(pcm_int16_to_float(x_raw_int16)), cfg.filter_sos)


def downsample_for_plot(y: np.ndarray, sr: int, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    if y.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    step = max(1, int(np.ceil(y.size / max(max_points, 1))))
    y_plot = y[::step]
    t_plot = (np.arange(y_plot.size, dtype=np.float32) * step) / float(sr)
    return t_plot, y_plot


def _plot_signal_or_text(ax, x: np.ndarray, cfg: EdgeTrimConfig, empty_text: str):
    ax.grid(True, alpha=0.2, linewidth=0.4)
    if x.size == 0:
        ax.text(0.5, 0.5, empty_text, transform=ax.transAxes, ha="center", va="center")
        return
    t_plot, y_plot = downsample_for_plot(x, cfg.raw_sample_rate, cfg.plot_max_points_per_position)
    ax.plot(t_plot, y_plot, linewidth=cfg.plot_linewidth)


def save_patient_waveform_figures(recordings_qc: pd.DataFrame, cfg: EdgeTrimConfig):
    if (not cfg.plot_patient_figures) or len(recordings_qc) == 0:
        return
    plot_dir = ensure_dir(Path(cfg.output_dir) / cfg.plot_subdir)
    patient_ids = sorted(recordings_qc["patient_id"].astype(str).unique().tolist())
    total = len(patient_ids)
    if total == 0:
        return

    print(f"Start plotting patient figures: {total} patients -> {plot_dir}")
    start_time = time.time()
    done = 0
    last_print = 0.0
    recordings_by_patient = {str(pid): g.copy() for pid, g in recordings_qc.groupby("patient_id", dropna=False)}

    for patient_id in patient_ids:
        g_rec = recordings_by_patient.get(patient_id, pd.DataFrame())
        fig, axes = plt.subplots(len(cfg.plot_positions), 2, figsize=(18, 10), sharex=False)
        if len(cfg.plot_positions) == 1:
            axes = np.asarray([axes])

        axes[0, 0].set_title("Original waveform (before trim)")
        axes[0, 1].set_title("Trimmed waveform (after trim)")

        for row_idx, position in enumerate(cfg.plot_positions):
            ax_left = axes[row_idx, 0]
            ax_right = axes[row_idx, 1]

            ax_left.set_ylabel(position, rotation=0, labelpad=18, va="center")
            rec_pos = g_rec[g_rec["position"].astype(str) == position]
            if len(rec_pos) == 0:
                _plot_signal_or_text(ax_left, np.array([], dtype=np.float32), cfg, "Missing recording")
                _plot_signal_or_text(ax_right, np.array([], dtype=np.float32), cfg, "Missing recording")
                continue

            rec_row = rec_pos.iloc[0]
            input_pcm_path = str(rec_row.get("input_pcm_path", "") or "")
            output_npy_path = str(rec_row.get("output_npy_path", "") or "")
            file_fail_reason = str(rec_row.get("file_fail_reason", "") or "")
            post_head_trim = float(rec_row.get("post_trim_head_ratio", 0.0) or 0.0)
            post_tail_trim = float(rec_row.get("post_trim_tail_ratio", 0.0) or 0.0)
            dyn_head = float(rec_row.get("head_trim_ratio", 0.0) or 0.0)
            dyn_tail = float(rec_row.get("tail_trim_ratio", 0.0) or 0.0)
            dyn_applied = int(rec_row.get("edge_trim_applied", 0) or 0)

            try:
                x_filt_orig = load_filtered_original_signal_for_plot(input_pcm_path, cfg)
                _plot_signal_or_text(ax_left, x_filt_orig, cfg, "Empty original signal")
            except Exception as e:
                _plot_signal_or_text(ax_left, np.array([], dtype=np.float32), cfg, f"Plot failed: {e}")

            if file_fail_reason:
                _plot_signal_or_text(ax_right, np.array([], dtype=np.float32), cfg, f"Trim failed: {file_fail_reason}")
                continue

            try:
                x_filt_trim = load_filtered_signal_for_plot(output_npy_path, cfg)
                _plot_signal_or_text(ax_right, x_filt_trim, cfg, "No retained samples after trimming")
                extra = f", dyn H {100 * dyn_head:.1f}% / T {100 * dyn_tail:.1f}%" if dyn_applied == 1 else ", dyn none"
                ax_right.text(0.99, 0.92, f"post fixed H {100 * post_head_trim:.1f}% / T {100 * post_tail_trim:.1f}%{extra}", transform=ax_right.transAxes, ha="right", va="top", fontsize=9)
            except Exception as e:
                _plot_signal_or_text(ax_right, np.array([], dtype=np.float32), cfg, f"Plot failed: {e}")

        axes[-1, 0].set_xlabel("Time before trim (s)")
        axes[-1, 1].set_xlabel("Time after trim (s)")
        fig.suptitle(f"{patient_id} | AEMPT waveform comparison before vs after edge trimming", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(plot_dir / f"{patient_id}.png", dpi=200)
        plt.close(fig)

        done += 1
        now = time.time()
        if (now - last_print) >= cfg.progress_refresh_sec or done == total:
            _print_progress(done, total, start_time, cfg.progress_bar_width, final=(done == total), stage_label="Plotting  ")
            last_print = now


def _progress_bar(done: int, total: int, width: int = 28) -> str:
    filled = int(width * done / total) if total > 0 else 0
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _print_progress(done: int, total: int, start_time: float, width: int = 28, final: bool = False, stage_label: str = "Processing"):
    elapsed = max(time.time() - start_time, 1e-6)
    speed = done / elapsed
    eta = (total - done) / speed if speed > 1e-12 else float("inf")
    bar = _progress_bar(done, total, width)
    msg = f"\r{stage_label} {done}/{total} {bar}  {100.0 * done / max(total, 1):6.2f}%  {speed:6.2f} rec/s  ETA {eta:7.1f}s"
    print(msg, end="\n" if final else "", flush=True)


def run_edge_trim(cfg: EdgeTrimConfig):
    out_dir = ensure_dir(cfg.output_dir)
    manifest = expand_manifest_from_excel(cfg)
    manifest.to_csv(out_dir / "manifest_long_raw.csv", index=False, encoding="utf-8-sig")
    if len(manifest) == 0:
        print("manifest is empty, nothing to process.")
        return

    rows = manifest.to_dict("records")
    recording_rows: List[Dict[str, object]] = []
    edge_rows: List[Dict[str, object]] = []

    max_workers = cfg.num_workers if cfg.num_workers is not None else max(1, min((os.cpu_count() or 1) - 1, 8))
    print(f"Start edge trimming: {len(rows)} recordings, workers={max_workers}, chunksize={cfg.chunksize}")
    start_time = time.time()
    done = 0
    last_print = 0.0

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(cfg,)) as ex:
        for rec_row, edge_row in ex.map(_process_one_recording_worker, rows, chunksize=cfg.chunksize):
            recording_rows.append(rec_row)
            edge_rows.append(edge_row)
            done += 1
            now = time.time()
            if (now - last_print) >= cfg.progress_refresh_sec or done == len(rows):
                _print_progress(done, len(rows), start_time, cfg.progress_bar_width, final=(done == len(rows)), stage_label="Processing")
                last_print = now

    recordings_qc = pd.DataFrame(recording_rows).sort_values(["patient_id", "position"]).reset_index(drop=True)
    edge_truncation_df = pd.DataFrame(edge_rows).sort_values(["patient_id", "position"]).reset_index(drop=True)
    patients_qc = summarize_patient_edge_qc(recordings_qc)

    recordings_qc.to_csv(out_dir / "recordings_qc.csv", index=False, encoding="utf-8-sig")
    edge_truncation_df.to_csv(out_dir / "edge_truncation.csv", index=False, encoding="utf-8-sig")
    patients_qc.to_csv(out_dir / "patients_qc.csv", index=False, encoding="utf-8-sig")

    manifest_preprocessed = recordings_qc.loc[recordings_qc["recording_edge_pass"] == 1, ["patient_id", "position", "output_npy_path"]].copy()
    manifest_preprocessed.to_csv(out_dir / "manifest_long_preprocessed.csv", index=False, encoding="utf-8-sig")

    save_patient_waveform_figures(recordings_qc, cfg)

    print(f"Saved to: {out_dir}")
    print("Generated files:")
    print("  - manifest_long_raw.csv")
    print("  - manifest_long_preprocessed.csv")
    print("  - recordings_qc.csv")
    print("  - edge_truncation.csv")
    print("  - patients_qc.csv")
    print("  - <output_dir>/<patient_id>/<position>.npy")
    if cfg.plot_patient_figures:
        print(f"  - {cfg.plot_subdir}/<patient_id>.png")


if __name__ == "__main__":
    cfg = EdgeTrimConfig(
        excel_path="../Data/patient_info.xlsx",
        output_dir="./Data_trimmed",
        patient_id_col="编码",
        position_path_cols={"A": "A_path", "E": "E_path", "P": "P_path", "M": "M_path", "T": "T_path"},
        raw_sample_rate=8000,
        edge_post_trim_head_ratio=0.05,
        edge_post_trim_tail_ratio=0.10,
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
        edge_sim_thresh=0.80,
        num_workers=8,
        chunksize=16,
        manifest_sample_n=None,
        plot_patient_figures=True,
        plot_subdir="_plots_edge_trim",
    )
    run_edge_trim(cfg)
