from __future__ import annotations

"""
Segmentation-free engineered acoustic descriptors (EAD) for fixed-length heart-sound windows.

Design goals
------------
1. No dependence on heart-sound cycle segmentation.
2. Stable on fixed windows such as 4 s / 1 s stride.
3. Lightweight, fully CPU-based, and easy to reproduce.
4. Return one fixed-dimensional descriptor vector per window.

Recommended publication-facing name
-----------------------------------
Engineered Acoustic Descriptors (EAD)

Implementation notes
--------------------
- Input waveform is expected to be 1D and sampled at 8 kHz by default.
- The extractor uses normalized / ratio-style statistics whenever possible,
  so the descriptors remain reasonably stable across windows.
- No plotting is performed here. This file is a pure feature-extraction module.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.signal import hilbert, welch

EPS = 1e-8

DEFAULT_BANDS_HZ: Tuple[Tuple[float, float], ...] = (
    (20.0, 60.0),
    (60.0, 120.0),
    (120.0, 250.0),
    (250.0, 500.0),
    (500.0, 1000.0),
    (1000.0, 2000.0),
)


def _safe_float(x: float) -> float:
    x = float(x)
    return x if np.isfinite(x) else 0.0


def _validate_waveform(waveform: np.ndarray) -> np.ndarray:
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"waveform must be 1D, but got shape={x.shape}")
    if x.size < 32:
        raise ValueError(f"waveform is too short for stable feature extraction: n={x.size}")
    return np.ascontiguousarray(x)


def _zero_crossing_rate(x: np.ndarray) -> float:
    signs = np.signbit(x)
    zc = np.count_nonzero(signs[1:] != signs[:-1])
    return _safe_float(zc / max(len(x) - 1, 1))


def _shannon_energy(x: np.ndarray) -> float:
    p = np.square(x).astype(np.float64)
    p_sum = float(p.sum())
    if p_sum <= EPS:
        return 0.0
    p = p / p_sum
    return _safe_float(-np.sum(p * np.log(p + EPS)))


def _log_energy(x: np.ndarray) -> float:
    return _safe_float(np.log(np.mean(np.square(x), dtype=np.float64) + EPS))


def _envelope(x: np.ndarray) -> np.ndarray:
    env = np.abs(hilbert(x.astype(np.float64)))
    return np.asarray(env, dtype=np.float32)


def _normalized_autocorrelation(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64, copy=False)
    x64 = x64 - np.mean(x64)
    n = x64.size
    nfft = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(x64, n=nfft)
    acf = np.fft.irfft(spec * np.conjugate(spec), n=nfft)[:n]
    denom = float(acf[0]) if n > 0 else 0.0
    if denom <= EPS:
        return np.zeros(n, dtype=np.float32)
    acf = acf / denom
    return np.asarray(acf, dtype=np.float32)


def _acf_features(x: np.ndarray, sr: int) -> List[float]:
    acf = _normalized_autocorrelation(x)
    lag_min = int(round(0.25 * sr))
    lag_max = min(len(acf) - 1, int(round(1.50 * sr)))
    if lag_max <= lag_min:
        return [0.0, 0.0, 0.0]

    region = acf[lag_min : lag_max + 1]
    peak_rel = int(np.argmax(region))
    peak_idx = lag_min + peak_rel
    peak_val = _safe_float(region[peak_rel])
    peak_lag_sec = _safe_float(peak_idx / float(sr))

    one_sec = min(len(acf) - 1, int(round(1.0 * sr)))
    below = np.where(acf[1 : one_sec + 1] < 0.5)[0]
    decay_sec = _safe_float((below[0] + 1) / float(sr)) if below.size > 0 else _safe_float(one_sec / float(sr))
    return [peak_val, peak_lag_sec, decay_sec]


def _spectral_features(x: np.ndarray, sr: int) -> Tuple[List[float], np.ndarray, np.ndarray]:
    nperseg = min(1024, len(x))
    noverlap = nperseg // 2
    freqs, psd = welch(
        x.astype(np.float64, copy=False),
        fs=float(sr),
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )
    psd = np.asarray(psd, dtype=np.float64)
    psd = np.maximum(psd, EPS)

    power_sum = float(psd.sum())
    p = psd / power_sum
    centroid = _safe_float(np.sum(freqs * p))
    bandwidth = _safe_float(np.sqrt(np.sum(np.square(freqs - centroid) * p)))

    cdf = np.cumsum(p)
    rolloff_idx = int(np.searchsorted(cdf, 0.85))
    rolloff_idx = min(max(rolloff_idx, 0), len(freqs) - 1)
    rolloff_85 = _safe_float(freqs[rolloff_idx])

    flatness = _safe_float(np.exp(np.mean(np.log(psd))) / np.mean(psd))
    entropy = _safe_float(-np.sum(p * np.log(p + EPS)) / np.log(len(p) + EPS))

    dom_idx = int(np.argmax(psd))
    dom_freq = _safe_float(freqs[dom_idx])
    dom_ratio = _safe_float(psd[dom_idx] / power_sum)

    valid = freqs > 0
    if np.count_nonzero(valid) >= 2:
        slope = _safe_float(np.polyfit(np.log(freqs[valid]), np.log(psd[valid]), deg=1)[0])
    else:
        slope = 0.0

    low_mask = (freqs >= 20.0) & (freqs < 150.0)
    high_mask = (freqs >= 150.0) & (freqs < 600.0)
    low_e = float(psd[low_mask].sum()) if np.any(low_mask) else 0.0
    high_e = float(psd[high_mask].sum()) if np.any(high_mask) else 0.0
    low_high_ratio = _safe_float(low_e / (high_e + EPS))

    feats = [
        centroid,
        bandwidth,
        rolloff_85,
        flatness,
        entropy,
        dom_freq,
        dom_ratio,
        slope,
        low_high_ratio,
    ]
    return feats, np.asarray(freqs, dtype=np.float32), np.asarray(psd, dtype=np.float32)


def _band_energy_ratios(freqs: np.ndarray, psd: np.ndarray, bands_hz: Sequence[Tuple[float, float]]) -> List[float]:
    psd64 = psd.astype(np.float64, copy=False)
    total = float(psd64.sum()) + EPS
    feats: List[float] = []
    for lo, hi in bands_hz:
        mask = (freqs >= float(lo)) & (freqs < float(hi))
        band_e = float(psd64[mask].sum()) if np.any(mask) else 0.0
        feats.append(_safe_float(band_e / total))
    return feats


def _describe_series(x: np.ndarray) -> Tuple[float, float, float, float]:
    x64 = x.astype(np.float64, copy=False)
    mean = _safe_float(np.mean(x64))
    std = _safe_float(np.std(x64, ddof=0))
    if np.allclose(std, 0.0):
        return mean, std, 0.0, 0.0
    skew = _safe_float(stats.skew(x64, bias=False, nan_policy="omit"))
    kurt = _safe_float(stats.kurtosis(x64, fisher=True, bias=False, nan_policy="omit"))
    return mean, std, skew, kurt


@dataclass(frozen=True)
class EngineeredAcousticDescriptorConfig:
    sample_rate: int = 8000
    spectral_bands_hz: Tuple[Tuple[float, float], ...] = DEFAULT_BANDS_HZ


class EngineeredAcousticDescriptorExtractor:
    """Extract one fixed-dimensional, segmentation-free descriptor vector from each window."""

    def __init__(self, config: EngineeredAcousticDescriptorConfig | None = None) -> None:
        self.config = config or EngineeredAcousticDescriptorConfig()
        self.feature_names: List[str] = self._build_feature_names()

    @staticmethod
    def _build_feature_names() -> List[str]:
        names = [
            "time_rms",
            "time_abs_mean",
            "time_std",
            "time_peak_abs",
            "time_crest_factor",
            "time_zero_cross_rate",
            "time_shannon_energy",
            "time_log_energy",
            "time_skewness",
            "time_kurtosis",
            "env_mean",
            "env_std",
            "env_cv",
            "env_crest_factor",
            "env_skewness",
            "env_kurtosis",
            "acf_peak_value",
            "acf_peak_lag_sec",
            "acf_decay50_sec",
            "spec_centroid_hz",
            "spec_bandwidth_hz",
            "spec_rolloff85_hz",
            "spec_flatness",
            "spec_entropy",
            "spec_dominant_freq_hz",
            "spec_dominant_power_ratio",
            "spec_loglog_slope",
            "spec_low_high_ratio",
        ]
        for lo, hi in DEFAULT_BANDS_HZ:
            names.append(f"band_ratio_{int(lo)}_{int(hi)}Hz")
        return names

    def n_features(self) -> int:
        return len(self.feature_names)

    def extract_one(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        if int(sr) != int(self.config.sample_rate):
            raise ValueError(
                f"EAD extractor expects sample_rate={self.config.sample_rate}, but got sr={sr}. "
                "Please keep the window library sampling rate fixed."
            )

        x = _validate_waveform(waveform)

        rms = _safe_float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))
        abs_mean = _safe_float(np.mean(np.abs(x), dtype=np.float64))
        std = _safe_float(np.std(x, ddof=0))
        peak_abs = _safe_float(np.max(np.abs(x)))
        crest_factor = _safe_float(peak_abs / (rms + EPS))
        zcr = _zero_crossing_rate(x)
        shannon = _shannon_energy(x)
        log_energy = _log_energy(x)
        _, _, time_skew, time_kurt = _describe_series(x)

        env = _envelope(x)
        env_mean, env_std, env_skew, env_kurt = _describe_series(env)
        env_cv = _safe_float(env_std / (env_mean + EPS))
        env_peak = _safe_float(np.max(env))
        env_crest = _safe_float(env_peak / (env_mean + EPS))

        acf_peak, acf_lag_sec, acf_decay_sec = _acf_features(x, sr=sr)

        spec_feats, freqs, psd = _spectral_features(x, sr=sr)
        band_ratios = _band_energy_ratios(freqs, psd, self.config.spectral_bands_hz)

        feats = [
            rms,
            abs_mean,
            std,
            peak_abs,
            crest_factor,
            zcr,
            shannon,
            log_energy,
            time_skew,
            time_kurt,
            env_mean,
            env_std,
            env_cv,
            env_crest,
            env_skew,
            env_kurt,
            acf_peak,
            acf_lag_sec,
            acf_decay_sec,
            *spec_feats,
            *band_ratios,
        ]
        out = np.asarray(feats, dtype=np.float32)

        if out.shape[0] != len(self.feature_names):
            raise RuntimeError(
                f"EAD feature-length mismatch: got {out.shape[0]} values, "
                f"but feature_names has {len(self.feature_names)} entries."
            )
        if not np.all(np.isfinite(out)):
            raise RuntimeError("EAD extraction produced non-finite values.")
        return out

    def extract_many(self, waveforms: Sequence[np.ndarray], sr: int) -> np.ndarray:
        feats = [self.extract_one(w, sr=sr) for w in waveforms]
        return np.stack(feats, axis=0).astype(np.float32, copy=False)
