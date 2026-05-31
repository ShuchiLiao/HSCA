from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import resample_poly


# librosa is only used for log-Mel feature extraction.
# Keep the import local and explicit so this file stays lightweight.
try:
    import librosa
except Exception:  # pragma: no cover - handled at runtime if librosa is unavailable
    librosa = None


EPS = 1e-8


def resample_waveform(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample one 1D waveform to the target sample rate.

    Parameters
    ----------
    waveform:
        Input waveform with shape [T].
    orig_sr:
        Original sample rate.
    target_sr:
        Target sample rate.

    Returns
    -------
    np.ndarray
        Resampled waveform, shape [T_new], dtype float32.
    """
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"waveform must be 1D, but got shape={x.shape}")
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"orig_sr and target_sr must be positive, got {orig_sr}, {target_sr}")

    if orig_sr == target_sr:
        return x.astype(np.float32, copy=False)

    # Use gcd-reduced polyphase resampling for speed and good numerical behavior.
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    y = resample_poly(x, up=up, down=down)
    return np.asarray(y, dtype=np.float32)


def right_pad_waveforms(
    waveforms: List[np.ndarray],
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Right-pad a list of variable-length 1D waveforms into one batch.

    Parameters
    ----------
    waveforms:
        A list of 1D waveform arrays.
    pad_value:
        Value used for right padding.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        padded_waveforms: [B, L_max], float32
        lengths: [B], int64
    """
    if len(waveforms) == 0:
        raise ValueError("waveforms must be a non-empty list")

    xs = [np.asarray(w, dtype=np.float32) for w in waveforms]
    for i, x in enumerate(xs):
        if x.ndim != 1:
            raise ValueError(f"waveforms[{i}] must be 1D, but got shape={x.shape}")

    lengths = np.asarray([x.shape[0] for x in xs], dtype=np.int64)
    max_len = int(lengths.max())
    out = np.full((len(xs), max_len), pad_value, dtype=np.float32)

    for i, x in enumerate(xs):
        out[i, : x.shape[0]] = x

    return out, lengths


def waveforms_to_logmel(
    waveforms: np.ndarray,
    sample_rate: int,
    n_mels: int,
    frame_ms: float,
    hop_ms: float,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Convert a batch of waveforms [B, L] into log-Mel features [B, T, F].

    Notes
    -----
    - This function intentionally keeps the output convention fixed to [B, T, F].
    - AST can consume this layout directly.
    - BYOL-A can transpose later inside its own adapter if it needs [B, 1, F, T].
    """
    if librosa is None:
        raise ImportError(
            "librosa is required for waveforms_to_logmel(), but it is not available."
        )

    x = np.asarray(waveforms, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"waveforms must have shape [B, L], but got {x.shape}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if n_mels <= 0:
        raise ValueError(f"n_mels must be positive, got {n_mels}")

    n_fft = int(round(sample_rate * frame_ms / 1000.0))
    hop_length = int(round(sample_rate * hop_ms / 1000.0))
    win_length = n_fft
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError(
            f"Invalid frame/hop setting: frame_ms={frame_ms}, hop_ms={hop_ms}, sample_rate={sample_rate}"
        )

    features = []
    for i in range(x.shape[0]):
        mel = librosa.feature.melspectrogram(
            y=x[i],
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
            center=True,
            power=2.0,
            n_mels=n_mels,
            fmin=float(fmin),
            fmax=float(fmax),
        )
        logmel = np.log(np.maximum(mel, EPS)).T  # [T, F]
        features.append(np.asarray(logmel, dtype=np.float32))

    # If all inputs were right-padded to the same length, T will also match.
    time_lengths = {feat.shape[0] for feat in features}
    if len(time_lengths) != 1:
        raise ValueError(
            "All waveforms in a batch must have the same padded length before log-Mel extraction. "
            f"Got time lengths: {sorted(time_lengths)}"
        )

    return np.stack(features, axis=0).astype(np.float32, copy=False)  # [B, T, F]


def apply_global_scalar_norm(
    features: np.ndarray,
    mean: float,
    std: float,
    divide_by_2std: bool = False,
) -> np.ndarray:
    """Apply scalar global normalization to features [B, T, F].

    Parameters
    ----------
    features:
        Input feature batch with shape [B, T, F].
    mean, std:
        Scalar mean/std computed on the training set.
    divide_by_2std:
        If True, use (x - mean) / (2 * std), which matches AST-style usage.
        If False, use standard (x - mean) / std.
    """
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"features must have shape [B, T, F], but got {x.shape}")

    denom = float(std)
    if divide_by_2std:
        denom *= 2.0
    denom = max(denom, EPS)

    y = (x - float(mean)) / denom
    return np.asarray(y, dtype=np.float32)


def load_stats_json(stats_path: str) -> Dict[str, float]:
    """Load a stats json file that must contain at least mean and std."""
    path = Path(stats_path)
    with path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    if not isinstance(stats, dict):
        raise ValueError(f"Stats json must contain a dict, but got type={type(stats)}")
    if "mean" not in stats or "std" not in stats:
        raise ValueError(f"Stats json must contain at least 'mean' and 'std': {stats_path}")

    return {
        **stats,
        "mean": float(stats["mean"]),
        "std": float(stats["std"]),
    }
