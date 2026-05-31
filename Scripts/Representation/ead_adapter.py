from __future__ import annotations

"""
Adapter for segmentation-free engineered acoustic descriptors (EAD).

The rest of the representation-learning codebase expects each route to expose a
unified adapter interface:

    extract_embeddings(waveforms: List[np.ndarray], src_sr: int) -> np.ndarray

This file wraps EAD extraction into that same interface.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np

from audio_utils import resample_waveform
from ead_features import (
    EngineeredAcousticDescriptorConfig,
    EngineeredAcousticDescriptorExtractor,
)


class EngineeredAcousticDescriptorAdapter:
    """
    Adapter for segmentation-free engineered acoustic descriptors (EAD).

    Notes
    -----
    - CPU-first and deterministic.
    - Returns one descriptor vector per input window.
    - The method name `extract_embeddings()` is kept for compatibility with the
      existing route extraction pipeline.
    """

    def __init__(
        self,
        device: str = "cpu",
        target_sr: int = 8000,
        n_jobs: int = 1,
        verbose: bool = True,
    ) -> None:
        self.device = str(device)  # kept only for API compatibility
        self.n_jobs = max(int(n_jobs), 1)
        self.verbose = bool(verbose)

        self.config = EngineeredAcousticDescriptorConfig(sample_rate=int(target_sr))
        self.extractor = EngineeredAcousticDescriptorExtractor(config=self.config)
        self.feature_names = self.extractor.feature_names
        self._logged_first_batch = False

        if self.verbose:
            print(
                "[info] Initialized EAD adapter | "
                f"target_sr={self.config.sample_rate}, "
                f"n_features={len(self.feature_names)}, "
                f"n_jobs={self.n_jobs}, device={self.device}"
            )

    def _prepare_waveform(self, waveform: np.ndarray, src_sr: int) -> np.ndarray:
        x = np.asarray(waveform, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError(f"Each waveform must be 1D, but got shape={x.shape}")

        if int(src_sr) != int(self.config.sample_rate):
            if self.verbose:
                print(
                    f"[warn] EAD adapter received src_sr={src_sr}; "
                    f"resampling to target_sr={self.config.sample_rate}."
                )
            x = resample_waveform(x, orig_sr=int(src_sr), target_sr=int(self.config.sample_rate))
        return x

    def _extract_one(self, waveform: np.ndarray, src_sr: int) -> np.ndarray:
        x = self._prepare_waveform(waveform, src_sr=src_sr)
        return self.extractor.extract_one(x, sr=self.config.sample_rate)

    def extract_embeddings(self, waveforms: List[np.ndarray], src_sr: int) -> np.ndarray:
        if len(waveforms) == 0:
            raise ValueError("waveforms must be a non-empty list")
        if int(src_sr) <= 0:
            raise ValueError(f"src_sr must be positive, but got {src_sr}")

        if self.verbose and not self._logged_first_batch:
            print(
                "[info] EAD first extraction batch | "
                f"batch_size={len(waveforms)}, src_sr={src_sr}, n_jobs={self.n_jobs}"
            )
            self._logged_first_batch = True

        if self.n_jobs == 1 or len(waveforms) == 1:
            feats = [self._extract_one(w, src_sr=int(src_sr)) for w in waveforms]
        else:
            max_workers = min(self.n_jobs, len(waveforms))
            if self.verbose:
                print(f"[info] EAD parallel extraction with {max_workers} worker(s).")
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                feats = list(ex.map(lambda w: self._extract_one(w, src_sr=int(src_sr)), waveforms))

        out = np.stack(feats, axis=0).astype(np.float32, copy=False)
        if out.ndim != 2:
            raise RuntimeError(f"EAD adapter must return [B, D], but got shape={out.shape}")
        if not np.all(np.isfinite(out)):
            raise RuntimeError("EAD adapter produced non-finite descriptor values.")
        return out
