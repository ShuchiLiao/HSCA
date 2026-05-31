from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from audio_utils import (
    apply_global_scalar_norm,
    load_stats_json,
    resample_waveform,
    right_pad_waveforms,
    waveforms_to_logmel,
)
from routes import get_route_config


class BYOLAAdapter:
    """Adapter shared by BYOL-A training and embedding extraction.

    Fixed protocol
    --------------
    - waveform layer: directly use window waveforms
    - resampling layer: resample to 16000 Hz
    - spectrogram layer: build log-Mel using the project-level common utility
    - normalization layer: read byola_stats.json and use (x - mean) / std

    Notes
    -----
    - `waveforms_to_features()` returns [B, T, F].
    - `make_ssl_views()` and the internal model input layout convert features to
      [B, 1, F, T], which is a common BYOL-A-friendly layout.
    """

    def __init__(self, checkpoint_path: Optional[str], stats_path: str, device: str) -> None:
        self.checkpoint_path = checkpoint_path if checkpoint_path is None else str(checkpoint_path)
        self.stats_path = str(stats_path)
        self.device = torch.device(device)
        self.cfg = get_route_config("byola")
        self.stats = load_stats_json(self.stats_path)
        self.model = self._load_model(self.checkpoint_path) if self.checkpoint_path is not None else None
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load a BYOL-A encoder checkpoint.

        Expected formats
        ----------------
        - a full torch nn.Module
        - a dict containing a full model under key 'model'
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"BYOL-A checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, nn.Module):
            return ckpt
        if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), nn.Module):
            return ckpt["model"]

        raise ValueError(
            "Unsupported BYOL-A checkpoint format. "
            "Please provide a checkpoint that stores a full torch model object."
        )

    def waveforms_to_features(self, waveforms: List[np.ndarray], src_sr: int) -> np.ndarray:
        """Convert a waveform list into normalized log-Mel features [B, T, F]."""
        xs_16k = [resample_waveform(w, orig_sr=src_sr, target_sr=self.cfg.target_sr) for w in waveforms]
        padded, _ = right_pad_waveforms(xs_16k, pad_value=0.0)

        logmel = waveforms_to_logmel(
            waveforms=padded,
            sample_rate=self.cfg.target_sr,
            n_mels=int(self.cfg.n_mels),
            frame_ms=float(self.cfg.frame_ms),
            hop_ms=float(self.cfg.hop_ms),
            fmin=float(self.cfg.fmin),
            fmax=float(self.cfg.fmax),
        )
        logmel = apply_global_scalar_norm(
            features=logmel,
            mean=float(self.stats["mean"]),
            std=float(self.stats["std"]),
            divide_by_2std=False,
        )
        return logmel.astype(np.float32, copy=False)

    def _features_to_model_input(self, features_btf: np.ndarray) -> torch.Tensor:
        """Convert [B, T, F] -> [B, 1, F, T] for BYOL-A style models."""
        if features_btf.ndim != 3:
            raise ValueError(f"Expected features with shape [B, T, F], got {features_btf.shape}")
        x = np.transpose(features_btf, (0, 2, 1))[:, None, :, :]
        return torch.from_numpy(x.astype(np.float32, copy=False)).to(self.device)

    def _extract_embedding_from_output(self, output: Any) -> torch.Tensor:
        """Normalize different BYOL-A outputs into one [B, D] embedding tensor."""
        if torch.is_tensor(output):
            x = output
        elif isinstance(output, dict):
            for key in ("embedding", "features", "projection", "x", "last_hidden_state"):
                if key in output and torch.is_tensor(output[key]):
                    x = output[key]
                    break
            else:
                raise ValueError("Could not find a tensor-like embedding in BYOL-A output dict.")
        elif isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
            x = output[0]
        else:
            raise ValueError(f"Unsupported BYOL-A output type: {type(output)}")

        if x.ndim == 4:
            x = x.mean(dim=(-1, -2))
        elif x.ndim == 3:
            x = x.mean(dim=1)
        if x.ndim != 2:
            raise ValueError(f"Expected BYOL-A embedding shape [B, D], got {tuple(x.shape)}")
        return x

    def _augment_features(self, x: torch.Tensor) -> torch.Tensor:
        """Create one simple SSL view in feature space.

        Input
        -----
        x: [B, 1, F, T]

        Augmentations
        -------------
        - small additive Gaussian noise
        - random time masking
        - random frequency masking
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [B, 1, F, T], got {tuple(x.shape)}")

        y = x.clone()

        # Small feature-space Gaussian noise.
        noise = 0.01 * torch.randn_like(y)
        y = y + noise

        bsz, _, n_freq, n_time = y.shape

        # Random time masking.
        max_t = max(1, n_time // 10)
        for i in range(bsz):
            width = int(torch.randint(low=0, high=max_t + 1, size=(1,), device=y.device).item())
            if width > 0 and width < n_time:
                start = int(torch.randint(low=0, high=n_time - width + 1, size=(1,), device=y.device).item())
                y[i, :, :, start : start + width] = 0.0

        # Random frequency masking.
        max_f = max(1, n_freq // 10)
        for i in range(bsz):
            width = int(torch.randint(low=0, high=max_f + 1, size=(1,), device=y.device).item())
            if width > 0 and width < n_freq:
                start = int(torch.randint(low=0, high=n_freq - width + 1, size=(1,), device=y.device).item())
                y[i, :, start : start + width, :] = 0.0

        return y

    def extract_embeddings(self, waveforms: List[np.ndarray], src_sr: int) -> np.ndarray:
        """Extract one embedding vector for each waveform."""
        if self.model is None:
            raise ValueError("checkpoint_path is None, so embedding extraction is unavailable.")

        features_btf = self.waveforms_to_features(waveforms, src_sr)
        x = self._features_to_model_input(features_btf)

        with torch.no_grad():
            if hasattr(self.model, "forward_features"):
                output = self.model.forward_features(x)
            else:
                output = self.model(x)

        emb = self._extract_embedding_from_output(output)
        return emb.detach().cpu().numpy().astype(np.float32, copy=False)

    def make_ssl_views(self, waveforms: List[np.ndarray], src_sr: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create two BYOL-A feature views for SSL training.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Two tensors with shape [B, 1, F, T].
        """
        features_btf = self.waveforms_to_features(waveforms, src_sr)
        x = self._features_to_model_input(features_btf)
        view1 = self._augment_features(x)
        view2 = self._augment_features(x)
        return view1, view2
