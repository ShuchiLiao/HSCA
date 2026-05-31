from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

import numpy as np
import torch
import torch.nn as nn

from audio_utils import resample_waveform, right_pad_waveforms
from routes import get_route_config


class BEATsAdapter:
    """Adapter shared by frozen BEATs and BEATs domain adaptation.

    Fixed protocol
    --------------
    - waveform layer: directly use window waveforms
    - resampling layer: resample to 16000 Hz
    - spectrogram layer: use the official BEATs frontend
    - normalization layer: use the official BEATs defaults

    Design note
    -----------
    This adapter resolves the local vendored BEATs repo by itself, so the
    project entry script (extract_embeddings.py) does not need to modify
    sys.path or care where the official repo lives.
    """

    def __init__(self, checkpoint_path: str, device: str, trainable: bool = False) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.device = torch.device(device)
        self.trainable = bool(trainable)
        self.cfg = get_route_config("beats")

        self.model = self._load_model(self.checkpoint_path)
        self.model.to(self.device)

        # Frozen extraction should be eval(); domain adaptation can stay train().
        self.model.train(self.trainable)
        if not self.trainable:
            self.model.eval()

    def _import_local_official_beats(self):
        """Import vendored official BEATs implementation from ./beats/.

        Expected layout
        ---------------
        representation_learning/
        ├── beats_adapter.py
        └── beats/
            ├── BEATs.py
            ├── BEATsconfig.py
            └── ...

        We add the local ./beats directory itself to sys.path, so imports inside
        the official repo that rely on sibling modules can continue to work.
        """
        this_file = Path(__file__).resolve()
        project_root = this_file.parent
        beats_dir = project_root / "beats"

        if not beats_dir.exists():
            raise FileNotFoundError(f"Cannot find local BEATs repo directory: {beats_dir}")

        beats_dir_str = str(beats_dir)
        if beats_dir_str not in sys.path:
            sys.path.insert(0, beats_dir_str)

        # Try the most common official layout first: BEATs.py defines both classes.
        try:
            from BEATs import BEATs, BEATsConfig  # type: ignore
            return BEATs, BEATsConfig
        except Exception:
            pass

        # Fallback: some local copies may separate config into BEATsconfig.py.
        try:
            from BEATs import BEATs  # type: ignore
            from BEATsconfig import BEATsConfig  # type: ignore
            return BEATs, BEATsConfig
        except Exception as e:
            raise ImportError(
                "Failed to import local official BEATs implementation from "
                f"{beats_dir}. Please check that BEATs.py is present, and if "
                "your local copy uses a separate config file, that "
                "BEATsconfig.py is also present. Also ensure all official "
                "BEATs dependencies are installed."
            ) from e

    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load a BEATs model checkpoint.

        Supported checkpoint formats
        ----------------------------
        - a full torch nn.Module
        - a dict containing a full nn.Module under key 'model'
        - an official BEATs checkpoint containing 'cfg' and 'model'
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"BEATs checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(path, map_location="cpu")

        # Case 1: checkpoint already stores a full nn.Module.
        if isinstance(ckpt, nn.Module):
            return ckpt

        # Case 2: checkpoint stores a full model object under key 'model'.
        if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), nn.Module):
            return ckpt["model"]

        # Case 3: official BEATs checkpoint: {'cfg': ..., 'model': state_dict, ...}
        if isinstance(ckpt, dict) and "cfg" in ckpt and "model" in ckpt and isinstance(ckpt["model"], dict):
            BEATs, BEATsConfig = self._import_local_official_beats()
            beats_cfg = BEATsConfig(ckpt["cfg"])
            model = BEATs(beats_cfg)
            model.load_state_dict(ckpt["model"], strict=True)
            return model

        raise ValueError(
            "Unsupported BEATs checkpoint format. "
            "Expected a full torch model or an official checkpoint containing "
            "'cfg' and 'model'."
        )

    def _prepare_batch(self, waveforms: List[np.ndarray], src_sr: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resample to 16 kHz and create a padding mask.

        Returns
        -------
        batch_waveforms:
            Tensor with shape [B, L_max].
        padding_mask:
            Bool tensor with shape [B, L_max], where True marks padded positions.
        """
        if src_sr <= 0:
            raise ValueError(f"src_sr must be positive, got {src_sr}")
        if len(waveforms) == 0:
            raise ValueError("waveforms must be a non-empty list")

        xs_16k = [resample_waveform(w, orig_sr=src_sr, target_sr=self.cfg.target_sr) for w in waveforms]
        padded, lengths = right_pad_waveforms(xs_16k, pad_value=0.0)

        batch = torch.from_numpy(padded).to(self.device)
        lengths_t = torch.from_numpy(lengths).to(self.device)

        max_len = int(batch.shape[1])
        index = torch.arange(max_len, device=self.device).unsqueeze(0)
        padding_mask = index >= lengths_t.unsqueeze(1)
        return batch, padding_mask

    def _extract_embedding_from_features(self, features: Any) -> torch.Tensor:
        """Convert BEATs returned features to one [B, D] embedding tensor."""
        if torch.is_tensor(features):
            x = features
        elif isinstance(features, (tuple, list)) and len(features) > 0 and torch.is_tensor(features[0]):
            x = features[0]
        elif isinstance(features, dict):
            for key in ("embedding", "features", "x", "last_hidden_state"):
                if key in features and torch.is_tensor(features[key]):
                    x = features[key]
                    break
            else:
                raise ValueError(
                    "Could not find a tensor-like BEATs feature in the returned dict. "
                    f"Available keys: {list(features.keys())}"
                )
        else:
            raise ValueError(f"Unsupported BEATs feature type: {type(features)}")

        if x.ndim == 3:
            x = x.mean(dim=1)

        if x.ndim != 2:
            raise ValueError(f"Expected BEATs embedding shape [B, D], got {tuple(x.shape)}")

        return x

    def extract_embeddings(self, waveforms: List[np.ndarray], src_sr: int) -> np.ndarray:
        """Extract one embedding vector for each waveform.

        For frozen embedding extraction, keep the model in eval mode and disable
        gradients. This yields deterministic features under the frozen route.
        """
        batch, padding_mask = self._prepare_batch(waveforms, src_sr)

        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, "extract_features"):
                features = self.model.extract_features(batch, padding_mask=padding_mask)
            else:
                features = self.model(batch, padding_mask=padding_mask)

        emb = self._extract_embedding_from_features(features)
        return emb.detach().cpu().numpy().astype(np.float32, copy=False)

    def forward_for_ssl(self, waveforms: List[np.ndarray], src_sr: int) -> Dict[str, Any]:
        """Prepare a trainable forward pass for BEATs domain adaptation.

        Returns a dict so the later SSL training script can reuse the batch
        waveform tensor, padding mask, and the raw model output.
        """
        batch, padding_mask = self._prepare_batch(waveforms, src_sr)

        if self.trainable:
            self.model.train()
        else:
            self.model.eval()

        if hasattr(self.model, "extract_features"):
            model_output = self.model.extract_features(batch, padding_mask=padding_mask)
        else:
            model_output = self.model(batch, padding_mask=padding_mask)

        return {
            "waveforms": batch,
            "padding_mask": padding_mask,
            "model_output": model_output,
        }