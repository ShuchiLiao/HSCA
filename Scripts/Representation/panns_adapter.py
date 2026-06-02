from __future__ import annotations

from pathlib import Path
from typing import List
import sys

import numpy as np
import torch
import torch.nn as nn

from audio_utils import resample_waveform, right_pad_waveforms
from routes import get_route_config


class PANNsAdapter:
    """Adapter for the frozen PANNs route.

    Fixed protocol
    --------------
    - waveform layer: directly use window waveforms
    - resampling layer: resample to 32000 Hz
    - spectrogram layer: use the official PANNs model's internal frontend
    - normalization layer: no extra dataset-level mean/std normalization

    Important note about embedding extraction
    -----------------------------------------
    Official PANNs Cnn14 returns output_dict["embedding"] in forward().
    This embedding is produced with dropout in the network.
    Therefore, when extracting deterministic frozen embeddings, the model
    MUST stay in eval mode. Otherwise repeated extraction on the same input
    may produce slightly different embeddings.
    """

    def __init__(self, checkpoint_path: str, device: str) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.device = torch.device(device)
        self.cfg = get_route_config("panns")

        self.model = self._load_model(self.checkpoint_path)
        self.model.to(self.device)

        # Critical for frozen embedding extraction:
        # official PANNs "embedding" goes through dropout, so eval() is required.
        self.model.eval()

    def _import_local_official_cnn14(self):
        """Import vendored official PANNs Cnn14 from ./panns/pytorch/models.py.

        Expected layout
        ---------------
        Scripts/Representation/
        ├── panns_adapter.py
        └── panns/
            └── pytorch/
                ├── models.py
                └── pytorch_utils.py
        """
        this_file = Path(__file__).resolve()
        project_root = this_file.parent
        panns_pytorch_dir = project_root / "panns" / "pytorch"

        if not panns_pytorch_dir.exists():
            raise FileNotFoundError(
                f"Cannot find local PANNs pytorch directory: {panns_pytorch_dir}"
            )

        # Make vendored official repo importable.
        # We add the exact directory containing models.py and pytorch_utils.py,
        # so that:
        #   from models import Cnn14
        # and models.py -> from pytorch_utils import ...
        # both work reliably.
        panns_pytorch_dir_str = str(panns_pytorch_dir)
        if panns_pytorch_dir_str not in sys.path:
            sys.path.insert(0, panns_pytorch_dir_str)

        try:
            from models import Cnn14  # type: ignore
        except Exception as e:
            raise ImportError(
                "Failed to import local official PANNs Cnn14 from "
                f"{panns_pytorch_dir}. "
                "Please check that models.py and pytorch_utils.py are both present "
                "and that official PANNs dependencies are installed."
            ) from e

        return Cnn14

    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load a PANNs Cnn14 model from checkpoint without using panns_inference."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"PANNs checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(path, map_location="cpu")

        # Case 1: checkpoint already stores a full nn.Module.
        if isinstance(ckpt, nn.Module):
            return ckpt

        # Case 2: checkpoint stores a full model object under key "model".
        if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), nn.Module):
            return ckpt["model"]

        # Case 3: checkpoint is a state dict. Build local official Cnn14.
        Cnn14 = self._import_local_official_cnn14()

        model = Cnn14(
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527,
        )

        state_dict = ckpt
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state_dict = ckpt[key]
                    break

        if not isinstance(state_dict, dict):
            raise ValueError(
                "Unsupported PANNs checkpoint format. "
                "Expected a full model or a state dict."
            )

        cleaned_state = {}
        for k, v in state_dict.items():
            new_key = str(k)
            if new_key.startswith("module."):
                new_key = new_key[len("module.") :]
            cleaned_state[new_key] = v

        missing, unexpected = model.load_state_dict(cleaned_state, strict=False)

        if len(unexpected) > 0:
            print(f"[PANNsAdapter] Unexpected checkpoint keys: {unexpected}")
        if len(missing) > 0:
            print(f"[PANNsAdapter] Missing checkpoint keys: {missing}")

        return model

    def _prepare_batch(self, waveforms: List[np.ndarray], src_sr: int) -> torch.Tensor:
        """Resample waveform list to 32 kHz and right-pad into one batch tensor."""
        if src_sr <= 0:
            raise ValueError(f"src_sr must be positive, got {src_sr}")
        if len(waveforms) == 0:
            raise ValueError("waveforms must be a non-empty list")

        xs_32k = [
            resample_waveform(w, orig_sr=src_sr, target_sr=self.cfg.target_sr)
            for w in waveforms
        ]
        padded, _ = right_pad_waveforms(xs_32k, pad_value=0.0)
        return torch.from_numpy(padded).to(self.device)

    def _forward_model(self, batch_waveforms: torch.Tensor) -> torch.Tensor:
        """Run the model and extract one embedding vector per waveform."""
        # Safety guard: embedding goes through dropout in training mode,
        # so deterministic feature extraction must always use eval mode.
        self.model.eval()

        with torch.no_grad():
            output = self.model(batch_waveforms)

        if isinstance(output, dict):
            if "embedding" in output:
                emb = output["embedding"]
            elif "clipwise_output" in output:
                raise ValueError(
                    "PANNs output contains 'clipwise_output' but not 'embedding'. "
                    "This checkpoint/model path is not suitable for the current "
                    "frozen-embedding protocol."
                )
            else:
                raise ValueError(
                    f"PANNs output dict does not contain 'embedding'. "
                    f"Available keys: {list(output.keys())}"
                )
        elif torch.is_tensor(output):
            raise ValueError(
                "PANNs forward returned a raw tensor instead of an output dict. "
                "Please verify whether this tensor is truly the penultimate embedding."
            )
        else:
            raise ValueError(f"Unsupported PANNs output type: {type(output)}")

        if emb.ndim == 3:
            emb = emb.mean(dim=1)

        if emb.ndim != 2:
            raise ValueError(f"Expected PANNs embedding shape [B, D], got {tuple(emb.shape)}")

        return emb

    def extract_embeddings(self, waveforms: List[np.ndarray], src_sr: int) -> np.ndarray:
        """Extract one embedding vector for each waveform."""
        batch_waveforms = self._prepare_batch(waveforms, src_sr)
        emb = self._forward_model(batch_waveforms)
        return emb.detach().cpu().numpy().astype(np.float32, copy=False)