from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class ASTAdapter:
    """Adapter for the frozen AST route.

    Fixed protocol
    --------------
    - waveform layer: directly use window waveforms
    - resampling layer: resample to 16000 Hz
    - spectrogram layer: 128-bin log-Mel / fbank with 25 ms frame, 10 ms hop
    - normalization layer: read global stats and use (x - mean) / (2 * std)
    """

    def __init__(self, checkpoint_path: str, stats_path: str, device: str) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.stats_path = str(stats_path)
        self.device = torch.device(device)
        self.cfg = get_route_config("ast")
        self.stats = load_stats_json(self.stats_path)
        self.model: Optional[nn.Module] = None
        self._model_input_tdim: Optional[int] = None

    def _load_model(self, checkpoint_path: str, input_tdim: int) -> nn.Module:
        """Load official ASTModel from ast/src/models/ast_models.py."""
        path = Path(checkpoint_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"AST checkpoint not found: {checkpoint_path}")

        # Still support full-model checkpoints.
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, nn.Module):
            return ckpt
        if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), nn.Module):
            return ckpt["model"]

        # Official AST repo path: Representation_learning/ast
        this_file = Path(__file__).resolve()
        project_root = this_file.parent
        ast_root = project_root / "ast"
        ast_src = ast_root / "src"
        ast_models_dir = ast_src / "models"

        if not ast_src.exists():
            raise FileNotFoundError(f"Cannot find AST official repo src directory: {ast_src}")

        if str(ast_src) not in sys.path:
            sys.path.insert(0, str(ast_src))

        # The official AST code looks for ../../pretrained_models/audioset_10_10_0.4593.pth
        # relative to ast/src/models if we temporarily chdir there.
        pretrained_dir = ast_root / "pretrained_models"
        pretrained_dir.mkdir(parents=True, exist_ok=True)
        expected_ckpt = pretrained_dir / "audioset_10_10_0.4593.pth"

        if path != expected_ckpt.resolve():
            shutil.copyfile(path, expected_ckpt)

        cwd = os.getcwd()
        try:
            os.chdir(ast_models_dir)

            from models.ast_models import ASTModel  # type: ignore

            model = ASTModel(
                label_dim=527,
                fstride=10,
                tstride=10,
                input_fdim=128,
                input_tdim=int(input_tdim),
                imagenet_pretrain=True,
                audioset_pretrain=True,
                model_size="base384",
                verbose=False,
            )
        finally:
            os.chdir(cwd)

        return model

    def _ensure_model(self, input_tdim: int) -> None:
        """Lazy-load AST after input_tdim is known."""
        if self.model is None:
            self.model = self._load_model(self.checkpoint_path, input_tdim=input_tdim)
            self.model.to(self.device)
            self.model.eval()
            self._model_input_tdim = int(input_tdim)
            return

        if self._model_input_tdim != int(input_tdim):
            raise ValueError(
                f"AST model was initialized with input_tdim={self._model_input_tdim}, "
                f"but current batch has input_tdim={input_tdim}. "
                "Please keep window length fixed."
            )

    def _waveforms_to_ast_input(self, waveforms: List[np.ndarray], src_sr: int) -> torch.Tensor:
        """Convert waveform list into AST input tensor with shape [B, T, 128]."""
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
            divide_by_2std=True,
        )
        return torch.from_numpy(logmel).to(self.device)

    def _extract_embedding_from_output(self, output: Any) -> torch.Tensor:
        """Normalize different AST outputs into one [B, D] embedding tensor."""
        if torch.is_tensor(output):
            x = output
        elif isinstance(output, dict):
            for key in ("embedding", "features", "x", "last_hidden_state", "logits"):
                if key in output and torch.is_tensor(output[key]):
                    x = output[key]
                    break
            else:
                raise ValueError("Could not find a tensor-like embedding in AST output dict.")
        elif isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
            x = output[0]
        else:
            raise ValueError(f"Unsupported AST output type: {type(output)}")

        if x.ndim == 3:
            # Common transformer case: [B, N_tokens, D]. Use the first token if available.
            x = x[:, 0, :]
        if x.ndim != 2:
            raise ValueError(f"Expected AST embedding shape [B, D], got {tuple(x.shape)}")
        return x

    def _forward_official_ast_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Forward official ASTModel and return the 768-d embedding before mlp_head.

        Input x shape: [B, T, 128].
        Output shape: [B, 768].
        """
        if self.model is None:
            raise RuntimeError("AST model is not initialized.")

        m = self.model

        x = x.unsqueeze(1)  # [B, 1, T, F]
        x = x.transpose(2, 3)  # [B, 1, F, T]

        B = x.shape[0]
        x = m.v.patch_embed(x)

        cls_tokens = m.v.cls_token.expand(B, -1, -1)
        dist_token = m.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + m.v.pos_embed
        x = m.v.pos_drop(x)

        for blk in m.v.blocks:
            x = blk(x)

        x = m.v.norm(x)

        # This is the AST representation before classification head.
        emb = (x[:, 0] + x[:, 1]) / 2
        return emb


    def extract_embeddings(self, waveforms: List[np.ndarray], src_sr: int) -> np.ndarray:
        """Extract one embedding vector for each waveform."""
        x = self._waveforms_to_ast_input(waveforms, src_sr)

        self._ensure_model(input_tdim=int(x.shape[1]))

        with torch.no_grad():
            if self.model is not None and hasattr(self.model, "v") and hasattr(self.model, "mlp_head"):
                emb = self._forward_official_ast_embedding(x)
            elif self.model is not None and hasattr(self.model, "forward_features"):
                output = self.model.forward_features(x)
                emb = self._extract_embedding_from_output(output)
            else:
                output = self.model(x)  # type: ignore[operator]
                emb = self._extract_embedding_from_output(output)

        return emb.detach().cpu().numpy().astype(np.float32, copy=False)
