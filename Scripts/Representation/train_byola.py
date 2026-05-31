from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from byola_adapter import BYOLAAdapter
from window_dataset import WindowDataset, load_patient_ids


class SimpleBYOLAEncoder(nn.Module):
    """A compact encoder that accepts [B, 1, F, T] and returns [B, D].

    This is intentionally small and generic so the saved ``best.pt`` can later be
    loaded directly by ``BYOLAAdapter.extract_embeddings()``.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        z = self.proj(h)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class BYOLALearner(nn.Module):
    """A minimal BYOL-style learner built around a small encoder."""

    def __init__(self, embed_dim: int = 256, ema_decay: float = 0.99) -> None:
        super().__init__()
        self.online_encoder = SimpleBYOLAEncoder(embed_dim=embed_dim)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ema_decay = float(ema_decay)

    @torch.no_grad()
    def update_target(self) -> None:
        for p_online, p_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(self.ema_decay).add_(p_online.data, alpha=1.0 - self.ema_decay)

    def forward_online(self, x: torch.Tensor) -> torch.Tensor:
        z = self.online_encoder.forward_features(x)
        p = self.predictor(z)
        return p

    @torch.no_grad()
    def forward_target(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_encoder.forward_features(x)


SRC_SR = 8000


def _collate_waveform_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "patient_id": [item["patient_id"] for item in batch],
        "position": [item["position"] for item in batch],
        "window_id": [item["window_id"] for item in batch],
        "waveforms": [item["waveform"] for item in batch],
        "src_sr": int(batch[0]["src_sr"]),
    }


def build_train_val_datasets(
    window_index_csv: str,
    recording_manifest_csv: str,
    train_patient_list: str,
    val_patient_list: str,
) -> Tuple[WindowDataset, WindowDataset]:
    train_ids = load_patient_ids(train_patient_list)
    val_ids = load_patient_ids(val_patient_list)

    train_dataset = WindowDataset(
        window_index_csv=window_index_csv,
        recording_manifest_csv=recording_manifest_csv,
        patient_ids=train_ids,
    )
    val_dataset = WindowDataset(
        window_index_csv=window_index_csv,
        recording_manifest_csv=recording_manifest_csv,
        patient_ids=val_ids,
    )
    return train_dataset, val_dataset


def _negative_cosine(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    p = F.normalize(p, p=2, dim=1)
    z = F.normalize(z.detach(), p=2, dim=1)
    return 1.0 - (p * z).sum(dim=1).mean()


def train_one_epoch(model, adapter, loader, optimizer, device) -> Dict[str, float]:
    """Train one epoch of BYOL-A using adapter-provided SSL views."""
    del device

    model.train()
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        waveforms = batch["waveforms"]
        src_sr = int(batch["src_sr"])
        view1, view2 = adapter.make_ssl_views(waveforms, src_sr=src_sr)

        p1 = model.forward_online(view1)
        p2 = model.forward_online(view2)
        with torch.no_grad():
            z1 = model.forward_target(view1)
            z2 = model.forward_target(view2)

        loss = 0.5 * (_negative_cosine(p1, z2) + _negative_cosine(p2, z1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.update_target()

        bsz = int(view1.shape[0])
        total_loss += float(loss.detach().item()) * bsz
        total_items += bsz

    mean_loss = total_loss / max(total_items, 1)
    return {"loss": float(mean_loss)}


@torch.no_grad()
def validate_one_epoch(model, adapter, loader, device) -> Dict[str, float]:
    """Validate one epoch of BYOL-A."""
    del device

    model.eval()
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        waveforms = batch["waveforms"]
        src_sr = int(batch["src_sr"])
        view1, view2 = adapter.make_ssl_views(waveforms, src_sr=src_sr)

        p1 = model.forward_online(view1)
        p2 = model.forward_online(view2)
        z1 = model.forward_target(view1)
        z2 = model.forward_target(view2)

        loss = 0.5 * (_negative_cosine(p1, z2) + _negative_cosine(p2, z1))

        bsz = int(view1.shape[0])
        total_loss += float(loss.detach().item()) * bsz
        total_items += bsz

    mean_loss = total_loss / max(total_items, 1)
    return {"loss": float(mean_loss)}


def save_checkpoint(state: dict, ckpt_path: str) -> None:
    path = Path(ckpt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train BYOL-A on an unlabeled heart-sound window library."
    )
    parser.add_argument("--window-index-csv", type=str, required=True)
    parser.add_argument("--recording-manifest-csv", type=str, required=True)
    parser.add_argument("--train-patient-list", type=str, required=True)
    parser.add_argument("--val-patient-list", type=str, required=True)
    parser.add_argument("--stats-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset = build_train_val_datasets(
        window_index_csv=args.window_index_csv,
        recording_manifest_csv=args.recording_manifest_csv,
        train_patient_list=args.train_patient_list,
        val_patient_list=args.val_patient_list,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=_collate_waveform_batch,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=_collate_waveform_batch,
        drop_last=False,
    )

    adapter = BYOLAAdapter(
        checkpoint_path=None,
        stats_path=args.stats_path,
        device=args.device,
    )
    model = BYOLALearner(embed_dim=int(args.embed_dim), ema_decay=float(args.ema_decay)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    history: List[Dict[str, float]] = []
    best_val = float("inf")

    train_config = {
        "window_index_csv": args.window_index_csv,
        "recording_manifest_csv": args.recording_manifest_csv,
        "train_patient_list": args.train_patient_list,
        "val_patient_list": args.val_patient_list,
        "stats_path": args.stats_path,
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "device": args.device,
        "embed_dim": int(args.embed_dim),
        "ema_decay": float(args.ema_decay),
        "src_sr": SRC_SR,
        "objective": "minimal BYOL-style view prediction on log-Mel SSL views",
    }
    (out_dir / "train_config.json").write_text(json.dumps(train_config, indent=2, ensure_ascii=False), encoding="utf-8")

    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_one_epoch(model, adapter, train_loader, optimizer, args.device)
        val_metrics = validate_one_epoch(model, adapter, val_loader, args.device)

        row = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "val_loss": float(val_metrics["loss"]),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "train_log.csv", index=False, encoding="utf-8-sig")

        # Save only the encoder model for later embedding extraction.
        state = {
            "epoch": epoch,
            "model": model.online_encoder,
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_config": train_config,
        }
        save_checkpoint(state, str(out_dir / "last.pt"))

        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            save_checkpoint(state, str(out_dir / "best.pt"))

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f}"
        )


if __name__ == "__main__":
    main()
