from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from beats_adapter import BEATsAdapter
from window_dataset import WindowDataset, load_patient_ids


SRC_SR = 8000


def _collate_waveform_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep waveforms as a Python list and metadata as parallel lists."""
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
    """Build patient-wise train/validation datasets from the fixed window library."""
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


def _augment_waveforms(waveforms: List[np.ndarray]) -> List[np.ndarray]:
    """Create a light waveform-space SSL view.

    The goal here is not to reproduce the exact official BEATs pretraining loss,
    but to provide a stable, minimal domain-adaptation objective compatible with
    the existing adapter and fixed window protocol.
    """
    out: List[np.ndarray] = []
    for w in waveforms:
        x = np.asarray(w, dtype=np.float32).copy()
        if x.ndim != 1:
            raise ValueError(f"Each waveform must be 1D, got shape={x.shape}")

        # Small random gain.
        gain = np.random.uniform(0.95, 1.05)
        x *= np.float32(gain)

        # Small additive Gaussian noise.
        x += np.random.normal(loc=0.0, scale=0.005, size=x.shape).astype(np.float32)

        # Random short time masking.
        n = x.shape[0]
        if n >= 16:
            max_mask = max(1, n // 20)
            mask_width = np.random.randint(0, max_mask + 1)
            if mask_width > 0 and mask_width < n:
                start = np.random.randint(0, n - mask_width + 1)
                x[start : start + mask_width] = 0.0

        out.append(x.astype(np.float32, copy=False))
    return out


def _output_to_embedding(output: Any) -> torch.Tensor:
    """Normalize adapter/model outputs into one [B, D] embedding tensor."""
    if torch.is_tensor(output):
        x = output
    elif isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
        x = output[0]
    elif isinstance(output, dict):
        for key in ("embedding", "features", "x", "last_hidden_state"):
            if key in output and torch.is_tensor(output[key]):
                x = output[key]
                break
        else:
            raise ValueError("Could not find a tensor-like feature in model_output dict.")
    else:
        raise ValueError(f"Unsupported model_output type: {type(output)}")

    if x.ndim == 3:
        x = x.mean(dim=1)
    if x.ndim != 2:
        raise ValueError(f"Expected embedding shape [B, D], got {tuple(x.shape)}")
    return x


def train_one_epoch(model, loader, optimizer, device) -> Dict[str, float]:
    """Train one epoch with a simple view-consistency objective.

    Parameters
    ----------
    model:
        A trainable BEATsAdapter instance.
    loader:
        DataLoader yielding waveform batches.
    optimizer:
        Optimizer built on ``model.model.parameters()``.
    device:
        Kept for the required interface; the adapter already owns its device.

    Returns
    -------
    dict
        At least ``{"loss": float}``.
    """
    del device  # adapter already carries the actual torch device

    model.model.train()
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        waveforms = batch["waveforms"]
        src_sr = int(batch["src_sr"])

        view1 = _augment_waveforms(waveforms)
        view2 = _augment_waveforms(waveforms)

        out1 = model.forward_for_ssl(view1, src_sr=src_sr)
        out2 = model.forward_for_ssl(view2, src_sr=src_sr)

        emb1 = _output_to_embedding(out1["model_output"])
        emb2 = _output_to_embedding(out2["model_output"])
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)

        loss = 1.0 - (emb1 * emb2).sum(dim=1).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bsz = int(emb1.shape[0])
        total_loss += float(loss.detach().item()) * bsz
        total_items += bsz

    mean_loss = total_loss / max(total_items, 1)
    return {"loss": float(mean_loss)}


@torch.no_grad()
def validate_one_epoch(model, loader, device) -> Dict[str, float]:
    """Validate one epoch with the same view-consistency loss."""
    del device

    model.model.eval()
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        waveforms = batch["waveforms"]
        src_sr = int(batch["src_sr"])

        view1 = _augment_waveforms(waveforms)
        view2 = _augment_waveforms(waveforms)

        out1 = model.forward_for_ssl(view1, src_sr=src_sr)
        out2 = model.forward_for_ssl(view2, src_sr=src_sr)

        emb1 = _output_to_embedding(out1["model_output"])
        emb2 = _output_to_embedding(out2["model_output"])
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)

        loss = 1.0 - (emb1 * emb2).sum(dim=1).mean()

        bsz = int(emb1.shape[0])
        total_loss += float(loss.detach().item()) * bsz
        total_items += bsz

    mean_loss = total_loss / max(total_items, 1)
    return {"loss": float(mean_loss)}


def save_checkpoint(state: dict, ckpt_path: str) -> None:
    """Save a checkpoint to disk."""
    path = Path(ckpt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continue pretraining / domain adaptation of BEATs on an unlabeled heart-sound window library."
    )
    parser.add_argument("--window-index-csv", type=str, required=True)
    parser.add_argument("--recording-manifest-csv", type=str, required=True)
    parser.add_argument("--train-patient-list", type=str, required=True)
    parser.add_argument("--val-patient-list", type=str, required=True)
    parser.add_argument("--init-checkpoint-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
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

    adapter = BEATsAdapter(
        checkpoint_path=args.init_checkpoint_path,
        device=args.device,
        trainable=True,
    )
    optimizer = torch.optim.Adam(
        [p for p in adapter.model.parameters() if p.requires_grad],
        lr=float(args.lr),
    )

    history: List[Dict[str, float]] = []
    best_val = float("inf")

    train_config = {
        "window_index_csv": args.window_index_csv,
        "recording_manifest_csv": args.recording_manifest_csv,
        "train_patient_list": args.train_patient_list,
        "val_patient_list": args.val_patient_list,
        "init_checkpoint_path": args.init_checkpoint_path,
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "device": args.device,
        "src_sr": SRC_SR,
        "objective": "two-view cosine consistency on BEATs embeddings",
    }
    (out_dir / "train_config.json").write_text(json.dumps(train_config, indent=2, ensure_ascii=False), encoding="utf-8")

    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_one_epoch(adapter, train_loader, optimizer, args.device)
        val_metrics = validate_one_epoch(adapter, val_loader, args.device)

        row = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "val_loss": float(val_metrics["loss"]),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "train_log.csv", index=False, encoding="utf-8-sig")

        state = {
            "epoch": epoch,
            "model": adapter.model,
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
