"""Baseline distance-space builders for robust clinical alignment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

try:  # pragma: no cover
    from .config import (
        PipelineConfig,
        PlotConfig,
        build_default_config,
        ensure_output_dirs,
        log_done,
        log_info,
        print_banner,
        set_publication_plot_style,
    )
    from .core import ClinicalAlignmentData, _ensure_square_distance_matrix, upper_triangle_vector
    from .clinical_processing import normalize_patient_id
except ImportError:  # pragma: no cover
    from config import (
        PipelineConfig,
        PlotConfig,
        build_default_config,
        ensure_output_dirs,
        log_done,
        log_info,
        print_banner,
        set_publication_plot_style,
    )
    from core import ClinicalAlignmentData, _ensure_square_distance_matrix, upper_triangle_vector
    from clinical_processing import normalize_patient_id


# =============================================================================
# Small helpers
# =============================================================================


def _apply_axes_style(ax: plt.Axes, plot_cfg: Optional[PlotConfig] = None, show_grid: bool = False, grid_axis: str = "both") -> None:
    if plot_cfg is None:
        plot_cfg = PlotConfig()
    ax.grid(show_grid, axis=grid_axis, alpha=plot_cfg.axis_grid_alpha, linewidth=plot_cfg.axis_grid_linewidth)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.4)


def _detect_id_col(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Cannot find patient-id column. Candidates={list(candidates)} | columns={list(df.columns)}")


# =============================================================================
# Baseline-space builders
# =============================================================================


def load_patient_embeddings(
    embedding_npy: str | Path,
    meta_csv: str | Path,
    patient_order: Sequence[str],
    patient_id_candidates: Sequence[str] = ("patient_id", "编码", "PID", "pid"),
) -> np.ndarray:
    print_banner("Loading patient-level embedding baseline")
    embedding_npy = Path(embedding_npy)
    meta_csv = Path(meta_csv)
    if not embedding_npy.exists():
        raise FileNotFoundError(f"Patient embedding npy not found: {embedding_npy}")
    if not meta_csv.exists():
        raise FileNotFoundError(f"Patient embedding meta csv not found: {meta_csv}")

    log_info(f"patient_embedding_npy  : {embedding_npy}")
    log_info(f"patient_embedding_meta : {meta_csv}")

    X = np.load(embedding_npy)
    meta = pd.read_csv(meta_csv)
    id_col = _detect_id_col(meta, patient_id_candidates)
    meta["patient_id"] = normalize_patient_id(meta[id_col])

    if len(meta) != len(X):
        raise ValueError(f"Embedding rows != meta rows: {len(X)} vs {len(meta)}")
    if X.ndim != 2:
        raise ValueError(f"Patient embedding array must be 2D, got shape={X.shape}")

    emb_df = pd.DataFrame({"patient_id": meta["patient_id"].astype(str).tolist()})
    emb_df["row_idx"] = np.arange(len(emb_df), dtype=int)
    emb_df = emb_df.drop_duplicates(subset=["patient_id"], keep="first")

    order_df = pd.DataFrame({"patient_id": list(map(str, patient_order))})
    merged = order_df.merge(emb_df, on="patient_id", how="left", validate="1:1")
    if merged["row_idx"].isna().any():
        missing = int(merged["row_idx"].isna().sum())
        raise ValueError(f"{missing} patients in patient_order missing from patient embedding metadata.")

    idx = merged["row_idx"].to_numpy(dtype=int)
    X_aligned = np.asarray(X[idx], dtype=np.float64)
    log_done(f"Loaded and aligned patient embeddings | shape={X_aligned.shape}")
    return X_aligned




def _detect_position_col(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Cannot find position column. Candidates={list(candidates)} | columns={list(df.columns)}")


def load_position_embeddings_by_position(
    embedding_npy: str | Path,
    meta_csv: str | Path,
    patient_order: Sequence[str],
    positions: Sequence[str] = ("A", "E", "M", "P", "T"),
    patient_id_candidates: Sequence[str] = ("patient_id", "编码", "PID", "pid"),
    position_candidates: Sequence[str] = ("position", "Position", "pos", "site", "auscultation_position"),
) -> Dict[str, np.ndarray]:
    """Load position-level embeddings and align each position to patient_order.

    Returns
    -------
    Dict[str, np.ndarray]
        For each requested position, an array of shape [n_patients, d] aligned to
        patient_order. Missing patient-position rows are filled with NaN so that
        downstream distance construction can average across available positions.
    """
    print_banner("Loading position-level embedding baseline")
    embedding_npy = Path(embedding_npy)
    meta_csv = Path(meta_csv)
    if not embedding_npy.exists():
        raise FileNotFoundError(f"Position embedding npy not found: {embedding_npy}")
    if not meta_csv.exists():
        raise FileNotFoundError(f"Position embedding meta csv not found: {meta_csv}")

    log_info(f"position_embedding_npy  : {embedding_npy}")
    log_info(f"position_embedding_meta : {meta_csv}")

    X = np.load(embedding_npy)
    meta = pd.read_csv(meta_csv)
    id_col = _detect_id_col(meta, patient_id_candidates)
    pos_col = _detect_position_col(meta, position_candidates)
    meta = meta.copy()
    meta["patient_id"] = normalize_patient_id(meta[id_col])
    meta["position"] = meta[pos_col].astype(str).str.strip().str.upper()

    if len(meta) != len(X):
        raise ValueError(f"Position embedding rows != meta rows: {len(X)} vs {len(meta)}")
    if X.ndim != 2:
        raise ValueError(f"Position embedding array must be 2D, got shape={X.shape}")

    work = pd.DataFrame({
        "patient_id": meta["patient_id"].astype("object"),
        "position": meta["position"].astype(str),
        "row_idx": np.arange(len(meta), dtype=int),
    })
    work = work.loc[work["patient_id"].notna()].copy()
    work = work.drop_duplicates(subset=["patient_id", "position"], keep="first")

    order_df = pd.DataFrame({"patient_id": list(map(str, patient_order))})
    out: Dict[str, np.ndarray] = {}
    d = int(X.shape[1])
    for pos in [str(p).upper() for p in positions]:
        sub = work.loc[work["position"] == pos, ["patient_id", "row_idx"]].copy()
        merged = order_df.merge(sub, on="patient_id", how="left", validate="1:1")
        arr = np.full((len(order_df), d), np.nan, dtype=np.float64)
        has = merged["row_idx"].notna().to_numpy()
        if np.any(has):
            arr[has] = np.asarray(X[merged.loc[has, "row_idx"].to_numpy(dtype=int)], dtype=np.float64)
        missing = int((~has).sum())
        if missing > 0:
            log_info(f"Position {pos}: missing embeddings for {missing} patients; distances will average across available positions.")
        out[pos] = arr
    log_done(f"Loaded and aligned position embeddings | positions={list(out.keys())} | shape_per_position={[v.shape for v in out.values()]}")
    return out


def compute_position_average_embedding_distance_matrix(
    position_embeddings: Mapping[str, np.ndarray],
    metric: str = "cosine",
) -> np.ndarray:
    """Compute patient distance by averaging position-specific embedding distances.

    This is a decomposition-matched mean baseline:
    window -> position mean embedding (already provided) -> position distance ->
    average across positions at the patient-pair level.
    """
    print_banner(f"Computing position-averaged embedding distance | metric={metric}")
    if len(position_embeddings) == 0:
        raise ValueError("position_embeddings is empty.")
    names = list(position_embeddings.keys())
    first = np.asarray(position_embeddings[names[0]], dtype=np.float64)
    if first.ndim != 2:
        raise ValueError(f"Each position embedding array must be 2D, got shape={first.shape}")
    n = int(first.shape[0])
    accum = np.zeros((n, n), dtype=np.float64)
    weight_sum = np.zeros((n, n), dtype=np.float64)

    for pos, X_pos in position_embeddings.items():
        X_pos = np.asarray(X_pos, dtype=np.float64)
        if X_pos.shape[0] != n:
            raise ValueError(f"Position {pos} has inconsistent patient count: expected {n}, got {X_pos.shape[0]}")
        if X_pos.ndim != 2:
            raise ValueError(f"Position {pos} embedding array must be 2D, got shape={X_pos.shape}")
        valid = np.all(np.isfinite(X_pos), axis=1)
        if int(valid.sum()) < 2:
            log_info(f"Skipping position {pos}: <2 valid embeddings.")
            continue
        X_valid = X_pos[valid]
        if metric == "cosine":
            norms = np.linalg.norm(X_valid, axis=1, keepdims=True)
            X_valid = X_valid / np.clip(norms, 1e-12, None)
        D_valid = squareform(pdist(X_valid, metric=metric), checks=False)
        full = np.zeros((n, n), dtype=np.float64)
        valid_idx = np.where(valid)[0]
        full[np.ix_(valid_idx, valid_idx)] = D_valid
        w = np.zeros((n, n), dtype=np.float64)
        w[np.ix_(valid_idx, valid_idx)] = 1.0
        accum += full * w
        weight_sum += w

    if np.any(weight_sum == 0):
        zero_pairs = int(np.sum(weight_sum == 0))
        raise ValueError(
            f"Some patient pairs have zero shared valid positions in the position-level mean baseline: {zero_pairs} entries. "
            "Please check position embeddings / metadata coverage."
        )
    D = np.divide(accum, weight_sum, out=np.zeros_like(accum), where=weight_sum > 0)
    D = _ensure_square_distance_matrix(D, name=f"position_average_embedding_distance_{metric}")
    log_done(f"Computed position-averaged embedding distance matrix | shape={D.shape}")
    return D


def compute_embedding_distance_matrix(patient_embeddings: np.ndarray, metric: str = "cosine") -> np.ndarray:
    print_banner(f"Computing patient embedding distance | metric={metric}")
    X = np.asarray(patient_embeddings, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"patient_embeddings must be 2D, got shape={X.shape}")
    if metric == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.clip(norms, 1e-12, None)
    D = squareform(pdist(X, metric=metric), checks=False)
    D = _ensure_square_distance_matrix(D, name=f"embedding_distance_{metric}")
    log_done(f"Computed patient embedding distance matrix | shape={D.shape}")
    return D


def load_external_distance_matrix(distance_npy: str | Path, patient_order_csv: str | Path, prepared_patient_order: Sequence[str]) -> np.ndarray:
    print_banner("Loading external baseline distance matrix")
    distance_npy = Path(distance_npy)
    patient_order_csv = Path(patient_order_csv)
    if not distance_npy.exists():
        raise FileNotFoundError(f"External baseline distance npy not found: {distance_npy}")
    if not patient_order_csv.exists():
        raise FileNotFoundError(f"External baseline patient order csv not found: {patient_order_csv}")

    D = np.load(distance_npy)
    D = _ensure_square_distance_matrix(D, name=f"external_distance_{distance_npy.stem}")
    order_df = pd.read_csv(patient_order_csv)
    if "patient_id" not in order_df.columns:
        raise ValueError(f"patient order csv must contain patient_id: {patient_order_csv}")
    external_order = normalize_patient_id(order_df["patient_id"]).astype(str).tolist()
    if len(external_order) != D.shape[0]:
        raise ValueError(f"External patient order length mismatch: len(order)={len(external_order)} vs D.shape[0]={D.shape[0]}")

    lookup = {pid: idx for idx, pid in enumerate(external_order)}
    missing = [pid for pid in prepared_patient_order if pid not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} prepared patients missing from external baseline order. Example: {missing[:5]}")
    idx = np.array([lookup[str(pid)] for pid in prepared_patient_order], dtype=int)
    D_aligned = D[np.ix_(idx, idx)]
    D_aligned = _ensure_square_distance_matrix(D_aligned, name="external_distance_aligned")
    log_done(f"Loaded and aligned external distance matrix | shape={D_aligned.shape}")
    return D_aligned


def prepare_baseline_distance_matrices(data: ClinicalAlignmentData, cfg: PipelineConfig) -> Dict[str, np.ndarray]:
    print_banner("Preparing baseline distance spaces")
    baselines: Dict[str, np.ndarray] = {}
    requested = tuple(cfg.analysis.baseline_spaces)
    if "pointcloud_ot" in requested:
        baselines["pointcloud_ot"] = np.asarray(data.acoustic_distance, dtype=np.float64)
        log_done("Added baseline space: pointcloud_ot")
    if "beats_meanpool" in requested:
        beats_pos_npy = Path(getattr(cfg.paths, "beats_position_embedding_npy", ""))
        beats_pos_meta = Path(getattr(cfg.paths, "beats_position_embedding_meta_csv", ""))
        if str(beats_pos_npy) not in {"", "."} and beats_pos_npy.exists() and str(beats_pos_meta) not in {"", "."} and beats_pos_meta.exists():
            X_pos = load_position_embeddings_by_position(
                beats_pos_npy,
                beats_pos_meta,
                data.patient_order,
                positions=getattr(cfg.analysis, "positions", ("A", "E", "M", "P", "T")),
                patient_id_candidates=cfg.analysis.patient_id_candidates,
            )
            baselines["beats_meanpool"] = compute_position_average_embedding_distance_matrix(X_pos, metric="cosine")
            log_done("Added baseline space: beats_meanpool (position mean -> position distance -> average across positions)")
        else:
            X = load_patient_embeddings(cfg.paths.beats_patient_embedding_npy, cfg.paths.beats_patient_embedding_meta_csv, data.patient_order, patient_id_candidates=cfg.analysis.patient_id_candidates)
            baselines["beats_meanpool"] = compute_embedding_distance_matrix(X, metric="cosine")
            log_done("Added baseline space: beats_meanpool (legacy patient-level embedding distance)")
    if "ead_meanpool" in requested:
        X = load_patient_embeddings(cfg.paths.ead_patient_embedding_npy, cfg.paths.ead_patient_embedding_meta_csv, data.patient_order, patient_id_candidates=cfg.analysis.patient_id_candidates)
        baselines["ead_meanpool"] = compute_embedding_distance_matrix(X, metric="cosine")
    if "ead_pointcloud_ot" in requested:
        baselines["ead_pointcloud_ot"] = load_external_distance_matrix(cfg.paths.ead_pointcloud_distance_npy, cfg.paths.ead_pointcloud_patient_order_csv, data.patient_order)
    log_done(f"Prepared baseline spaces: {list(baselines.keys())}")
    return baselines


# =============================================================================
# Space-comparison utilities
# =============================================================================


def compare_distance_spaces(distance_mats: Mapping[str, np.ndarray]) -> pd.DataFrame:
    print_banner("Comparing distance spaces")
    names = list(distance_mats.keys())
    tri = {name: upper_triangle_vector(_ensure_square_distance_matrix(D, name=name)) for name, D in distance_mats.items()}
    rows: list[dict[str, Any]] = []
    for name_i in names:
        for name_j in names:
            xi = tri[name_i]
            xj = tri[name_j]
            pearson = float(np.corrcoef(xi, xj)[0, 1]) if np.std(xi) > 0 and np.std(xj) > 0 else np.nan
            spearman = float(pd.Series(xi).corr(pd.Series(xj), method="spearman"))
            rows.append({"space_i": name_i, "space_j": name_j, "pearson_r": pearson, "spearman_r": spearman})
    out = pd.DataFrame(rows)
    log_done(f"Computed distance-space comparison table | rows={len(out)}")
    return out


def plot_distance_space_correlation_heatmap(comparison_df: pd.DataFrame, metric: str, out_path: str | Path, plot_cfg: Optional[PlotConfig] = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    set_publication_plot_style(plot_cfg)
    plot_cfg = PlotConfig() if plot_cfg is None else plot_cfg
    pivot = comparison_df.pivot(index="space_i", columns="space_j", values=metric)
    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    cmap = sns.blend_palette(["#F7FBFF", "#9ECAE1", "#3182BD", "#08519C"], as_cmap=True)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=plot_cfg.heatmap_grid_linewidth,
        linecolor=plot_cfg.heatmap_grid_color,
        cbar_kws={"shrink": 0.85, "label": metric},
        ax=ax,
    )
    ax.set_title(f"Distance-space comparison ({metric})")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=plot_cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    log_done(f"Saved distance-space correlation heatmap to: {out_path}")


def save_baseline_artifacts(baseline_mats: Mapping[str, np.ndarray], comparison_df: pd.DataFrame, out_dir: str | Path, plot_cfg: Optional[PlotConfig] = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, D in baseline_mats.items():
        np.save(out_dir / f"distance_{name}.npy", D)
    comparison_df.to_csv(out_dir / "distance_space_comparison.csv", index=False)
    plot_distance_space_correlation_heatmap(comparison_df, "pearson_r", out_dir / "distance_space_pearson_heatmap.png", plot_cfg=plot_cfg)
    plot_distance_space_correlation_heatmap(comparison_df, "spearman_r", out_dir / "distance_space_spearman_heatmap.png", plot_cfg=plot_cfg)
    summary = {"spaces": list(baseline_mats.keys())}
    with (out_dir / "baseline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log_done(f"Saved baseline artifacts to: {out_dir}")


# =============================================================================
# CLI
# =============================================================================


def _build_cli() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Build baseline distance spaces for robust clinical alignment.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prepared-dir", type=str, default=None, help="Directory containing prepared_data.pkl")
    parser.add_argument("--beats-patient-embedding-npy", type=str, default=None)
    parser.add_argument("--beats-patient-embedding-meta-csv", type=str, default=None)
    parser.add_argument("--ead-patient-embedding-npy", type=str, default=None)
    parser.add_argument("--ead-patient-embedding-meta-csv", type=str, default=None)
    parser.add_argument("--ead-pointcloud-distance-npy", type=str, default=None)
    parser.add_argument("--ead-pointcloud-patient-order-csv", type=str, default=None)
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()
    cfg = build_default_config()
    if args.beats_patient_embedding_npy:
        cfg.paths.beats_patient_embedding_npy = args.beats_patient_embedding_npy
    if args.beats_patient_embedding_meta_csv:
        cfg.paths.beats_patient_embedding_meta_csv = args.beats_patient_embedding_meta_csv
    if args.ead_patient_embedding_npy:
        cfg.paths.ead_patient_embedding_npy = args.ead_patient_embedding_npy
    if args.ead_patient_embedding_meta_csv:
        cfg.paths.ead_patient_embedding_meta_csv = args.ead_patient_embedding_meta_csv
    if args.ead_pointcloud_distance_npy:
        cfg.paths.ead_pointcloud_distance_npy = args.ead_pointcloud_distance_npy
    if args.ead_pointcloud_patient_order_csv:
        cfg.paths.ead_pointcloud_patient_order_csv = args.ead_pointcloud_patient_order_csv
    if args.out_root:
        cfg.paths.output_root = args.out_root
    if args.seed is not None:
        cfg.analysis.random_seed = int(args.seed)

    dirs = ensure_output_dirs(cfg.paths.output_root)
    prepared_dir = Path(args.prepared_dir) if args.prepared_dir else dirs["prepared"]
    data = ClinicalAlignmentData.load(prepared_dir)
    baseline_mats = prepare_baseline_distance_matrices(data, cfg)
    comparison_df = compare_distance_spaces(baseline_mats)
    save_baseline_artifacts(baseline_mats, comparison_df, dirs["robust_baselines"], plot_cfg=cfg.plot)
    log_done("Baseline preparation finished.")


if __name__ == "__main__":  # pragma: no cover
    main()
