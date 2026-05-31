"""Main and robustness analyses for patient-wise clinical alignment.

This module implements three manuscript-core analyses:
1. Global distance-to-distance alignment.
2. Neighborhood consistency analysis.
3. Retrieval-based clinical validation.

It also implements four robustness / supplementary analyses:
1. Confounder-adjusted global alignment.
2. Patient-level bootstrap / subsampling confidence intervals.
3. Baseline-space comparison.
4. Discovery / validation split analysis.

Note
----
The earlier retrieval-confidence / abstention block was removed because it did
not provide stable, supportive evidence under the current confidence heuristic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, r2_score, explained_variance_score, cohen_kappa_score

try:  # pragma: no cover
    from .config import (
        PlotConfig,
        set_publication_plot_style,
        print_banner,
        log_info,
        log_warn,
        log_done,
    )
    from .clinical_registry import (
        VariableSpec,
        get_neighbor_variables,
        get_retrieval_targets,
    )
    from .core import (
        ClinicalAlignmentData,
        compute_clinical_distance_matrix,
        build_all_single_variable_distance_matrices,
        get_knn_from_distance,
        upper_triangle_vector,
    )
except ImportError:  # pragma: no cover
    from config import (
        PlotConfig,
        set_publication_plot_style,
        print_banner,
        log_info,
        log_warn,
        log_done,
    )
    from clinical_registry import (
        VariableSpec,
        get_neighbor_variables,
        get_retrieval_targets,
    )
    from core import (
        ClinicalAlignmentData,
        compute_clinical_distance_matrix,
        build_all_single_variable_distance_matrices,
        get_knn_from_distance,
        upper_triangle_vector,
    )


# =============================================================================
# Small shared helpers
# =============================================================================


def _apply_axes_style(
    ax: plt.Axes,
    plot_cfg: Optional[PlotConfig] = None,
    show_grid: bool = False,
    grid_axis: str = "both",
) -> None:
    if plot_cfg is None:
        plot_cfg = PlotConfig()
    ax.grid(show_grid, axis=grid_axis, alpha=plot_cfg.axis_grid_alpha, linewidth=plot_cfg.axis_grid_linewidth)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.4)


def _prepare_out_dir(out_dir: str | Path | None, create: bool = False) -> Optional[Path]:
    if out_dir is None:
        return None
    out_dir = Path(out_dir)
    if create:
        out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _ensure_square_distance_matrix(D: np.ndarray, name: str = "distance_matrix") -> np.ndarray:
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"{name} must be square, got shape={D.shape}")
    if np.any(~np.isfinite(D)):
        raise ValueError(f"{name} contains NaN or Inf.")
    if np.any(D < -1e-10):
        raise ValueError(f"{name} contains negative values.")
    if not np.allclose(D, D.T, atol=1e-7):
        raise ValueError(f"{name} is not symmetric within tolerance.")
    np.fill_diagonal(D, 0.0)
    return D


def _save_fig(fig: plt.Figure, out_path: str | Path, dpi: int = 300) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log_done(f"Saved figure to: {out_path}")


def _sample_for_scatter(x: np.ndarray, y: np.ndarray, max_points: int = 50000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) <= max_points:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_points, replace=False)
    return x[idx], y[idx]


def _subset_distance_matrix(D: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return _ensure_square_distance_matrix(D[np.ix_(idx, idx)], name="subset_distance")


def _subset_dataframe(df: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    return df.iloc[idx].reset_index(drop=True).copy()


def _get_patient_level_bootstrap_indices(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=n, endpoint=False)


def _get_patient_level_subsample_indices(n: int, seed: int, fraction: float = 0.8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = int(round(float(fraction) * n))
    m = min(max(m, 3), n)
    return np.sort(rng.choice(np.arange(n, dtype=int), size=m, replace=False))


def _matrix_correlations(D1: np.ndarray, D2: np.ndarray) -> Dict[str, float]:
    x = upper_triangle_vector(D1).astype(np.float64)
    y = upper_triangle_vector(D2).astype(np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return {"pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan, "n_pairs": int(len(x))}
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {"pearson_r": float(pr), "pearson_p": float(pp), "spearman_r": float(sr), "spearman_p": float(sp), "n_pairs": int(len(x))}


def _resolve_sigma(distances: np.ndarray, sigma: str | float) -> float:
    if isinstance(sigma, str) and sigma == "median":
        d = np.asarray(distances, dtype=np.float64)
        d = d[np.isfinite(d)]
        d = d[d > 0]
        if len(d) == 0:
            return 1.0
        return max(float(np.median(d)), 1e-8)
    try:
        return max(float(sigma), 1e-8)
    except Exception as exc:
        raise ValueError(f"Unsupported sigma setting: {sigma}") from exc


def _kernel_weights(distances: np.ndarray, kernel: str = "rbf", sigma: str | float = "median") -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float64)
    if kernel == "rbf":
        s = _resolve_sigma(distances, sigma)
        w = np.exp(-(distances ** 2) / (s ** 2))
    elif kernel == "inverse_distance":
        w = 1.0 / np.maximum(distances, 1e-8)
    else:
        raise ValueError(f"Unsupported retrieval kernel: {kernel}")
    if np.sum(w) <= 0:
        w = np.ones_like(w)
    return w / np.sum(w)


def _quantile_interval(values: Sequence[float], ci: float = 0.95) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    alpha = 1.0 - float(ci)
    lo = float(np.quantile(arr, alpha / 2))
    hi = float(np.quantile(arr, 1 - alpha / 2))
    return lo, hi


def _binned_trend(x: np.ndarray, y: np.ndarray, n_bins: int = 24) -> pd.DataFrame:
    if len(x) < max(10, n_bins):
        return pd.DataFrame(columns=["x_mid", "y_med", "y_lo", "y_hi"])
    bins = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), n_bins + 1)
    out: List[Dict[str, float]] = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (x >= left) & (x < right if right < bins[-1] else x <= right)
        if np.sum(mask) < 10:
            continue
        ys = y[mask]
        out.append(
            {
                "x_mid": float((left + right) / 2.0),
                "y_med": float(np.nanmedian(ys)),
                "y_lo": float(np.nanquantile(ys, 0.25)),
                "y_hi": float(np.nanquantile(ys, 0.75)),
            }
        )
    return pd.DataFrame(out)


def _metric_display_name(metric: str) -> str:
    return {
        "pearson_r": "Pearson r",
        "spearman_r": "Spearman r",
        "improvement_ratio": "Improvement ratio",
        "balanced_accuracy": "Balanced accuracy",
        "spearman": "Spearman",
    }.get(metric, metric)


def _to_evaluation_scale(values: np.ndarray, spec: VariableSpec) -> np.ndarray:
    """Map cleaned values to an interpretable evaluation scale.

    Variables cleaned with log1p are inverse-transformed back to their original
    units before error-based retrieval metrics are computed.
    """
    arr = np.asarray(values, dtype=np.float64)
    if str(getattr(spec, "transform", "none")).lower() == "log1p":
        return np.expm1(arr)
    return arr


def _robust_iqr_denom(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    q1 = float(np.nanquantile(arr, 0.25))
    q3 = float(np.nanquantile(arr, 0.75))
    denom = q3 - q1
    if not np.isfinite(denom) or denom <= 1e-12:
        denom = float(np.nanstd(arr))
    if not np.isfinite(denom) or denom <= 1e-12:
        denom = float(np.nanmax(arr) - np.nanmin(arr))
    if not np.isfinite(denom) or denom <= 1e-12:
        denom = 1.0
    return denom


def _continuous_tolerance(spec: VariableSpec, y_true_eval: np.ndarray) -> float:
    preset = {
        "NTproBNP": 300.0,
        "LA_mm": 5.0,
        "LVEDD_mm": 5.0,
        "EF_Teich": 5.0,
        "heart_rate": 10.0,
        "Hb": 10.0,
        "CRP": 5.0,
        "D_dimer": 0.25,
        "hsTnT": 0.014,
        "IVS_mm": 2.0,
        "LVPW_mm": 2.0,
    }
    if spec.clean_name in preset:
        return float(preset[spec.clean_name])
    return 0.5 * _robust_iqr_denom(y_true_eval)


def _within_tolerance_accuracy(y_true: np.ndarray, y_pred: np.ndarray, tol: float) -> float:
    if not np.isfinite(tol) or tol <= 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred) <= tol))


def _quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return np.nan
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    valid = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[valid]
    yp = yp[valid]
    if len(yt) == 0:
        return np.nan
    min_cat = int(np.nanmin(yt))
    max_cat = int(np.nanmax(yt))
    yp_round = np.rint(yp).astype(int)
    yp_round = np.clip(yp_round, min_cat, max_cat)
    try:
        return float(cohen_kappa_score(yt.astype(int), yp_round, weights="quadratic"))
    except Exception:
        return np.nan


def _concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    valid = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[valid]
    yp = yp[valid]
    n = int(len(yt))
    if n < 2:
        return np.nan
    iu = np.triu_indices(n, k=1)
    dt = yt[iu[0]] - yt[iu[1]]
    dp = yp[iu[0]] - yp[iu[1]]
    valid_pairs = dt != 0
    dt = dt[valid_pairs]
    dp = dp[valid_pairs]
    if len(dt) == 0:
        return np.nan
    concordant = np.sum(dt * dp > 0)
    ties = np.sum(dp == 0)
    return float((concordant + 0.5 * ties) / len(dt))


# =============================================================================
# Plotting
# =============================================================================


def _benjamini_hochberg_qvalues(values: pd.Series) -> pd.Series:
    """Compute Benjamini-Hochberg q-values while preserving the original index."""
    s = pd.to_numeric(values, errors="coerce").astype(float)
    mask = s.notna()
    if int(mask.sum()) == 0:
        return pd.Series(np.nan, index=s.index, dtype=float)

    p = s.loc[mask].to_numpy(dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order]
    m = float(len(ranked))
    q = ranked * m / np.arange(1, len(ranked) + 1, dtype=np.float64)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    out = pd.Series(np.nan, index=s.index, dtype=float)
    out.loc[s.loc[mask].index[order]] = q
    return out


def _add_fdr_columns(
    df: pd.DataFrame,
    p_col: str,
    overall_col: str,
    grouped_cols: Optional[Sequence[str]] = None,
    grouped_col_name: Optional[str] = None,
) -> pd.DataFrame:
    """Add BH-FDR q-value columns overall and optionally within groups."""
    out = df.copy()
    if p_col not in out.columns:
        return out
    out[overall_col] = _benjamini_hochberg_qvalues(out[p_col])
    if grouped_cols and grouped_col_name:
        out[grouped_col_name] = np.nan
        q_parts: List[pd.Series] = []
        for _, sub in out.groupby(list(grouped_cols), dropna=False, sort=False):
            q_parts.append(_benjamini_hochberg_qvalues(sub[p_col]))
        if len(q_parts) > 0:
            q = pd.concat(q_parts).sort_index()
            out.loc[q.index, grouped_col_name] = q.values
    return out


def load_position_distance_matrices(
    position_distance_dir: str | Path,
    positions: Sequence[str],
    expected_n: Optional[int] = None,
    file_pattern: str = "position_distance_{position}.npy",
) -> Dict[str, np.ndarray]:
    """Load one patient-distance matrix per auscultation position."""
    position_distance_dir = Path(position_distance_dir)
    mats: Dict[str, np.ndarray] = {}
    for pos in positions:
        fp = position_distance_dir / str(file_pattern).format(position=str(pos))
        if not fp.exists():
            raise FileNotFoundError(f"Position distance matrix not found: {fp}")
        D = np.load(fp)
        D = _ensure_square_distance_matrix(D, name=f"position_distance_{pos}")
        if expected_n is not None and D.shape[0] != int(expected_n):
            raise ValueError(
                f"Position distance size mismatch for {pos}: expected {expected_n}, got {D.shape[0]}"
            )
        mats[str(pos)] = D
    return mats


def _aggregate_distance_matrices(
    mats: Mapping[str, np.ndarray],
    selected_positions: Optional[Sequence[str]] = None,
    weights: Optional[Mapping[str, float]] = None,
    aggregation: str = "mean",
) -> np.ndarray:
    """Aggregate multiple position-specific distance matrices into one matrix."""
    if aggregation != "mean":
        raise ValueError(f"Unsupported position aggregation: {aggregation}")
    pos_list = list(selected_positions) if selected_positions is not None else list(mats.keys())
    if len(pos_list) == 0:
        raise ValueError("selected_positions must contain at least one position.")

    arrays: List[np.ndarray] = []
    w_list: List[float] = []
    for pos in pos_list:
        if pos not in mats:
            raise KeyError(f"Missing position matrix for: {pos}")
        arrays.append(_ensure_square_distance_matrix(mats[pos], name=f"position_distance_{pos}"))
        if weights is None:
            w_list.append(1.0)
        else:
            w_list.append(float(weights.get(pos, 1.0)))
    stack = np.stack(arrays, axis=0)
    w = np.asarray(w_list, dtype=np.float64)
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("Position weights must be finite and non-negative.")
    if float(np.sum(w)) <= 0:
        raise ValueError("Sum of position weights must be positive.")
    D = np.average(stack, axis=0, weights=w)
    return _ensure_square_distance_matrix(D, name="aggregated_position_distance")


def build_leave_one_out_position_distance_matrices(
    position_mats: Mapping[str, np.ndarray],
    weights: Optional[Mapping[str, float]] = None,
    aggregation: str = "mean",
) -> Dict[str, np.ndarray]:
    """Build leave-one-position-out aggregated distance matrices."""
    positions = list(position_mats.keys())
    if len(positions) < 2:
        raise ValueError("At least two positions are required for leave-one-out aggregation.")
    loo: Dict[str, np.ndarray] = {}
    for dropped in positions:
        keep = [p for p in positions if p != dropped]
        loo[str(dropped)] = _aggregate_distance_matrices(position_mats, selected_positions=keep, weights=weights, aggregation=aggregation)
    return loo


def run_position_leave_one_out_contribution(
    data: ClinicalAlignmentData,
    registry: Sequence[VariableSpec],
    position_distance_dir: str | Path,
    positions: Sequence[str],
    groups: Sequence[str],
    knn_list: Sequence[int],
    retrieval_kernel: str,
    retrieval_sigma: str | float,
    out_dir: str | Path | None,
    plot_cfg: PlotConfig,
    n_perm: int = 0,
    n_jobs: int = 1,
    random_seed: int = 42,
    file_pattern: str = "position_distance_{position}.npy",
    aggregation: str = "mean",
    weights: Optional[Mapping[str, float]] = None,
    reference_mode: str = "aggregate_all_positions",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate leave-one-position-out performance drops as position contribution.

    Positive drop_from_reference means removing that position hurts performance,
    which supports interpreting that position as contributing useful information.
    """
    print_banner("Running leave-one-position-out position contribution analysis")
    out_dir = _prepare_out_dir(out_dir, create=out_dir is not None)

    pos_mats = load_position_distance_matrices(
        position_distance_dir=position_distance_dir,
        positions=positions,
        expected_n=len(data.patient_order),
        file_pattern=file_pattern,
    )
    if reference_mode != "aggregate_all_positions":
        raise ValueError(f"Unsupported reference_mode: {reference_mode}")
    D_reference = _aggregate_distance_matrices(pos_mats, selected_positions=list(positions), weights=weights, aggregation=aggregation)
    D_loo_dict = build_leave_one_out_position_distance_matrices(pos_mats, weights=weights, aggregation=aggregation)

    ref_corr_vs_main = _matrix_correlations(D_reference, data.acoustic_distance)

    ref_ga = run_global_distance_alignment(
        data=data,
        registry=registry,
        groups=groups,
        n_perm=n_perm,
        n_jobs=n_jobs,
        random_seed=random_seed,
        out_dir=None,
        plot_cfg=plot_cfg,
        acoustic_distance=D_reference,
        save_group_distances=False,
        save_outputs=False,
    )
    ref_nb_group, _ = run_neighborhood_consistency(
        data=data,
        registry=registry,
        groups=groups,
        knn_list=knn_list,
        random_repeats=100,
        random_seed=random_seed,
        out_dir=None,
        plot_cfg=plot_cfg,
        acoustic_distance=D_reference,
        save_outputs=False,
    )
    ref_rt, _, _ = run_retrieval_validation(
        data=data,
        registry=registry,
        knn_list=knn_list,
        retrieval_kernel=retrieval_kernel,
        retrieval_sigma=retrieval_sigma,
        out_dir=None,
        plot_cfg=plot_cfg,
        acoustic_distance=D_reference,
        save_outputs=False,
    )

    global_rows: List[Dict[str, Any]] = []
    neighborhood_rows: List[Dict[str, Any]] = []
    retrieval_rows: List[Dict[str, Any]] = []

    for dropped_pos, D_loo in D_loo_dict.items():
        ga = run_global_distance_alignment(
            data=data,
            registry=registry,
            groups=groups,
            n_perm=n_perm,
            n_jobs=n_jobs,
            random_seed=random_seed,
            out_dir=None,
            plot_cfg=plot_cfg,
            acoustic_distance=D_loo,
            save_group_distances=False,
            save_outputs=False,
        )
        nb_group, _ = run_neighborhood_consistency(
            data=data,
            registry=registry,
            groups=groups,
            knn_list=knn_list,
            random_repeats=100,
            random_seed=random_seed,
            out_dir=None,
            plot_cfg=plot_cfg,
            acoustic_distance=D_loo,
            save_outputs=False,
        )
        rt, _, _ = run_retrieval_validation(
            data=data,
            registry=registry,
            knn_list=knn_list,
            retrieval_kernel=retrieval_kernel,
            retrieval_sigma=retrieval_sigma,
            out_dir=None,
            plot_cfg=plot_cfg,
            acoustic_distance=D_loo,
            save_outputs=False,
        )

        ga_m = ref_ga.merge(ga, on="group", how="inner", suffixes=("_reference", "_loo"))
        for _, row in ga_m.iterrows():
            global_rows.append(
                {
                    "dropped_position": dropped_pos,
                    "group": row["group"],
                    "reference_pearson_r": row["pearson_r_reference"],
                    "loo_pearson_r": row["pearson_r_loo"],
                    "drop_from_reference_pearson_r": row["pearson_r_reference"] - row["pearson_r_loo"],
                    "reference_spearman_r": row["spearman_r_reference"],
                    "loo_spearman_r": row["spearman_r_loo"],
                    "drop_from_reference_spearman_r": row["spearman_r_reference"] - row["spearman_r_loo"],
                }
            )

        nb_m = ref_nb_group.merge(nb_group, on=["group", "k"], how="inner", suffixes=("_reference", "_loo"))
        for _, row in nb_m.iterrows():
            neighborhood_rows.append(
                {
                    "dropped_position": dropped_pos,
                    "group": row["group"],
                    "k": int(row["k"]),
                    "reference_improvement_ratio": row["improvement_ratio_reference"],
                    "loo_improvement_ratio": row["improvement_ratio_loo"],
                    "drop_from_reference_improvement_ratio": row["improvement_ratio_reference"] - row["improvement_ratio_loo"],
                }
            )

        rt_keys = ["target", "group", "target_type", "k"]
        rt_m = ref_rt.merge(rt, on=rt_keys, how="inner", suffixes=("_reference", "_loo"))
        for _, row in rt_m.iterrows():
            metric_name = "spearman" if row["target_type"] in {"continuous", "ordinal"} else "balanced_accuracy"
            retrieval_rows.append(
                {
                    "dropped_position": dropped_pos,
                    "target": row["target"],
                    "group": row["group"],
                    "target_type": row["target_type"],
                    "k": int(row["k"]),
                    "metric_name": metric_name,
                    "reference_metric": row[f"{metric_name}_reference"],
                    "loo_metric": row[f"{metric_name}_loo"],
                    "drop_from_reference_metric": row[f"{metric_name}_reference"] - row[f"{metric_name}_loo"],
                }
            )

    global_df = pd.DataFrame(global_rows).sort_values(["group", "dropped_position"]).reset_index(drop=True)
    neighborhood_df = pd.DataFrame(neighborhood_rows).sort_values(["group", "k", "dropped_position"]).reset_index(drop=True)
    retrieval_df = pd.DataFrame(retrieval_rows).sort_values(["target", "k", "dropped_position"]).reset_index(drop=True)

    if out_dir is not None:
        global_df.to_csv(out_dir / "position_leave_one_out_global_summary.csv", index=False, encoding="utf-8-sig")
        neighborhood_df.to_csv(out_dir / "position_leave_one_out_neighborhood_summary.csv", index=False, encoding="utf-8-sig")
        retrieval_df.to_csv(out_dir / "position_leave_one_out_retrieval_summary.csv", index=False, encoding="utf-8-sig")
        meta = {
            "reference_mode": reference_mode,
            "aggregation": aggregation,
            "positions": list(map(str, positions)),
            "reference_vs_main_acoustic": ref_corr_vs_main,
        }
        with (out_dir / "position_leave_one_out_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    log_done(
        "Position leave-one-out contribution finished | "
        f"global_rows={len(global_df)} | neighborhood_rows={len(neighborhood_df)} | retrieval_rows={len(retrieval_df)}"
    )
    return global_df, neighborhood_df, retrieval_df



def plot_global_alignment_scatter(
    acoustic_distance: np.ndarray,
    clinical_distance: np.ndarray,
    group_name: str,
    out_dir: str | Path,
    plot_cfg: PlotConfig,
    seed: int = 42,
) -> None:
    set_publication_plot_style(plot_cfg)
    x = upper_triangle_vector(acoustic_distance)
    y = upper_triangle_vector(clinical_distance)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    x_s, y_s = _sample_for_scatter(x, y, max_points=plot_cfg.global_scatter_max_points, seed=seed)
    fig, ax = plt.subplots(figsize=(9.6, 8.0))
    cmap = sns.blend_palette(["#F1B44C", "#158F8C", "#2F6DB3", "#7B5AA6"], as_cmap=True)
    hb = ax.hexbin(x_s, y_s, gridsize=46, cmap=cmap, mincnt=1)
    trend = _binned_trend(x_s, y_s, n_bins=24)
    if len(trend) > 0:
        ax.fill_between(trend["x_mid"], trend["y_lo"], trend["y_hi"], color="#C75C5C", alpha=0.18, linewidth=0)
        ax.plot(trend["x_mid"], trend["y_med"], color="#C75C5C", linewidth=2.5, label="Binned median trend")
        ax.legend(loc="upper right")
    cbar = fig.colorbar(hb, ax=ax, shrink=0.82)
    cbar.set_label("Pair count")
    ax.set_title(f"Acoustic vs Clinical Distance | {group_name}")
    ax.set_xlabel("Acoustic distance")
    ax.set_ylabel("Clinical distance")
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False)
    _save_fig(fig, Path(out_dir) / f"acoustic_vs_clinical_distance_{group_name}.png", dpi=plot_cfg.dpi)


def plot_global_alignment_summary(summary_df: pd.DataFrame, out_dir: str | Path, plot_cfg: PlotConfig, stem: str = "global_alignment_summary") -> None:
    if len(summary_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = summary_df[["group", "pearson_r", "spearman_r"]].copy()
    plot_df = plot_df.sort_values("spearman_r", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(plot_df), dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, max(5.5, 1.35 * len(plot_df))))
    for y, (_, row) in zip(y_pos, plot_df.iterrows()):
        ax.plot([row["pearson_r"], row["spearman_r"]], [y, y], color="#B7B7B7", linewidth=2.0, zorder=1)
    ax.scatter(plot_df["pearson_r"], y_pos, s=140, color="#158F8C", label="Pearson r", zorder=3)
    ax.scatter(plot_df["spearman_r"], y_pos, s=140, color="#2F6DB3", label="Spearman r", zorder=3)
    ax.axvline(0.0, color="#7F7F7F", linestyle="--", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["group"].tolist())
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Clinical group")
    ax.set_title("Global acoustic-clinical distance alignment")
    ax.legend(loc="best")
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis="x")
    _save_fig(fig, Path(out_dir) / f"{stem}.png", dpi=plot_cfg.dpi)


def plot_global_alignment_forest(summary_df: pd.DataFrame, out_dir: str | Path, plot_cfg: PlotConfig, stem: str = "bootstrap_global_alignment_forest") -> None:
    if len(summary_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = summary_df.copy()
    plot_df["label"] = plot_df["group"].astype(str) + " | " + plot_df["metric"].map(_metric_display_name)
    plot_df = plot_df.sort_values(["group", "metric"]).reset_index(drop=True)
    y = np.arange(len(plot_df))[::-1]

    fig, ax = plt.subplots(figsize=(11, max(6.5, 0.7 * len(plot_df))))
    colors = {"pearson_r": "#158F8C", "spearman_r": "#2F6DB3", "improvement_ratio": "#E38B2C", "balanced_accuracy": "#7B5AA6", "spearman": "#C75C5C"}
    for yi, (_, row) in zip(y, plot_df.iterrows()):
        ax.plot([row["ci_lower"], row["ci_upper"]], [yi, yi], color=colors.get(str(row["metric"]), "#4C4C4C"), linewidth=2.6)
        ax.scatter(row["estimate_mean"], yi, s=110, color=colors.get(str(row["metric"]), "#4C4C4C"), zorder=3)
    ax.axvline(0.0, linestyle="--", color="#7F7F7F", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"].tolist())
    ax.set_xlabel("Estimate (mean with bootstrap CI)")
    ax.set_ylabel("")
    ax.set_title("Bootstrap robustness | global alignment")
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis="x")
    _save_fig(fig, Path(out_dir) / f"{stem}.png", dpi=plot_cfg.dpi)


def plot_bootstrap_global_distribution(raw_df: pd.DataFrame, out_dir: str | Path, plot_cfg: PlotConfig, stem: str = "bootstrap_global_alignment_distribution") -> None:
    if len(raw_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = raw_df.melt(id_vars=["group", "repeat"], value_vars=["pearson_r", "spearman_r"], var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    sns.violinplot(data=plot_df, x="group", y="value", hue="metric", inner=None, cut=0, linewidth=0, ax=ax)
    sns.boxplot(data=plot_df, x="group", y="value", hue="metric", showcaps=True, boxprops={"facecolor": "none", "edgecolor": "black"}, whiskerprops={"linewidth": 1.0}, medianprops={"color": "black"}, showfliers=False, width=0.28, ax=ax)
    sns.stripplot(data=plot_df.sample(min(len(plot_df), 1200), random_state=42), x="group", y="value", hue="metric", dodge=True, alpha=0.18, size=3.0, color="black", ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ["Pearson r", "Spearman r"], loc="best")
    ax.set_title("Bootstrap robustness | global alignment distribution")
    ax.set_xlabel("Clinical group")
    ax.set_ylabel("Bootstrap estimate")
    ax.axhline(0.0, linestyle="--", color="#7F7F7F", linewidth=1.2)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis="y")
    _save_fig(fig, Path(out_dir) / f"{stem}.png", dpi=plot_cfg.dpi)


def plot_variablewise_global_alignment_topn(
    summary_df: pd.DataFrame,
    out_dir: str | Path,
    plot_cfg: PlotConfig,
    top_n: int = 12,
    stem: str = "variablewise_global_alignment_topn",
) -> None:
    if len(summary_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = summary_df.sort_values("spearman_r", ascending=False).head(int(top_n)).copy()
    plot_df = plot_df.sort_values("spearman_r", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11.5, max(6.8, 0.55 * len(plot_df))))
    ax.hlines(y=plot_df["variable"], xmin=0.0, xmax=plot_df["spearman_r"], color="#B7B7B7", linewidth=2.0)
    sns.scatterplot(data=plot_df, x="spearman_r", y="variable", hue="group", s=130, ax=ax, zorder=3)
    ax.axvline(0.0, linestyle="--", color="#7F7F7F", linewidth=1.2)
    ax.set_title(f"Variable-wise global alignment | top {int(top_n)}")
    ax.set_xlabel("Spearman correlation")
    ax.set_ylabel("Clinical variable")
    ax.legend(loc="best")
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis="x")
    _save_fig(fig, Path(out_dir) / f"{stem}.png", dpi=plot_cfg.dpi)


def plot_variablewise_global_alignment_all(
    summary_df: pd.DataFrame,
    out_dir: str | Path,
    plot_cfg: PlotConfig,
    stem: str = "variablewise_global_alignment_all",
) -> None:
    if len(summary_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = summary_df.copy()
    plot_df["group_rank"] = plot_df["group"].map({"function": 0, "structure": 1, "burden": 2}).fillna(99)
    plot_df = plot_df.sort_values(["group_rank", "spearman_r"], ascending=[True, True]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12.8, max(8.0, 0.4 * len(plot_df))))
    sns.scatterplot(data=plot_df, x="spearman_r", y="variable", hue="group", size="coverage", sizes=(60, 180), ax=ax, zorder=3)
    ax.axvline(0.0, linestyle="--", color="#7F7F7F", linewidth=1.2)
    ax.set_title("Variable-wise global alignment | all variables")
    ax.set_xlabel("Spearman correlation")
    ax.set_ylabel("Clinical variable")
    ax.legend(loc="best")
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis="x")
    _save_fig(fig, Path(out_dir) / f"{stem}.png", dpi=plot_cfg.dpi)


def plot_neighborhood_summary(summary_df: pd.DataFrame, out_dir: str | Path, plot_cfg: PlotConfig, stem: str = "neighborhood_group_summary") -> None:
    if len(summary_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = summary_df.sort_values(["group", "k"]).copy()
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    sns.lineplot(data=plot_df, x="k", y="improvement_ratio", hue="group", marker="o", linewidth=2.6, markersize=9, ax=ax)
    ax.axhline(1.0, linestyle="--", color="#7F7F7F", linewidth=1.2)
    ax.set_title("Neighborhood clinical consistency")
    ax.set_xlabel("Top-k acoustic neighbors")
    ax.set_ylabel("Improvement ratio vs random")
    ax.legend(loc="best")
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis="y")
    _save_fig(fig, Path(out_dir) / f"{stem}.png", dpi=plot_cfg.dpi)


def plot_bootstrap_neighborhood_summary(summary_df: pd.DataFrame, out_dir: str | Path, plot_cfg: PlotConfig, stem: str = "bootstrap_neighborhood_summary") -> None:
    if len(summary_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    fig, ax = plt.subplots(figsize=(10.8, 7.3))
    for group, sub in summary_df.groupby("group", dropna=False):
        sub = sub.sort_values("k")
        ax.plot(sub["k"], sub["estimate_mean"], marker="o", linewidth=2.4, markersize=8, label=str(group))
        ax.fill_between(sub["k"], sub["ci_lower"], sub["ci_upper"], alpha=0.18)
    ax.axhline(1.0, linestyle="--", color="#7F7F7F", linewidth=1.2)
    ax.set_title("Bootstrap robustness | neighborhood consistency")
    ax.set_xlabel("Top-k acoustic neighbors")
    ax.set_ylabel("Improvement ratio")
    ax.legend(loc="best")
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis="y")
    _save_fig(fig, Path(out_dir) / f"{stem}.png", dpi=plot_cfg.dpi)


def plot_variable_level_neighbor_heatmap(variable_df: pd.DataFrame, out_dir: str | Path, plot_cfg: PlotConfig, stem: str = "variable_level_neighbor_heatmap") -> None:
    if len(variable_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    ranked = (
        variable_df.groupby("variable", dropna=False)["improvement_ratio"]
        .mean()
        .sort_values(ascending=False)
        .head(int(plot_cfg.neighborhood_heatmap_top_n))
        .index.tolist()
    )
    plot_df = variable_df.loc[variable_df["variable"].isin(ranked)].copy()
    pivot = plot_df.pivot_table(index="variable", columns="k", values="improvement_ratio", aggfunc="mean")
    pivot = pivot.reindex(ranked)
    fig, ax = plt.subplots(figsize=(10.5, max(6.5, 0.48 * len(pivot))))
    cmap = sns.blend_palette(["#F1B44C", "#158F8C", "#2F6DB3", "#7B5AA6"], as_cmap=True)
    sns.heatmap(
        pivot,
        cmap=cmap,
        ax=ax,
        linewidths=plot_cfg.heatmap_grid_linewidth,
        linecolor=plot_cfg.heatmap_grid_color,
        annot=plot_cfg.heatmap_annot,
        fmt=".2f",
        cbar_kws={"label": "Improvement ratio"},
    )
    ax.set_title(f"Variable-level neighbor consistency | top {len(pivot)}")
    ax.set_xlabel("Top-k acoustic neighbors")
    ax.set_ylabel("Variable")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False)
    _save_fig(fig, Path(out_dir) / f"{stem}.png", dpi=plot_cfg.dpi)


def _plot_metric_heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    out_path: str | Path,
    plot_cfg: PlotConfig,
    top_n: int,
) -> None:
    if len(df) == 0 or value_col not in df.columns:
        return
    ranked = df.groupby("target", dropna=False)[value_col].max().sort_values(ascending=False).head(int(top_n)).index.tolist()
    plot_df = df.loc[df["target"].isin(ranked)].copy()
    pivot = plot_df.pivot_table(index="target", columns="k", values=value_col, aggfunc="mean")
    pivot = pivot.reindex(ranked)
    fig, ax = plt.subplots(figsize=(9.8, max(6.0, 0.5 * len(pivot))))
    cmap = sns.blend_palette(["#F7FBFF", "#9ECAE1", "#3182BD", "#08519C"], as_cmap=True)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=plot_cfg.heatmap_grid_linewidth,
        linecolor=plot_cfg.heatmap_grid_color,
        cbar_kws={"label": _metric_display_name(value_col)},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Top-k acoustic neighbors")
    ax.set_ylabel("")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False)
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_retrieval_summary(summary_df: pd.DataFrame, out_dir: str | Path, plot_cfg: PlotConfig, stem_prefix: str = "retrieval") -> None:
    if len(summary_df) == 0:
        return
    set_publication_plot_style(plot_cfg)

    cont_df = summary_df.loc[summary_df["target_type"].isin(["continuous", "ordinal"])].copy()
    if len(cont_df) > 0 and "spearman" in cont_df.columns:
        _plot_metric_heatmap(
            cont_df,
            value_col="spearman",
            title=f"{stem_prefix.replace('_', ' ').title()} | continuous / ordinal targets",
            out_path=Path(out_dir) / f"{stem_prefix}_continuous_heatmap.png",
            plot_cfg=plot_cfg,
            top_n=plot_cfg.bootstrap_retrieval_top_n if stem_prefix.startswith("bootstrap") else plot_cfg.retrieval_heatmap_top_n_continuous,
        )

    cls_df = summary_df.loc[summary_df["target_type"].isin(["binary", "categorical"])].copy()
    if len(cls_df) > 0 and "balanced_accuracy" in cls_df.columns:
        _plot_metric_heatmap(
            cls_df,
            value_col="balanced_accuracy",
            title=f"{stem_prefix.replace('_', ' ').title()} | categorical targets",
            out_path=Path(out_dir) / f"{stem_prefix}_categorical_heatmap.png",
            plot_cfg=plot_cfg,
            top_n=plot_cfg.retrieval_heatmap_top_n_categorical,
        )



def plot_baseline_comparison(baseline_df: pd.DataFrame, out_dir: str | Path, plot_cfg: PlotConfig) -> None:
    if len(baseline_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = baseline_df.loc[baseline_df["analysis"] == "global_alignment"].copy()
    if len(plot_df) == 0:
        return

    main_df = plot_df.loc[plot_df["group"] != "all"].copy()
    if len(main_df) > 0:
        fig, ax = plt.subplots(figsize=(11, 7.0))
        sns.barplot(data=main_df, x="group", y="spearman_r", hue="space", ax=ax)
        ax.set_title("Baseline-space comparison | global alignment")
        ax.set_xlabel("Clinical group")
        ax.set_ylabel("Spearman correlation")
        ax.legend(loc="best")
        _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis="y")
        _save_fig(fig, Path(out_dir) / "baseline_comparison_global_alignment.png", dpi=plot_cfg.dpi)

    all_df = plot_df.loc[plot_df["group"] == "all"].copy()
    if len(all_df) > 0:
        fig, ax = plt.subplots(figsize=(9.0, 6.7))
        sns.barplot(data=all_df, x="group", y="spearman_r", hue="space", ax=ax)
        ax.set_title("Baseline-space comparison | integrated clinical distance")
        ax.set_xlabel("Clinical group")
        ax.set_ylabel("Spearman correlation")
        ax.legend(loc="best")
        _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis="y")
        _save_fig(fig, Path(out_dir) / "baseline_comparison_global_alignment_all.png", dpi=plot_cfg.dpi)


def plot_discovery_validation_selection_frequency(summary_df: pd.DataFrame, out_dir: str | Path, plot_cfg: PlotConfig) -> None:
    if len(summary_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    disc = summary_df.loc[summary_df["subset"] == "discovery"].copy()
    if len(disc) == 0:
        return

    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    freq = disc["best_group_from_discovery"].value_counts().rename_axis("group").reset_index(name="count")
    sns.barplot(data=freq, x="group", y="count", ax=ax)
    ax.set_title("Discovery-validation | best-group selection frequency")
    ax.set_xlabel("Best group selected in discovery")
    ax.set_ylabel("Count across splits")
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis="y")
    _save_fig(fig, Path(out_dir) / "discovery_validation_best_group_frequency.png", dpi=plot_cfg.dpi)

    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    kfreq = disc["best_k_from_discovery"].value_counts().rename_axis("k").reset_index(name="count")
    sns.barplot(data=kfreq, x="k", y="count", ax=ax)
    ax.set_title("Discovery-validation | best-k selection frequency")
    ax.set_xlabel("Best k selected in discovery")
    ax.set_ylabel("Count across splits")
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis="y")
    _save_fig(fig, Path(out_dir) / "discovery_validation_best_k_frequency.png", dpi=plot_cfg.dpi)


def plot_discovery_validation_paired(summary_df: pd.DataFrame, out_dir: str | Path, plot_cfg: PlotConfig, metric_col: str, stem: str, title: str, ylabel: str) -> None:
    if len(summary_df) == 0 or metric_col not in summary_df.columns:
        return
    set_publication_plot_style(plot_cfg)
    pivot = summary_df.pivot(index="split_id", columns="subset", values=metric_col)
    if not {"discovery", "validation"}.issubset(set(pivot.columns)):
        return
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    for split_id, row in pivot.iterrows():
        ax.plot([0, 1], [row["discovery"], row["validation"]], color="#B7B7B7", linewidth=1.8, zorder=1)
        ax.scatter([0, 1], [row["discovery"], row["validation"]], s=95, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["discovery", "validation"])
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis="y")
    _save_fig(fig, Path(out_dir) / f"{stem}.png", dpi=plot_cfg.dpi)


def plot_positionwise_global_alignment_heatmap(
    summary_df: pd.DataFrame,
    out_dir: str | Path,
    plot_cfg: PlotConfig,
    metric_col: str = "spearman_r",
    stem: Optional[str] = None,
) -> None:
    if len(summary_df) == 0 or metric_col not in summary_df.columns:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = summary_df.copy()
    plot_df["position"] = plot_df["position"].astype(str)
    plot_df["group_rank"] = plot_df["group"].map({"function": 0, "structure": 1, "burden": 2, "all": 3}).fillna(99)
    order_groups = (
        plot_df[["group", "group_rank"]]
        .drop_duplicates()
        .sort_values(["group_rank", "group"])["group"]
        .astype(str)
        .tolist()
    )
    order_positions = (
        plot_df[["position"]]
        .drop_duplicates()["position"]
        .astype(str)
        .tolist()
    )
    pivot = plot_df.pivot_table(index="group", columns="position", values=metric_col, aggfunc="mean")
    pivot = pivot.reindex(index=order_groups, columns=order_positions)
    if pivot.empty:
        return
    v = np.nanmax(np.abs(pivot.to_numpy(dtype=np.float64)))
    if not np.isfinite(v) or v <= 0:
        v = 1.0
    fig, ax = plt.subplots(figsize=(8.6, max(4.8, 1.0 + 0.75 * len(pivot))))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(
        pivot,
        cmap=cmap,
        center=0.0,
        vmin=-v,
        vmax=v,
        annot=plot_cfg.heatmap_annot,
        fmt=".2f",
        linewidths=plot_cfg.heatmap_grid_linewidth,
        linecolor=plot_cfg.heatmap_grid_color,
        cbar_kws={"label": _metric_display_name(metric_col)},
        ax=ax,
    )
    ax.set_title(f"Position-wise global alignment | {_metric_display_name(metric_col)}")
    ax.set_xlabel("Auscultation position")
    ax.set_ylabel("Clinical group")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False)
    stem = stem or f"positionwise_global_alignment_{metric_col}"
    _save_fig(fig, Path(out_dir) / f"{stem}.png", dpi=plot_cfg.dpi)



def plot_positionwise_variablewise_heatmaps(
    summary_df: pd.DataFrame,
    out_dir: str | Path,
    plot_cfg: PlotConfig,
    metric_col: str = "spearman_r",
    top_n: int = 12,
    include_groups: Optional[Sequence[str]] = None,
    stem_prefix: str = "positionwise_variablewise_alignment",
) -> None:
    if len(summary_df) == 0 or metric_col not in summary_df.columns:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = summary_df.copy()
    plot_df["position"] = plot_df["position"].astype(str)
    if include_groups is None:
        group_list = [g for g in ["function", "structure", "burden", "all"] if g in set(plot_df["group"].astype(str))]
    else:
        group_list = [str(g) for g in include_groups if str(g) in set(plot_df["group"].astype(str))]

    for group_name in group_list:
        sub = plot_df.loc[plot_df["group"].astype(str) == str(group_name)].copy()
        if len(sub) == 0:
            continue
        ranked = (
            sub.groupby("variable", dropna=False)[metric_col]
            .apply(lambda s: float(np.nanmax(np.abs(pd.to_numeric(s, errors="coerce")))))
            .sort_values(ascending=False)
            .head(int(top_n))
            .index.tolist()
        )
        sub = sub.loc[sub["variable"].isin(ranked)].copy()
        if len(sub) == 0:
            continue
        pivot = sub.pivot_table(index="variable", columns="position", values=metric_col, aggfunc="mean")
        pivot = pivot.reindex(index=ranked)
        if pivot.empty:
            continue
        v = np.nanmax(np.abs(pivot.to_numpy(dtype=np.float64)))
        if not np.isfinite(v) or v <= 0:
            v = 1.0
        fig, ax = plt.subplots(figsize=(8.8, max(6.0, 0.48 * len(pivot))))
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        sns.heatmap(
            pivot,
            cmap=cmap,
            center=0.0,
            vmin=-v,
            vmax=v,
            annot=plot_cfg.heatmap_annot,
            fmt=".2f",
            linewidths=plot_cfg.heatmap_grid_linewidth,
            linecolor=plot_cfg.heatmap_grid_color,
            cbar_kws={"label": _metric_display_name(metric_col)},
            ax=ax,
        )
        ax.set_title(f"Position-wise variable alignment | {group_name}")
        ax.set_xlabel("Auscultation position")
        ax.set_ylabel("Clinical variable")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False)
        _save_fig(fig, Path(out_dir) / f"{stem_prefix}_{group_name}_{metric_col}.png", dpi=plot_cfg.dpi)



def run_positionwise_global_alignment_suite(
    data: ClinicalAlignmentData,
    registry: Sequence[VariableSpec],
    groups: Sequence[str],
    position_distance_dir: str | Path,
    positions: Sequence[str],
    file_pattern: str,
    n_perm: int,
    n_jobs: int,
    random_seed: int,
    out_dir: str | Path,
    plot_cfg: PlotConfig,
    top_n: int = 12,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print_banner("Running position-wise global and variable-level alignment")
    out_dir = _prepare_out_dir(out_dir, create=True)
    pos_mats = load_position_distance_matrices(
        position_distance_dir=position_distance_dir,
        positions=positions,
        expected_n=len(data.patient_order),
        file_pattern=file_pattern,
    )

    global_frames: List[pd.DataFrame] = []
    variable_frames: List[pd.DataFrame] = []
    meta_rows: List[Dict[str, Any]] = []

    for pos in [str(p) for p in positions]:
        if pos not in pos_mats:
            continue
        pos_dir = out_dir / f"position_{pos}"
        pos_dir.mkdir(parents=True, exist_ok=True)
        D_pos = pos_mats[pos]
        ga = run_global_distance_alignment(
            data=data,
            registry=registry,
            groups=groups,
            n_perm=n_perm,
            n_jobs=n_jobs,
            random_seed=random_seed,
            out_dir=pos_dir,
            plot_cfg=plot_cfg,
            acoustic_distance=D_pos,
            save_group_distances=False,
            save_outputs=True,
        ).copy()
        ga.insert(0, "position", pos)
        global_frames.append(ga)

        va = run_variablewise_global_alignment(
            data=data,
            registry=registry,
            groups=groups,
            n_perm=n_perm,
            n_jobs=n_jobs,
            random_seed=random_seed,
            out_dir=pos_dir,
            plot_cfg=plot_cfg,
            acoustic_distance=D_pos,
            top_n=top_n,
        ).copy()
        va.insert(0, "position", pos)
        variable_frames.append(va)

        meta_rows.append(
            {
                "position": pos,
                "n_patients": int(D_pos.shape[0]),
                "distance_min": float(np.nanmin(D_pos)),
                "distance_max": float(np.nanmax(D_pos)),
                "distance_mean_upper": float(np.nanmean(upper_triangle_vector(D_pos))),
            }
        )

    global_summary = pd.concat(global_frames, ignore_index=True) if len(global_frames) > 0 else pd.DataFrame()
    variable_summary = pd.concat(variable_frames, ignore_index=True) if len(variable_frames) > 0 else pd.DataFrame()
    position_meta = pd.DataFrame(meta_rows)

    global_summary.to_csv(out_dir / "positionwise_global_alignment_summary.csv", index=False, encoding="utf-8-sig")
    variable_summary.to_csv(out_dir / "positionwise_variablewise_global_alignment_summary.csv", index=False, encoding="utf-8-sig")
    position_meta.to_csv(out_dir / "positionwise_distance_matrix_meta.csv", index=False, encoding="utf-8-sig")

    if len(global_summary) > 0:
        plot_positionwise_global_alignment_heatmap(global_summary, out_dir=out_dir, plot_cfg=plot_cfg, metric_col="spearman_r", stem="positionwise_global_alignment_spearman_heatmap")
        plot_positionwise_global_alignment_heatmap(global_summary, out_dir=out_dir, plot_cfg=plot_cfg, metric_col="pearson_r", stem="positionwise_global_alignment_pearson_heatmap")
    if len(variable_summary) > 0:
        plot_positionwise_variablewise_heatmaps(
            variable_summary,
            out_dir=out_dir,
            plot_cfg=plot_cfg,
            metric_col="spearman_r",
            top_n=top_n,
            include_groups=[g for g in groups],
            stem_prefix="positionwise_variablewise_alignment",
        )

    with (out_dir / "positionwise_alignment_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "positions": [str(p) for p in positions],
                "position_distance_dir": str(position_distance_dir),
                "file_pattern": str(file_pattern),
                "n_positions_ran": int(len(global_frames)),
                "n_global_rows": int(len(global_summary)),
                "n_variable_rows": int(len(variable_summary)),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    log_done(
        "Position-wise alignment finished | "
        f"positions={len(global_frames)} | global_rows={len(global_summary)} | variable_rows={len(variable_summary)}"
    )
    return global_summary, variable_summary


# =============================================================================
# Global distance-to-distance alignment
# =============================================================================


def _permute_distance_corr_once(D1: np.ndarray, D2: np.ndarray, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(D2.shape[0])
    D2p = D2[np.ix_(perm, perm)]
    corr = _matrix_correlations(D1, D2p)
    return corr["pearson_r"], corr["spearman_r"]


def permutation_matrix_alignment_test(
    D1: np.ndarray,
    D2: np.ndarray,
    n_perm: int = 1000,
    n_jobs: int = 1,
    seed: int = 42,
) -> Dict[str, float]:
    obs = _matrix_correlations(D1, D2)
    obs_p = float(obs["pearson_r"])
    obs_s = float(obs["spearman_r"])
    if n_perm <= 0:
        return {"perm_p_pearson": np.nan, "perm_p_spearman": np.nan}
    log_info(f"Running permutation matrix alignment test | n_perm={n_perm} | n_jobs={n_jobs}")
    seeds = [int(seed + 10007 * i) for i in range(n_perm)]
    if int(n_jobs) == 1:
        results = [_permute_distance_corr_once(D1, D2, s) for s in seeds]
    else:
        results = Parallel(n_jobs=int(n_jobs), prefer="processes")(delayed(_permute_distance_corr_once)(D1, D2, s) for s in seeds)
    pearsons = np.array([r[0] for r in results], dtype=np.float64)
    spearmans = np.array([r[1] for r in results], dtype=np.float64)
    p_p = float((np.sum(np.abs(pearsons) >= abs(obs_p)) + 1) / (n_perm + 1))
    p_s = float((np.sum(np.abs(spearmans) >= abs(obs_s)) + 1) / (n_perm + 1))
    return {"perm_p_pearson": p_p, "perm_p_spearman": p_s}


def run_global_distance_alignment(
    data: ClinicalAlignmentData,
    registry: Sequence[VariableSpec],
    groups: Sequence[str],
    n_perm: int,
    n_jobs: int,
    random_seed: int,
    out_dir: str | Path | None,
    plot_cfg: PlotConfig,
    acoustic_distance: Optional[np.ndarray] = None,
    save_group_distances: bool = True,
    save_outputs: bool = True,
) -> pd.DataFrame:
    print_banner("Running global distance-to-distance clinical alignment")
    out_dir = _prepare_out_dir(out_dir, create=bool(save_outputs or save_group_distances))
    D_acoustic = _ensure_square_distance_matrix(acoustic_distance if acoustic_distance is not None else data.acoustic_distance, name="acoustic_distance")

    rows: List[Dict[str, Any]] = []
    for group in groups:
        if acoustic_distance is None and str(group) in data.clinical_distance_mats:
            D_clin = data.clinical_distance_mats[str(group)]
            variable_summary = pd.DataFrame()
        else:
            D_clin, variable_summary = compute_clinical_distance_matrix(data.clinical_plus_technical, registry, group_name=str(group))
        if save_group_distances and out_dir is not None:
            np.save(out_dir / f"clinical_distance_{group}.npy", D_clin)
            if len(variable_summary) > 0:
                variable_summary.to_csv(out_dir / f"clinical_distance_variables_{group}.csv", index=False, encoding="utf-8-sig")

        corr = _matrix_correlations(D_acoustic, D_clin)
        perm = permutation_matrix_alignment_test(D_acoustic, D_clin, n_perm=n_perm, n_jobs=n_jobs, seed=random_seed)
        rows.append({"group": str(group), **corr, **perm})
        if save_outputs and out_dir is not None:
            plot_global_alignment_scatter(D_acoustic, D_clin, str(group), out_dir=out_dir, plot_cfg=plot_cfg, seed=random_seed)

    summary_df = pd.DataFrame(rows).sort_values("spearman_r", ascending=False).reset_index(drop=True)
    if save_outputs and out_dir is not None:
        summary_df.to_csv(out_dir / "global_alignment_summary.csv", index=False, encoding="utf-8-sig")
        main_df = summary_df.loc[summary_df["group"] != "all"].copy()
        all_df = summary_df.loc[summary_df["group"] == "all"].copy()
        if len(main_df) > 0:
            plot_global_alignment_summary(main_df, out_dir=out_dir, plot_cfg=plot_cfg, stem="global_alignment_summary")
        if len(all_df) > 0:
            plot_global_alignment_summary(all_df, out_dir=out_dir, plot_cfg=plot_cfg, stem="global_alignment_summary_all")
    log_done(f"Global alignment finished | rows={len(summary_df)}")
    return summary_df


def run_variablewise_global_alignment(
    data: ClinicalAlignmentData,
    registry: Sequence[VariableSpec],
    groups: Sequence[str],
    n_perm: int,
    n_jobs: int,
    random_seed: int,
    out_dir: str | Path,
    plot_cfg: PlotConfig,
    acoustic_distance: Optional[np.ndarray] = None,
    top_n: int = 12,
) -> pd.DataFrame:
    print_banner("Running variable-wise global alignment")
    out_dir = _prepare_out_dir(out_dir, create=True)
    D_acoustic = _ensure_square_distance_matrix(acoustic_distance if acoustic_distance is not None else data.acoustic_distance, name="acoustic_distance")

    allowed_groups = [g for g in groups if str(g) != "all"]
    single_mats, info_df = build_all_single_variable_distance_matrices(data.clinical_plus_technical, registry, groups=allowed_groups)
    rows: List[Dict[str, Any]] = []
    for _, info in info_df.iterrows():
        var = str(info["clean_name"])
        D_var = single_mats[var]
        corr = _matrix_correlations(D_acoustic, D_var)
        perm = permutation_matrix_alignment_test(D_acoustic, D_var, n_perm=n_perm, n_jobs=n_jobs, seed=random_seed)
        rows.append({
            "variable": var,
            "raw_name": info["raw_name"],
            "group": info["group"],
            "var_type": info["var_type"],
            "coverage": info["coverage"],
            "effective_scale": info["effective_scale"],
            **corr,
            **perm,
        })
    summary_df = pd.DataFrame(rows).sort_values(["spearman_r", "pearson_r"], ascending=False).reset_index(drop=True)
    summary_df = _add_fdr_columns(
        summary_df,
        p_col="pearson_p",
        overall_col="pearson_q_overall",
        grouped_cols=["group"],
        grouped_col_name="pearson_q_within_group",
    )
    summary_df = _add_fdr_columns(
        summary_df,
        p_col="spearman_p",
        overall_col="spearman_q_overall",
        grouped_cols=["group"],
        grouped_col_name="spearman_q_within_group",
    )
    summary_df = _add_fdr_columns(
        summary_df,
        p_col="perm_p_pearson",
        overall_col="perm_q_pearson_overall",
        grouped_cols=["group"],
        grouped_col_name="perm_q_pearson_within_group",
    )
    summary_df = _add_fdr_columns(
        summary_df,
        p_col="perm_p_spearman",
        overall_col="perm_q_spearman_overall",
        grouped_cols=["group"],
        grouped_col_name="perm_q_spearman_within_group",
    )
    summary_df.to_csv(out_dir / "variablewise_global_alignment_summary.csv", index=False, encoding="utf-8-sig")
    info_df.to_csv(out_dir / "variablewise_distance_variables.csv", index=False, encoding="utf-8-sig")
    plot_variablewise_global_alignment_topn(summary_df, out_dir, plot_cfg, top_n=top_n)
    plot_variablewise_global_alignment_all(summary_df, out_dir, plot_cfg)
    log_done(f"Variable-wise global alignment finished | rows={len(summary_df)}")
    return summary_df


# =============================================================================
# Neighborhood consistency
# =============================================================================


def _random_neighbor_indices(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty((n, k), dtype=np.int32)
    all_idx = np.arange(n, dtype=np.int32)
    for i in range(n):
        candidates = np.concatenate([all_idx[:i], all_idx[i + 1 :]])
        out[i] = rng.choice(candidates, size=k, replace=False)
    return out


def _mean_neighbor_distance_from_matrix(D: np.ndarray, neighbor_idx: np.ndarray) -> np.ndarray:
    rows = np.arange(D.shape[0])[:, None]
    return np.mean(D[rows, neighbor_idx], axis=1)


def _variable_neighbor_difference(values: np.ndarray, spec: VariableSpec, neighbor_idx: np.ndarray) -> np.ndarray:
    x = np.asarray(values)
    n, _ = neighbor_idx.shape
    out = np.full(n, np.nan, dtype=np.float64)
    if spec.var_type in {"continuous", "ordinal"}:
        x_num = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=np.float64)
        for i in range(n):
            nbr = neighbor_idx[i]
            xi = x_num[i]
            xj = x_num[nbr]
            mask = np.isfinite(xj) & np.isfinite(xi)
            if np.any(mask):
                out[i] = float(np.mean(np.abs(xi - xj[mask])))
    elif spec.var_type in {"binary", "categorical"}:
        x_obj = pd.Series(x).astype("object").to_numpy()
        for i in range(n):
            nbr = neighbor_idx[i]
            xi = x_obj[i]
            xj = x_obj[nbr]
            mask = pd.notna(xj) & pd.notna(xi)
            if np.any(mask):
                out[i] = float(np.mean(xi != xj[mask]))
    return out


def run_neighborhood_consistency(
    data: ClinicalAlignmentData,
    registry: Sequence[VariableSpec],
    groups: Sequence[str],
    knn_list: Sequence[int],
    random_repeats: int,
    random_seed: int,
    out_dir: str | Path | None,
    plot_cfg: PlotConfig,
    acoustic_distance: Optional[np.ndarray] = None,
    save_outputs: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print_banner("Running neighborhood consistency analysis")
    out_dir = _prepare_out_dir(out_dir, create=bool(save_outputs))

    rng = np.random.default_rng(random_seed)
    D_acoustic = _ensure_square_distance_matrix(acoustic_distance if acoustic_distance is not None else data.acoustic_distance, name="acoustic_distance")
    n = D_acoustic.shape[0]

    group_rows: List[Dict[str, Any]] = []
    variable_rows: List[Dict[str, Any]] = []

    group_mats: Dict[str, np.ndarray] = {}
    for group in groups:
        if acoustic_distance is None and str(group) in data.clinical_distance_mats:
            group_mats[str(group)] = data.clinical_distance_mats[str(group)]
        else:
            Dg, _ = compute_clinical_distance_matrix(data.clinical_plus_technical, registry, group_name=str(group))
            group_mats[str(group)] = Dg

    for k in knn_list:
        log_info(f"Neighborhood consistency | k={k}")
        acoustic_knn = get_knn_from_distance(D_acoustic, k=int(k))
        random_neighbor_sets = [_random_neighbor_indices(n, int(k), rng) for _ in range(int(random_repeats))]

        for group in groups:
            Dg = group_mats[str(group)]
            observed_mean = float(np.nanmean(_mean_neighbor_distance_from_matrix(Dg, acoustic_knn)))
            random_means = np.array([float(np.nanmean(_mean_neighbor_distance_from_matrix(Dg, rand_idx))) for rand_idx in random_neighbor_sets], dtype=np.float64)
            random_mean = float(np.nanmean(random_means))
            improvement_ratio = float(random_mean / observed_mean) if observed_mean > 0 else np.nan
            perm_p = float((np.sum(random_means <= observed_mean) + 1) / (len(random_means) + 1))
            group_rows.append({"group": str(group), "k": int(k), "observed_mean_distance": observed_mean, "random_mean_distance": random_mean, "improvement_ratio": improvement_ratio, "perm_p": perm_p})

        var_specs = get_neighbor_variables(registry, group="all")
        for spec in var_specs:
            if spec.clean_name not in data.clinical_plus_technical.columns:
                continue
            values = data.clinical_plus_technical[spec.clean_name].to_numpy()
            observed = _variable_neighbor_difference(values, spec, acoustic_knn)
            observed_mean = float(np.nanmean(observed))
            random_means = np.array([float(np.nanmean(_variable_neighbor_difference(values, spec, rand_idx))) for rand_idx in random_neighbor_sets], dtype=np.float64)
            random_mean = float(np.nanmean(random_means))
            improvement_ratio = float(random_mean / observed_mean) if observed_mean > 0 else np.nan
            perm_p = float((np.sum(random_means <= observed_mean) + 1) / (len(random_means) + 1))
            variable_rows.append({"variable": spec.clean_name, "group": spec.group, "var_type": spec.var_type, "k": int(k), "observed_mean_difference": observed_mean, "random_mean_difference": random_mean, "improvement_ratio": improvement_ratio, "perm_p": perm_p})

    group_df = pd.DataFrame(group_rows).sort_values(["group", "k"]).reset_index(drop=True)
    variable_df = pd.DataFrame(variable_rows).sort_values(["group", "variable", "k"]).reset_index(drop=True)
    if len(variable_df) > 0:
        variable_df = _add_fdr_columns(
            variable_df,
            p_col="perm_p",
            overall_col="perm_q_overall_all_k",
            grouped_cols=["k"],
            grouped_col_name="perm_q_overall_within_k",
        )
        variable_df = _add_fdr_columns(
            variable_df,
            p_col="perm_p",
            overall_col="perm_q_overall_all_k_dup",
            grouped_cols=["group", "k"],
            grouped_col_name="perm_q_within_group_within_k",
        ).drop(columns=["perm_q_overall_all_k_dup"])
    if save_outputs and out_dir is not None:
        group_df.to_csv(out_dir / "neighborhood_group_summary.csv", index=False, encoding="utf-8-sig")
        variable_df.to_csv(out_dir / "neighborhood_variable_summary.csv", index=False, encoding="utf-8-sig")
        main_df = group_df.loc[group_df["group"] != "all"].copy()
        all_df = group_df.loc[group_df["group"] == "all"].copy()
        if len(main_df) > 0:
            plot_neighborhood_summary(main_df, out_dir, plot_cfg, stem="neighborhood_group_summary")
        if len(all_df) > 0:
            plot_neighborhood_summary(all_df, out_dir, plot_cfg, stem="neighborhood_group_summary_all")
        plot_variable_level_neighbor_heatmap(variable_df, out_dir, plot_cfg)
    log_done(f"Neighborhood consistency finished | group_rows={len(group_df)} | variable_rows={len(variable_df)}")
    return group_df, variable_df


# =============================================================================
# Retrieval-based validation
# =============================================================================


def _weighted_prediction(neighbor_values: np.ndarray, neighbor_distances: np.ndarray, spec: VariableSpec, kernel: str, sigma: str | float) -> Tuple[Any, float | None]:
    valid_mask = pd.notna(neighbor_values)
    if not np.any(valid_mask):
        return np.nan, np.nan
    vals = neighbor_values[valid_mask]
    d = neighbor_distances[valid_mask]
    w = _kernel_weights(d, kernel=kernel, sigma=sigma)

    if spec.var_type in {"continuous", "ordinal"}:
        vals_num = pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(dtype=np.float64)
        valid_num = np.isfinite(vals_num)
        if not np.any(valid_num):
            return np.nan, np.nan
        vals_num = vals_num[valid_num]
        w_num = w[valid_num]
        w_num = w_num / np.sum(w_num)
        return float(np.sum(w_num * vals_num)), np.nan

    if spec.var_type == "binary":
        vals_num = pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(dtype=np.float64)
        valid_num = np.isfinite(vals_num)
        if not np.any(valid_num):
            return np.nan, np.nan
        vals_num = vals_num[valid_num]
        w_num = w[valid_num]
        w_num = w_num / np.sum(w_num)
        p1 = float(np.sum(w_num * vals_num))
        pred = float(p1 >= 0.5)
        return pred, p1

    vals_obj = pd.Series(vals).astype("object").to_numpy()
    cats = pd.unique(vals_obj)
    scores = {cat: float(np.sum(w[vals_obj == cat])) for cat in cats}
    pred = max(scores.items(), key=lambda kv: kv[1])[0]
    return pred, np.nan


def _evaluate_retrieval_predictions(y_true: pd.Series, pred: pd.Series, prob: pd.Series, spec: VariableSpec) -> Dict[str, float]:
    yt = y_true.copy()
    yp = pred.copy()
    pp = prob.copy()
    mask = yt.notna() & yp.notna()

    if spec.var_type == "continuous":
        yt_num = pd.to_numeric(yt[mask], errors="coerce").astype(float)
        yp_num = pd.to_numeric(yp[mask], errors="coerce").astype(float)
        mask2 = yt_num.notna() & yp_num.notna()
        yt_num = yt_num[mask2].to_numpy(dtype=np.float64)
        yp_num = yp_num[mask2].to_numpy(dtype=np.float64)
        if len(yt_num) == 0:
            return {
                "n_eval": 0, "mae": np.nan, "rmse": np.nan, "spearman": np.nan,
                "nmae_iqr": np.nan, "within_tolerance_accuracy": np.nan, "tolerance_value": np.nan,
                "r2": np.nan, "explained_variance": np.nan, "c_index": np.nan,
            }
        yt_eval = _to_evaluation_scale(yt_num, spec)
        yp_eval = _to_evaluation_scale(yp_num, spec)
        mae = float(np.mean(np.abs(yt_eval - yp_eval)))
        rmse = float(np.sqrt(np.mean((yt_eval - yp_eval) ** 2)))
        rho = float(spearmanr(yt_eval, yp_eval).correlation) if len(yt_eval) >= 3 else np.nan
        denom = _robust_iqr_denom(yt_eval)
        nmae = float(mae / denom) if np.isfinite(denom) and denom > 0 else np.nan
        tol = _continuous_tolerance(spec, yt_eval)
        tol_acc = _within_tolerance_accuracy(yt_eval, yp_eval, tol)
        try:
            r2 = float(r2_score(yt_eval, yp_eval))
        except Exception:
            r2 = np.nan
        try:
            evs = float(explained_variance_score(yt_eval, yp_eval))
        except Exception:
            evs = np.nan
        cidx = _concordance_index(yt_eval, yp_eval)
        return {
            "n_eval": int(len(yt_eval)),
            "mae": mae,
            "rmse": rmse,
            "spearman": rho,
            "nmae_iqr": nmae,
            "within_tolerance_accuracy": tol_acc,
            "tolerance_value": float(tol),
            "r2": r2,
            "explained_variance": evs,
            "c_index": cidx,
        }

    if spec.var_type == "ordinal":
        yt_num = pd.to_numeric(yt[mask], errors="coerce").astype(float)
        yp_num = pd.to_numeric(yp[mask], errors="coerce").astype(float)
        mask2 = yt_num.notna() & yp_num.notna()
        yt_num = yt_num[mask2].to_numpy(dtype=np.float64)
        yp_num = yp_num[mask2].to_numpy(dtype=np.float64)
        if len(yt_num) == 0:
            return {
                "n_eval": 0, "mae": np.nan, "rmse": np.nan, "spearman": np.nan,
                "same_or_adjacent_accuracy": np.nan, "quadratic_weighted_kappa": np.nan, "c_index": np.nan,
            }
        mae = float(np.mean(np.abs(yt_num - yp_num)))
        rmse = float(np.sqrt(np.mean((yt_num - yp_num) ** 2)))
        rho = float(spearmanr(yt_num, yp_num).correlation) if len(yt_num) >= 3 else np.nan
        same_adj = _same_or_adjacent_accuracy(yt_num, yp_num)
        qwk = _quadratic_weighted_kappa(yt_num, yp_num)
        cidx = _concordance_index(yt_num, yp_num)
        return {
            "n_eval": int(len(yt_num)),
            "mae": mae,
            "rmse": rmse,
            "spearman": rho,
            "same_or_adjacent_accuracy": same_adj,
            "quadratic_weighted_kappa": qwk,
            "c_index": cidx,
        }

    if spec.var_type == "binary":
        yt_num = pd.to_numeric(yt[mask], errors="coerce").astype(float)
        yp_num = pd.to_numeric(yp[mask], errors="coerce").astype(float)
        pp_num = pd.to_numeric(pp[mask], errors="coerce").astype(float)
        mask2 = yt_num.notna() & yp_num.notna()
        yt_num = yt_num[mask2]
        yp_num = yp_num[mask2]
        pp_num = pp_num[mask2]
        if len(yt_num) == 0:
            return {"n_eval": 0, "balanced_accuracy": np.nan, "accuracy": np.nan, "auroc": np.nan}
        acc = float(np.mean(yt_num == yp_num))
        ba = float(balanced_accuracy_score(yt_num, yp_num))
        auroc = np.nan
        try:
            if len(np.unique(yt_num)) == 2 and np.all(np.isfinite(pp_num)):
                auroc = float(roc_auc_score(yt_num, pp_num))
        except Exception:
            pass
        return {"n_eval": int(len(yt_num)), "balanced_accuracy": ba, "accuracy": acc, "auroc": auroc}

    yt_obj = yt[mask].astype("object")
    yp_obj = yp[mask].astype("object")
    if len(yt_obj) == 0:
        return {"n_eval": 0, "accuracy": np.nan, "balanced_accuracy": np.nan}
    acc = float(np.mean(yt_obj == yp_obj))
    recalls = []
    for cls in pd.unique(yt_obj):
        cls_mask = yt_obj == cls
        recalls.append(float(np.mean(yp_obj[cls_mask] == cls)))
    ba = float(np.mean(recalls)) if len(recalls) else np.nan
    return {"n_eval": int(len(yt_obj)), "accuracy": acc, "balanced_accuracy": ba}


def _weighted_label_vote(labels: np.ndarray, weights: np.ndarray) -> Tuple[Any, float, Dict[Any, float]]:
    labels = np.asarray(labels)
    weights = np.asarray(weights, dtype=np.float64)
    valid = pd.notna(labels) & np.isfinite(weights)
    labels = labels[valid]
    weights = weights[valid]
    if len(labels) == 0:
        return np.nan, np.nan, {}
    if float(np.sum(weights)) <= 0:
        weights = np.ones_like(weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    scores: Dict[Any, float] = {}
    for lab in pd.unique(labels):
        scores[lab] = float(np.sum(weights[labels == lab]))
    pred = max(scores.items(), key=lambda kv: kv[1])[0]
    return pred, float(scores[pred]), scores


def _assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return out
    internal = np.asarray(edges[1:-1], dtype=np.float64)
    out[valid] = np.digitize(arr[valid], bins=internal, right=False).astype(np.float64)
    return out


def _build_continuous_binning(spec: VariableSpec, y_eval: np.ndarray) -> Tuple[np.ndarray, str, int]:
    arr = np.asarray(y_eval, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.array([-np.inf, np.inf], dtype=np.float64), "degenerate_1bin", 1

    fixed_edges = {
        "NTproBNP": [125.0, 300.0, 1000.0],
        "EF_Teich": [40.0, 50.0],
    }
    quantile_bins = {
        "LA_mm": 3,
        "LVEDD_mm": 3,
        "heart_rate": 3,
        "Hb": 3,
        "CRP": 3,
        "D_dimer": 3,
        "hsTnT": 3,
    }

    if spec.clean_name in fixed_edges:
        internal = np.asarray(fixed_edges[spec.clean_name], dtype=np.float64)
        scheme = f"fixed_{len(internal)+1}bin"
    else:
        q = int(quantile_bins.get(spec.clean_name, 3))
        qs = np.linspace(0.0, 1.0, q + 1)
        qvals = np.quantile(arr, qs)
        internal = np.unique(np.asarray(qvals[1:-1], dtype=np.float64))
        if len(internal) == 0:
            med = float(np.nanmedian(arr))
            internal = np.array([med], dtype=np.float64)
        scheme = f"quantile_{len(internal)+1}bin"

    edges = np.concatenate(([-np.inf], internal, [np.inf])).astype(np.float64)
    n_bins = int(len(edges) - 1)
    return edges, scheme, n_bins


def _same_or_adjacent_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    valid = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(valid):
        return np.nan
    return float(np.mean(np.abs(yt[valid] - yp[valid]) <= 1.0))


def _same_or_adjacent_bin_accuracy(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
    yt = np.asarray(y_true_bin, dtype=np.float64)
    yp = np.asarray(y_pred_bin, dtype=np.float64)
    valid = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(valid):
        return np.nan
    return float(np.mean(np.abs(yt[valid] - yp[valid]) <= 1.0))


def _macro_balanced_accuracy_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    valid = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[valid].astype(int)
    yp = yp[valid].astype(int)
    if len(yt) == 0:
        return np.nan
    recalls = []
    for cls in np.unique(yt):
        mask = yt == cls
        recalls.append(float(np.mean(yp[mask] == cls)))
    return float(np.mean(recalls)) if len(recalls) else np.nan


def _evaluate_continuous_coarse_profile(true_eval: np.ndarray, pred_bin: np.ndarray, pred_conf: np.ndarray, edges: np.ndarray, scheme: str) -> Dict[str, float]:
    true_bin = _assign_bins(true_eval, edges)
    yt = np.asarray(true_bin, dtype=np.float64)
    yp = np.asarray(pred_bin, dtype=np.float64)
    cf = np.asarray(pred_conf, dtype=np.float64)
    valid = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(valid):
        return {
            "coarse_n_eval": 0,
            "coarse_binning_scheme": scheme,
            "coarse_n_bins": int(len(edges) - 1),
            "coarse_exact_bin_accuracy": np.nan,
            "coarse_same_or_adjacent_bin_accuracy": np.nan,
            "coarse_balanced_accuracy": np.nan,
            "coarse_mean_confidence": np.nan,
        }
    yt = yt[valid]
    yp = yp[valid]
    cf = cf[valid]
    return {
        "coarse_n_eval": int(len(yt)),
        "coarse_binning_scheme": scheme,
        "coarse_n_bins": int(len(edges) - 1),
        "coarse_exact_bin_accuracy": float(np.mean(yt == yp)),
        "coarse_same_or_adjacent_bin_accuracy": _same_or_adjacent_bin_accuracy(yt, yp),
        "coarse_balanced_accuracy": _macro_balanced_accuracy_multiclass(yt, yp),
        "coarse_mean_confidence": float(np.nanmean(cf)) if len(cf) else np.nan,
    }


def _evaluate_ordinal_vote_profile(y_true: np.ndarray, pred_vote: np.ndarray, pred_conf: np.ndarray, same_w: np.ndarray, same_adj_w: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(pred_vote, dtype=np.float64)
    cf = np.asarray(pred_conf, dtype=np.float64)
    same_w = np.asarray(same_w, dtype=np.float64)
    same_adj_w = np.asarray(same_adj_w, dtype=np.float64)
    valid = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(valid):
        return {
            "ordinal_vote_n_eval": 0,
            "ordinal_vote_accuracy": np.nan,
            "ordinal_vote_same_or_adjacent_accuracy": np.nan,
            "ordinal_vote_qwk": np.nan,
            "ordinal_vote_c_index": np.nan,
            "ordinal_vote_mean_confidence": np.nan,
            "neighbor_same_grade_weight_mean": np.nan,
            "neighbor_same_or_adjacent_weight_mean": np.nan,
        }
    yt = yt[valid]
    yp = yp[valid]
    cf = cf[valid]
    same_w = same_w[valid]
    same_adj_w = same_adj_w[valid]
    return {
        "ordinal_vote_n_eval": int(len(yt)),
        "ordinal_vote_accuracy": float(np.mean(yt == yp)),
        "ordinal_vote_same_or_adjacent_accuracy": _same_or_adjacent_accuracy(yt, yp),
        "ordinal_vote_qwk": _quadratic_weighted_kappa(yt, yp),
        "ordinal_vote_c_index": _concordance_index(yt, yp),
        "ordinal_vote_mean_confidence": float(np.nanmean(cf)) if len(cf) else np.nan,
        "neighbor_same_grade_weight_mean": float(np.nanmean(same_w)) if len(same_w) else np.nan,
        "neighbor_same_or_adjacent_weight_mean": float(np.nanmean(same_adj_w)) if len(same_adj_w) else np.nan,
    }


def _build_confidence_rows(
    target: str,
    group: str,
    mode: str,
    k: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    coverage_grid: Sequence[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    cf = np.asarray(confidence, dtype=np.float64)
    valid = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(cf)
    if not np.any(valid):
        return rows
    yt = yt[valid]
    yp = yp[valid]
    cf = cf[valid]
    order = np.argsort(-cf)
    yt = yt[order]
    yp = yp[order]
    cf = cf[order]
    n = int(len(yt))
    for cov in coverage_grid:
        m = max(1, int(round(float(cov) * n)))
        yt_sub = yt[:m]
        yp_sub = yp[:m]
        cf_sub = cf[:m]
        row = {
            "target": target,
            "group": group,
            "mode": mode,
            "k": int(k),
            "coverage": float(cov),
            "n_eval": int(m),
            "min_confidence": float(np.min(cf_sub)),
            "mean_confidence": float(np.mean(cf_sub)),
        }
        if mode == "continuous_coarse":
            row.update({
                "exact_accuracy": float(np.mean(yt_sub == yp_sub)),
                "same_or_adjacent_accuracy": _same_or_adjacent_bin_accuracy(yt_sub, yp_sub),
                "balanced_accuracy": _macro_balanced_accuracy_multiclass(yt_sub, yp_sub),
            })
        elif mode == "ordinal_vote":
            row.update({
                "exact_accuracy": float(np.mean(yt_sub == yp_sub)),
                "same_or_adjacent_accuracy": _same_or_adjacent_accuracy(yt_sub, yp_sub),
                "quadratic_weighted_kappa": _quadratic_weighted_kappa(yt_sub, yp_sub),
                "c_index": _concordance_index(yt_sub, yp_sub),
            })
        rows.append(row)
    return rows


def run_retrieval_validation(
    data: ClinicalAlignmentData,
    registry: Sequence[VariableSpec],
    knn_list: Sequence[int],
    retrieval_kernel: str,
    retrieval_sigma: str | float,
    out_dir: str | Path | None,
    plot_cfg: PlotConfig,
    acoustic_distance: Optional[np.ndarray] = None,
    save_outputs: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print_banner("Running retrieval-based clinical validation")
    out_dir = _prepare_out_dir(out_dir, create=bool(save_outputs))

    D = _ensure_square_distance_matrix(acoustic_distance if acoustic_distance is not None else data.acoustic_distance, name="acoustic_distance")
    patient_ids = np.array(data.patient_order, dtype=object)
    target_specs = get_retrieval_targets(registry, group="all")

    summary_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    example_rows: List[Dict[str, Any]] = []
    neighbor_detail_rows: List[Dict[str, Any]] = []
    confidence_rows: List[Dict[str, Any]] = []

    coverage_grid = [0.25, 0.50, 0.75, 1.00]

    for spec in target_specs:
        if spec.clean_name not in data.clinical_plus_technical.columns:
            continue
        y = data.clinical_plus_technical[spec.clean_name]
        y_num_all = pd.to_numeric(y, errors="coerce").to_numpy(dtype=np.float64) if spec.var_type in {"continuous", "ordinal", "binary"} else None
        y_eval_all = _to_evaluation_scale(y_num_all, spec) if spec.var_type == "continuous" else None
        if spec.var_type == "continuous":
            bin_edges, bin_scheme, bin_count = _build_continuous_binning(spec, y_eval_all)
        else:
            bin_edges, bin_scheme, bin_count = None, None, None

        for k in knn_list:
            knn_idx = get_knn_from_distance(D, k=int(k))
            pred_values: List[Any] = []
            pred_probs: List[Any] = []
            top1_ids: List[Any] = []
            mean_topk_dist: List[float] = []

            coarse_true_bins: List[Any] = []
            coarse_pred_bins: List[Any] = []
            coarse_pred_conf: List[Any] = []

            ordinal_true_vals: List[Any] = []
            ordinal_vote_preds: List[Any] = []
            ordinal_vote_conf: List[Any] = []
            ordinal_same_w: List[Any] = []
            ordinal_same_adj_w: List[Any] = []

            for i in range(len(patient_ids)):
                nbr = knn_idx[i]
                nbr_values = y.iloc[nbr].to_numpy()
                nbr_dist = D[i, nbr]
                pred_val, pred_prob = _weighted_prediction(nbr_values, nbr_dist, spec, retrieval_kernel, retrieval_sigma)
                pred_values.append(pred_val)
                pred_probs.append(pred_prob)
                top1_ids.append(patient_ids[nbr[0]])
                mean_topk_dist.append(float(np.mean(nbr_dist)))
                w_detail = _kernel_weights(nbr_dist, kernel=retrieval_kernel, sigma=retrieval_sigma)
                for rank_j, (nbr_idx_j, dist_j, weight_j) in enumerate(zip(nbr, nbr_dist, w_detail), start=1):
                    neighbor_detail_rows.append(
                        {
                            "query_patient_id": patient_ids[i],
                            "target": spec.clean_name,
                            "target_type": spec.var_type,
                            "k": int(k),
                            "neighbor_rank": int(rank_j),
                            "neighbor_patient_id": patient_ids[nbr_idx_j],
                            "acoustic_distance": float(dist_j),
                            "neighbor_weight": float(weight_j),
                            "query_true_value": y.iloc[i],
                            "neighbor_true_value": y.iloc[nbr_idx_j],
                        }
                    )

                coarse_row = {
                    "true_bin": np.nan,
                    "predicted_bin": np.nan,
                    "predicted_bin_probability": np.nan,
                    "binning_scheme": np.nan,
                    "n_bins": np.nan,
                }
                ordinal_row = {
                    "predicted_vote_label": np.nan,
                    "predicted_vote_probability": np.nan,
                    "neighbor_same_grade_weight": np.nan,
                    "neighbor_same_or_adjacent_weight": np.nan,
                }

                if spec.var_type == "continuous":
                    true_eval = y_eval_all[i] if y_eval_all is not None else np.nan
                    true_bin = _assign_bins(np.array([true_eval], dtype=np.float64), bin_edges)[0] if np.isfinite(true_eval) else np.nan
                    nbr_num = pd.to_numeric(pd.Series(nbr_values), errors="coerce").to_numpy(dtype=np.float64)
                    nbr_eval = _to_evaluation_scale(nbr_num, spec)
                    nbr_bins = _assign_bins(nbr_eval, bin_edges)
                    pred_bin, pred_bin_prob, _ = _weighted_label_vote(nbr_bins, w_detail)
                    coarse_true_bins.append(true_bin)
                    coarse_pred_bins.append(pred_bin)
                    coarse_pred_conf.append(pred_bin_prob)
                    coarse_row = {
                        "true_bin": true_bin,
                        "predicted_bin": pred_bin,
                        "predicted_bin_probability": pred_bin_prob,
                        "binning_scheme": bin_scheme,
                        "n_bins": int(bin_count),
                    }

                if spec.var_type == "ordinal":
                    true_ord = pd.to_numeric(pd.Series([y.iloc[i]]), errors="coerce").iloc[0]
                    nbr_ord = pd.to_numeric(pd.Series(nbr_values), errors="coerce").to_numpy(dtype=np.float64)
                    pred_vote, pred_vote_prob, _ = _weighted_label_vote(nbr_ord, w_detail)
                    ordinal_true_vals.append(true_ord)
                    ordinal_vote_preds.append(pred_vote)
                    ordinal_vote_conf.append(pred_vote_prob)
                    if np.isfinite(true_ord):
                        valid_ord = np.isfinite(nbr_ord)
                        same_w = float(np.sum(w_detail[valid_ord & (nbr_ord == true_ord)])) if np.any(valid_ord) else np.nan
                        same_adj_w = float(np.sum(w_detail[valid_ord & (np.abs(nbr_ord - true_ord) <= 1.0)])) if np.any(valid_ord) else np.nan
                    else:
                        same_w = np.nan
                        same_adj_w = np.nan
                    ordinal_same_w.append(same_w)
                    ordinal_same_adj_w.append(same_adj_w)
                    ordinal_row = {
                        "predicted_vote_label": pred_vote,
                        "predicted_vote_probability": pred_vote_prob,
                        "neighbor_same_grade_weight": same_w,
                        "neighbor_same_or_adjacent_weight": same_adj_w,
                    }

                if int(k) == min(knn_list):
                    example_rows.append({
                        "patient_id": patient_ids[i],
                        "target": spec.clean_name,
                        "true_value": y.iloc[i],
                        "top1_neighbor_id": patient_ids[nbr[0]],
                        "topk_neighbor_ids": "|".join(map(str, patient_ids[nbr].tolist())),
                        "topk_neighbor_distances": "|".join([f"{d:.6f}" for d in nbr_dist.tolist()]),
                    })

                prediction_rows.append({
                    "patient_id": patient_ids[i],
                    "target": spec.clean_name,
                    "target_type": spec.var_type,
                    "k": int(k),
                    "true_value": y.iloc[i],
                    "predicted_value": pred_val,
                    "predicted_probability": pred_prob,
                    "top1_neighbor_id": patient_ids[nbr[0]],
                    "mean_topk_distance": float(np.mean(nbr_dist)),
                    **coarse_row,
                    **ordinal_row,
                })

            pred_series = pd.Series(pred_values, name="prediction")
            prob_series = pd.Series(pred_probs, name="probability")
            metrics = _evaluate_retrieval_predictions(y, pred_series, prob_series, spec)
            row = {"target": spec.clean_name, "group": spec.group, "target_type": spec.var_type, "k": int(k), **metrics}

            if spec.var_type == "continuous":
                coarse_metrics = _evaluate_continuous_coarse_profile(
                    true_eval=y_eval_all,
                    pred_bin=np.asarray(coarse_pred_bins, dtype=np.float64),
                    pred_conf=np.asarray(coarse_pred_conf, dtype=np.float64),
                    edges=bin_edges,
                    scheme=bin_scheme,
                )
                row.update(coarse_metrics)
                confidence_rows.extend(
                    _build_confidence_rows(
                        target=spec.clean_name,
                        group=spec.group,
                        mode="continuous_coarse",
                        k=int(k),
                        y_true=_assign_bins(y_eval_all, bin_edges),
                        y_pred=np.asarray(coarse_pred_bins, dtype=np.float64),
                        confidence=np.asarray(coarse_pred_conf, dtype=np.float64),
                        coverage_grid=coverage_grid,
                    )
                )

            if spec.var_type == "ordinal":
                ordinal_metrics = _evaluate_ordinal_vote_profile(
                    y_true=np.asarray(ordinal_true_vals, dtype=np.float64),
                    pred_vote=np.asarray(ordinal_vote_preds, dtype=np.float64),
                    pred_conf=np.asarray(ordinal_vote_conf, dtype=np.float64),
                    same_w=np.asarray(ordinal_same_w, dtype=np.float64),
                    same_adj_w=np.asarray(ordinal_same_adj_w, dtype=np.float64),
                )
                row.update(ordinal_metrics)
                confidence_rows.extend(
                    _build_confidence_rows(
                        target=spec.clean_name,
                        group=spec.group,
                        mode="ordinal_vote",
                        k=int(k),
                        y_true=np.asarray(ordinal_true_vals, dtype=np.float64),
                        y_pred=np.asarray(ordinal_vote_preds, dtype=np.float64),
                        confidence=np.asarray(ordinal_vote_conf, dtype=np.float64),
                        coverage_grid=coverage_grid,
                    )
                )

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(["target", "k"]).reset_index(drop=True)
    prediction_df = pd.DataFrame(prediction_rows)
    example_cases_df = pd.DataFrame(example_rows)
    neighbor_detail_df = pd.DataFrame(neighbor_detail_rows)
    confidence_df = pd.DataFrame(confidence_rows).sort_values(["target", "mode", "k", "coverage"]).reset_index(drop=True) if len(confidence_rows) else pd.DataFrame()

    if save_outputs and out_dir is not None:
        summary_df.to_csv(out_dir / "retrieval_summary.csv", index=False, encoding="utf-8-sig")
        prediction_df.to_csv(out_dir / "retrieval_predictions.csv", index=False, encoding="utf-8-sig")
        example_cases_df.to_csv(out_dir / "example_cases.csv", index=False, encoding="utf-8-sig")
        neighbor_detail_df.to_csv(out_dir / "retrieval_neighbor_details.csv", index=False, encoding="utf-8-sig")
        if len(confidence_df) > 0:
            confidence_df.to_csv(out_dir / "retrieval_confidence_summary.csv", index=False, encoding="utf-8-sig")
        plot_retrieval_summary(summary_df, out_dir, plot_cfg, stem_prefix="retrieval")
    log_done(f"Retrieval validation finished | summary_rows={len(summary_df)} | prediction_rows={len(prediction_df)}")
    return summary_df, prediction_df, example_cases_df


# =============================================================================
# Confounder adjustment
# =============================================================================


def _residualize_series(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    y_num = pd.to_numeric(y, errors="coerce").astype(float)
    X_num = X.apply(pd.to_numeric, errors="coerce").astype(float)
    mask = y_num.notna()
    for col in X_num.columns:
        mask &= X_num[col].notna()
    if mask.sum() < 10:
        return y_num
    yv = y_num[mask].to_numpy(dtype=np.float64)
    Xv = X_num.loc[mask].to_numpy(dtype=np.float64)
    Xv = np.column_stack([np.ones(len(Xv), dtype=np.float64), Xv])
    try:
        beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
        resid = yv - Xv @ beta
        out = pd.Series(np.nan, index=y.index, dtype=float)
        out.loc[mask] = resid
        return out
    except Exception:
        return y_num


def build_adjusted_clinical_table(
    data: ClinicalAlignmentData,
    registry: Sequence[VariableSpec],
    adjust_covariates: Sequence[str],
    technical_covariates: Sequence[str],
) -> pd.DataFrame:
    print_banner("Building confounder-adjusted clinical table")
    df = data.clinical_plus_technical.copy()
    covs = [c for c in list(adjust_covariates) + list(technical_covariates) if c in df.columns]
    if len(covs) == 0:
        log_warn("No adjustment covariates found; returning original table.")
        return df
    X = df[covs].copy()
    for spec in registry:
        if not spec.use_for_distance:
            continue
        if spec.clean_name not in df.columns:
            continue
        if spec.var_type in {"continuous", "ordinal"}:
            df[spec.clean_name] = _residualize_series(df[spec.clean_name], X)
    log_done(f"Adjusted clinical table built using covariates: {covs}")
    return df


def run_adjusted_global_alignment(
    data: ClinicalAlignmentData,
    registry: Sequence[VariableSpec],
    groups: Sequence[str],
    adjust_covariates: Sequence[str],
    technical_covariates: Sequence[str],
    n_perm: int,
    n_jobs: int,
    random_seed: int,
    out_dir: str | Path,
    plot_cfg: PlotConfig,
) -> pd.DataFrame:
    print_banner("Running adjusted global alignment")
    out_dir = _prepare_out_dir(out_dir, create=True)
    adjusted_df = build_adjusted_clinical_table(data, registry, adjust_covariates, technical_covariates)
    rows: List[Dict[str, Any]] = []
    for group in groups:
        D_clin, vars_df = compute_clinical_distance_matrix(adjusted_df, registry, group_name=str(group))
        np.save(out_dir / f"clinical_distance_adjusted_{group}.npy", D_clin)
        vars_df.to_csv(out_dir / f"clinical_distance_adjusted_variables_{group}.csv", index=False, encoding="utf-8-sig")
        corr = _matrix_correlations(data.acoustic_distance, D_clin)
        perm = permutation_matrix_alignment_test(data.acoustic_distance, D_clin, n_perm=n_perm, n_jobs=n_jobs, seed=random_seed)
        rows.append({"group": str(group), **corr, **perm})
    summary_df = pd.DataFrame(rows).sort_values("spearman_r", ascending=False).reset_index(drop=True)
    summary_df.to_csv(out_dir / "adjusted_global_alignment_summary.csv", index=False, encoding="utf-8-sig")
    main_df = summary_df.loc[summary_df["group"] != "all"].copy()
    all_df = summary_df.loc[summary_df["group"] == "all"].copy()
    if len(main_df) > 0:
        plot_global_alignment_summary(main_df, out_dir=out_dir, plot_cfg=plot_cfg, stem="adjusted_global_alignment_summary")
    if len(all_df) > 0:
        plot_global_alignment_summary(all_df, out_dir=out_dir, plot_cfg=plot_cfg, stem="adjusted_global_alignment_summary_all")
    return summary_df


# =============================================================================
# Bootstrap / subsampling confidence intervals
# =============================================================================


def _bootstrap_global_once(data: ClinicalAlignmentData, registry: Sequence[VariableSpec], groups: Sequence[str], seed: int) -> pd.DataFrame:
    idx = _get_patient_level_bootstrap_indices(len(data.patient_order), seed)
    D_ac = _subset_distance_matrix(data.acoustic_distance, idx)
    df_sub = _subset_dataframe(data.clinical_plus_technical, idx)
    rows = []
    for group in groups:
        D_clin, _ = compute_clinical_distance_matrix(df_sub, registry, group_name=str(group))
        corr = _matrix_correlations(D_ac, D_clin)
        rows.append({"group": str(group), "pearson_r": corr["pearson_r"], "spearman_r": corr["spearman_r"]})
    return pd.DataFrame(rows)


def bootstrap_global_alignment(
    data: ClinicalAlignmentData,
    registry: Sequence[VariableSpec],
    groups: Sequence[str],
    repeats: int,
    ci: float,
    n_jobs: int,
    seed: int,
    out_dir: str | Path,
    plot_cfg: Optional[PlotConfig] = None,
) -> pd.DataFrame:
    print_banner("Bootstrapping global alignment")
    out_dir = _prepare_out_dir(out_dir, create=True)
    seeds = [int(seed + 1009 * i) for i in range(int(repeats))]
    if int(n_jobs) == 1:
        reps = [_bootstrap_global_once(data, registry, groups, s) for s in seeds]
    else:
        reps = Parallel(n_jobs=int(n_jobs), prefer="processes")(delayed(_bootstrap_global_once)(data, registry, groups, s) for s in seeds)
    raw = pd.concat([df.assign(repeat=i) for i, df in enumerate(reps)], ignore_index=True)
    rows = []
    for group in groups:
        sub = raw.loc[raw["group"] == str(group)]
        for metric in ["pearson_r", "spearman_r"]:
            lo, hi = _quantile_interval(sub[metric], ci=ci)
            rows.append({"group": str(group), "metric": metric, "estimate_mean": float(np.nanmean(sub[metric])), "ci_lower": lo, "ci_upper": hi, "n_repeats": int(len(sub))})
    summary = pd.DataFrame(rows)
    raw.to_csv(out_dir / "bootstrap_global_alignment_raw.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "bootstrap_global_alignment_summary.csv", index=False, encoding="utf-8-sig")
    if plot_cfg is not None:
        plot_global_alignment_forest(summary, out_dir, plot_cfg, stem="bootstrap_global_alignment_forest")
        plot_bootstrap_global_distribution(raw, out_dir, plot_cfg, stem="bootstrap_global_alignment_distribution")
    return summary


def _bootstrap_neighborhood_once(
    data: ClinicalAlignmentData,
    registry: Sequence[VariableSpec],
    groups: Sequence[str],
    knn_list: Sequence[int],
    random_repeats: int,
    seed: int,
    subsample_fraction: float,
) -> pd.DataFrame:
    idx = _get_patient_level_subsample_indices(len(data.patient_order), seed, fraction=subsample_fraction)
    D_ac = _subset_distance_matrix(data.acoustic_distance, idx)
    sub = ClinicalAlignmentData(
        patient_order=[data.patient_order[i] for i in idx],
        acoustic_distance=D_ac,
        clinical_raw=_subset_dataframe(data.clinical_raw, idx),
        clinical_clean=_subset_dataframe(data.clinical_clean, idx),
        technical_covariates=_subset_dataframe(data.technical_covariates, idx),
        clinical_plus_technical=_subset_dataframe(data.clinical_plus_technical, idx),
        clinical_plus_technical_strata=_subset_dataframe(data.clinical_plus_technical_strata, idx),
        missingness_df=data.missingness_df,
        registry_df=data.registry_df,
        meta=data.meta,
        clinical_distance_mats={},
    )
    group_df, _ = run_neighborhood_consistency(sub, registry, groups, knn_list, random_repeats, seed, out_dir=None, plot_cfg=PlotConfig(), acoustic_distance=D_ac, save_outputs=False)
    return group_df


def bootstrap_neighborhood_consistency(
    data: ClinicalAlignmentData,
    registry: Sequence[VariableSpec],
    groups: Sequence[str],
    knn_list: Sequence[int],
    random_repeats: int,
    repeats: int,
    ci: float,
    n_jobs: int,
    seed: int,
    out_dir: str | Path,
    plot_cfg: Optional[PlotConfig] = None,
    subsample_fraction: float = 0.8,
) -> pd.DataFrame:
    print_banner("Bootstrapping neighborhood consistency")
    out_dir = _prepare_out_dir(out_dir, create=True)
    seeds = [int(seed + 2003 * i) for i in range(int(repeats))]
    if int(n_jobs) == 1:
        reps = [_bootstrap_neighborhood_once(data, registry, groups, knn_list, random_repeats, s, subsample_fraction) for s in seeds]
    else:
        reps = Parallel(n_jobs=int(n_jobs), prefer="processes")(delayed(_bootstrap_neighborhood_once)(data, registry, groups, knn_list, random_repeats, s, subsample_fraction) for s in seeds)
    raw = pd.concat([df.assign(repeat=i) for i, df in enumerate(reps)], ignore_index=True)
    rows = []
    for (group, k), sub in raw.groupby(["group", "k"], dropna=False):
        lo, hi = _quantile_interval(sub["improvement_ratio"], ci=ci)
        rows.append({"group": group, "k": int(k), "metric": "improvement_ratio", "estimate_mean": float(np.nanmean(sub["improvement_ratio"])), "ci_lower": lo, "ci_upper": hi, "n_repeats": int(len(sub))})
    summary = pd.DataFrame(rows)
    raw.to_csv(out_dir / "bootstrap_neighborhood_raw.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "bootstrap_neighborhood_summary.csv", index=False, encoding="utf-8-sig")
    if plot_cfg is not None:
        plot_bootstrap_neighborhood_summary(summary, out_dir, plot_cfg, stem="bootstrap_neighborhood_summary")
    return summary


def _bootstrap_retrieval_once(
    data: ClinicalAlignmentData,
    registry: Sequence[VariableSpec],
    knn_list: Sequence[int],
    kernel: str,
    sigma: str | float,
    seed: int,
    subsample_fraction: float,
) -> pd.DataFrame:
    idx = _get_patient_level_subsample_indices(len(data.patient_order), seed, fraction=subsample_fraction)
    D_ac = _subset_distance_matrix(data.acoustic_distance, idx)
    sub = ClinicalAlignmentData(
        patient_order=[data.patient_order[i] for i in idx],
        acoustic_distance=D_ac,
        clinical_raw=_subset_dataframe(data.clinical_raw, idx),
        clinical_clean=_subset_dataframe(data.clinical_clean, idx),
        technical_covariates=_subset_dataframe(data.technical_covariates, idx),
        clinical_plus_technical=_subset_dataframe(data.clinical_plus_technical, idx),
        clinical_plus_technical_strata=_subset_dataframe(data.clinical_plus_technical_strata, idx),
        missingness_df=data.missingness_df,
        registry_df=data.registry_df,
        meta=data.meta,
        clinical_distance_mats={},
    )
    summary_df, _, _ = run_retrieval_validation(sub, registry, knn_list, kernel, sigma, out_dir=None, plot_cfg=PlotConfig(), acoustic_distance=D_ac, save_outputs=False)
    return summary_df


def bootstrap_retrieval_validation(
    data: ClinicalAlignmentData,
    registry: Sequence[VariableSpec],
    knn_list: Sequence[int],
    kernel: str,
    sigma: str | float,
    repeats: int,
    ci: float,
    n_jobs: int,
    seed: int,
    out_dir: str | Path,
    plot_cfg: Optional[PlotConfig] = None,
    subsample_fraction: float = 0.8,
) -> pd.DataFrame:
    print_banner("Bootstrapping retrieval validation")
    out_dir = _prepare_out_dir(out_dir, create=True)
    seeds = [int(seed + 3001 * i) for i in range(int(repeats))]
    if int(n_jobs) == 1:
        reps = [_bootstrap_retrieval_once(data, registry, knn_list, kernel, sigma, s, subsample_fraction) for s in seeds]
    else:
        reps = Parallel(n_jobs=int(n_jobs), prefer="processes")(delayed(_bootstrap_retrieval_once)(data, registry, knn_list, kernel, sigma, s, subsample_fraction) for s in seeds)
    raw = pd.concat([df.assign(repeat=i) for i, df in enumerate(reps)], ignore_index=True)
    rows = []
    for keys, sub in raw.groupby(["target", "target_type", "k"], dropna=False):
        target, target_type, k = keys
        metric_col = "spearman" if target_type in {"continuous", "ordinal"} else "balanced_accuracy"
        if metric_col not in sub.columns:
            continue
        lo, hi = _quantile_interval(sub[metric_col], ci=ci)
        rows.append({"target": target, "target_type": target_type, "k": int(k), "metric": metric_col, "estimate_mean": float(np.nanmean(sub[metric_col])), "ci_lower": lo, "ci_upper": hi, "n_repeats": int(len(sub))})
    summary = pd.DataFrame(rows)
    raw.to_csv(out_dir / "bootstrap_retrieval_raw.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "bootstrap_retrieval_summary.csv", index=False, encoding="utf-8-sig")
    if plot_cfg is not None:
        cont = summary.loc[summary["metric"] == "spearman"].rename(columns={"estimate_mean": "spearman"})
        cls = summary.loc[summary["metric"] == "balanced_accuracy"].rename(columns={"estimate_mean": "balanced_accuracy"})
        combo = pd.concat([cont, cls], ignore_index=True, sort=False)
        if len(combo) > 0:
            plot_retrieval_summary(combo, out_dir, plot_cfg, stem_prefix="bootstrap_retrieval")
    return summary


# =============================================================================
# Baseline-space comparison
# =============================================================================


def load_saved_baseline_distance_matrices(baseline_dir: str | Path) -> Dict[str, np.ndarray]:
    baseline_dir = Path(baseline_dir)
    mats: Dict[str, np.ndarray] = {}
    for fp in sorted(baseline_dir.glob("distance_*.npy")):
        name = fp.stem.replace("distance_", "", 1)
        mats[name] = np.load(fp)
    if len(mats) == 0:
        raise FileNotFoundError(f"No baseline distance matrices found in: {baseline_dir}")
    log_done(f"Loaded saved baseline spaces: {list(mats.keys())}")
    return mats


def run_baseline_comparison(
    data: ClinicalAlignmentData,
    registry: Sequence[VariableSpec],
    baseline_mats: Mapping[str, np.ndarray],
    groups: Sequence[str],
    knn_list: Sequence[int],
    retrieval_kernel: str,
    retrieval_sigma: str | float,
    out_dir: str | Path,
    plot_cfg: PlotConfig,
) -> pd.DataFrame:
    print_banner("Running baseline-space comparison")
    out_dir = _prepare_out_dir(out_dir, create=True)

    rows: List[Dict[str, Any]] = []
    for space, D_space in baseline_mats.items():
        log_info(f"Baseline comparison | space={space}")

        ga = run_global_distance_alignment(
            data=data,
            registry=registry,
            groups=groups,
            n_perm=0,
            n_jobs=1,
            random_seed=42,
            out_dir=None,
            plot_cfg=plot_cfg,
            acoustic_distance=D_space,
            save_group_distances=False,
            save_outputs=False,
        )
        for _, r in ga.iterrows():
            rows.append({"space": space, "analysis": "global_alignment", **r.to_dict()})

        ng_group, _ = run_neighborhood_consistency(
            data=data,
            registry=registry,
            groups=groups,
            knn_list=knn_list,
            random_repeats=100,
            random_seed=42,
            out_dir=None,
            plot_cfg=plot_cfg,
            acoustic_distance=D_space,
            save_outputs=False,
        )
        for _, r in ng_group.iterrows():
            rows.append({"space": space, "analysis": "neighborhood", **r.to_dict()})

        rt, _, _ = run_retrieval_validation(
            data=data,
            registry=registry,
            knn_list=knn_list,
            retrieval_kernel=retrieval_kernel,
            retrieval_sigma=retrieval_sigma,
            out_dir=None,
            plot_cfg=plot_cfg,
            acoustic_distance=D_space,
            save_outputs=False,
        )
        for _, r in rt.iterrows():
            rows.append({"space": space, "analysis": "retrieval", **r.to_dict()})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / "baseline_comparison_summary.csv", index=False, encoding="utf-8-sig")
    plot_baseline_comparison(out_df, out_dir, plot_cfg)
    return out_df


# =============================================================================
# Discovery / validation split analysis
# =============================================================================


def run_discovery_validation_alignment(
    data: ClinicalAlignmentData,
    registry: Sequence[VariableSpec],
    groups: Sequence[str],
    knn_list: Sequence[int],
    retrieval_kernel: str,
    retrieval_sigma: str | float,
    discovery_fraction: float,
    split_repeats: int,
    random_seed: int,
    out_dir: str | Path,
    plot_cfg: Optional[PlotConfig] = None,
) -> pd.DataFrame:
    print_banner("Running discovery / validation robustness analysis")
    out_dir = _prepare_out_dir(out_dir, create=True)
    rng = np.random.default_rng(random_seed)
    n = len(data.patient_order)
    rows: List[Dict[str, Any]] = []

    for split_id in range(int(split_repeats)):
        perm = rng.permutation(n)
        n_disc = int(round(float(discovery_fraction) * n))
        disc_idx = np.sort(perm[:n_disc])
        val_idx = np.sort(perm[n_disc:])
        chosen: Dict[str, Any] = {}

        for subset_name, idx in [("discovery", disc_idx), ("validation", val_idx)]:
            D_sub = _subset_distance_matrix(data.acoustic_distance, idx)
            sub = ClinicalAlignmentData(
                patient_order=[data.patient_order[i] for i in idx],
                acoustic_distance=D_sub,
                clinical_raw=_subset_dataframe(data.clinical_raw, idx),
                clinical_clean=_subset_dataframe(data.clinical_clean, idx),
                technical_covariates=_subset_dataframe(data.technical_covariates, idx),
                clinical_plus_technical=_subset_dataframe(data.clinical_plus_technical, idx),
                clinical_plus_technical_strata=_subset_dataframe(data.clinical_plus_technical_strata, idx),
                missingness_df=data.missingness_df,
                registry_df=data.registry_df,
                meta=data.meta,
                clinical_distance_mats={},
            )
            ga = run_global_distance_alignment(sub, registry, groups, n_perm=0, n_jobs=1, random_seed=random_seed + split_id, out_dir=None, plot_cfg=PlotConfig(), acoustic_distance=D_sub, save_group_distances=False, save_outputs=False)
            ng_group, _ = run_neighborhood_consistency(sub, registry, groups, knn_list, random_repeats=100, random_seed=random_seed + split_id, out_dir=None, plot_cfg=PlotConfig(), acoustic_distance=D_sub, save_outputs=False)
            rt, _, _ = run_retrieval_validation(sub, registry, [int(np.median(list(knn_list)))], retrieval_kernel, retrieval_sigma, out_dir=None, plot_cfg=PlotConfig(), acoustic_distance=D_sub, save_outputs=False)

            if subset_name == "discovery":
                best_group = str(ga.sort_values("spearman_r", ascending=False).iloc[0]["group"])
                best_k = int(ng_group.loc[ng_group["group"] == best_group].sort_values("improvement_ratio", ascending=False).iloc[0]["k"])
                chosen = {"best_group": best_group, "best_k": best_k}

            group_to_use = chosen["best_group"] if subset_name == "validation" else best_group
            k_to_use = chosen["best_k"] if subset_name == "validation" else best_k
            rows.append({
                "split_id": split_id,
                "subset": subset_name,
                "best_group_from_discovery": group_to_use,
                "best_k_from_discovery": k_to_use,
                "global_spearman_best_group": float(ga.loc[ga["group"] == group_to_use, "spearman_r"].iloc[0]),
                "neighborhood_improvement_best_k": float(ng_group.loc[(ng_group["group"] == group_to_use) & (ng_group["k"] == k_to_use), "improvement_ratio"].iloc[0]),
                "retrieval_mean_spearman": float(np.nanmean(rt["spearman"])) if "spearman" in rt.columns else np.nan,
                "n_patients": int(len(idx)),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "discovery_validation_summary.csv", index=False, encoding="utf-8-sig")
    if plot_cfg is not None:
        plot_discovery_validation_selection_frequency(summary, out_dir, plot_cfg)
        plot_discovery_validation_paired(summary, out_dir, plot_cfg, metric_col="global_spearman_best_group", stem="discovery_validation_global_spearman", title="Discovery-validation | global alignment", ylabel="Spearman correlation")
        plot_discovery_validation_paired(summary, out_dir, plot_cfg, metric_col="neighborhood_improvement_best_k", stem="discovery_validation_neighborhood_improvement", title="Discovery-validation | neighborhood consistency", ylabel="Improvement ratio")
        plot_discovery_validation_paired(summary, out_dir, plot_cfg, metric_col="retrieval_mean_spearman", stem="discovery_validation_retrieval_mean_spearman", title="Discovery-validation | retrieval auxiliary score", ylabel="Mean retrieval Spearman")
    return summary


__all__ = [
    "run_global_distance_alignment",
    "run_variablewise_global_alignment",
    "run_neighborhood_consistency",
    "run_retrieval_validation",
    "run_adjusted_global_alignment",
    "bootstrap_global_alignment",
    "bootstrap_neighborhood_consistency",
    "bootstrap_retrieval_validation",
    "load_saved_baseline_distance_matrices",
    "run_baseline_comparison",
    "run_discovery_validation_alignment",
    "load_position_distance_matrices",
    "build_leave_one_out_position_distance_matrices",
    "run_position_leave_one_out_contribution",
    "plot_positionwise_global_alignment_heatmap",
    "plot_positionwise_variablewise_heatmaps",
    "run_positionwise_global_alignment_suite",
]
