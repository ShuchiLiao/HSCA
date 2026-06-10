#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot external validation results for internally fitted BEATs CCA axes.

This script is the plotting companion of
Scripts/Validation/alignment/run_apply_internal_cca_to_external.py.
It follows the plotting style of Scripts/Alignment/plot_cca_results.py in the
HSCA repository: Agg backend, seaborn white theme when available, Arial-like
publication fonts, clean scatter/box/forest/histogram figures, and PNG output.

Inputs are the CSV tables produced by run_apply_internal_cca_to_external.py:
    tables/external_axis_scores.csv
    tables/external_main_alignment_summary.csv
    tables/external_endpoint_auroc_summary.csv
    tables/external_permutation_null.csv

The script only regenerates figures. It does not refit CCA, recompute AUROC,
or change any external validation statistics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

from scipy import stats

try:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
    HAS_SKLEARN_METRICS = True
except Exception:
    HAS_SKLEARN_METRICS = False



AXIS_X = "cca_acoustic_axis1"
AXIS_Y = "cca_clinical_axis1"

CLINICAL_PROFILE_VARS = ["EF_Teich", "NTproBNP", "NYHA", "LVEDD_mm"]
ENDPOINT_ORDER = ["EF_lt_40", "NTproBNP_ge_900", "LVEDD_dilated", "NYHA_ge_3"]

PRETTY_LABELS: Dict[str, str] = {
    "EF_Teich": "EF",
    "NTproBNP": "NT-proBNP",
    "NTproBNP_raw": "NT-proBNP",
    "NYHA": "NYHA",
    "LVEDD_mm": "LVEDD",
    "EF_lt_40": "EF <40%",
    "NYHA_ge_3": "NYHA ≥3",
    "NTproBNP_ge_900": "NT-proBNP ≥900 pg/mL",
    "LVEDD_dilated": "LVEDD dilation",
}

# Color choices are kept close to Scripts/Alignment/plot_cca_results.py.
VAR_COLORS: Dict[str, str] = {
    "EF_Teich": "#009E73",       # green
    "NTproBNP": "#CC79A7",      # magenta
    "NYHA": "#56B4E9",          # sky blue
    "LVEDD_mm": "#0072B2",      # blue
}
ENDPOINT_COLORS: Dict[str, str] = {
    "EF_lt_40": "#009E73",
    "NYHA_ge_3": "#56B4E9",
    "NTproBNP_ge_900": "#CC79A7",
    "LVEDD_dilated": "#0072B2",
}


def now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_plotting() -> None:
    """Use the same general publication style as the main CCA plotting script."""
    if HAS_SEABORN:
        sns.set_theme(style="white", context="talk")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["font.size"] = 21
    plt.rcParams["axes.labelsize"] = 21
    plt.rcParams["xtick.labelsize"] = 21
    plt.rcParams["ytick.labelsize"] = 21
    plt.rcParams["legend.fontsize"] = 18
    plt.rcParams["axes.titlesize"] = 21
    plt.rcParams["figure.titlesize"] = 21
    plt.rcParams["axes.spines.top"] = True
    plt.rcParams["axes.spines.right"] = True
    plt.rcParams["axes.grid"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300


def savefig(path: Path) -> None:
    """Save one figure and close it. Titles are cleared for manuscript-style panels."""
    fig = plt.gcf()
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")
    for ax in fig.axes:
        ax.set_title("")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    log(f"Saved figure: {path}")


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        log(f"Skip missing table: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
        log(f"Loaded table: {path.name}, shape={df.shape}")
        return df
    except Exception as e:
        log(f"Failed to read {path}: {e}")
        return pd.DataFrame()


def p_text(p: float) -> str:
    if not np.isfinite(p):
        return "p=NA"
    if p < 1e-4:
        return "p<1e-4"
    if p < 0.001:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


def safe_spearman(x, y) -> Tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 5 or np.nanstd(x[mask]) < 1e-12 or np.nanstd(y[mask]) < 1e-12:
        return np.nan, np.nan, n
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p), n


def unit_label(var: str, values: Optional[pd.Series] = None) -> str:
    if var == "EF_Teich":
        return "EF (%)"
    if var == "NTproBNP":
        if values is not None:
            arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
            mx = np.nanmax(arr) if np.isfinite(arr).any() else np.nan
            if np.isfinite(mx) and mx <= 25:
                return "log1p(NT-proBNP [pg/mL])"
        return "NT-proBNP (pg/mL)"
    if var == "NYHA":
        return "NYHA class"
    if var == "LVEDD_mm":
        return "LVEDD (mm)"
    return PRETTY_LABELS.get(var, var)


def alignment_stats_text(summary_df: pd.DataFrame, null_df: pd.DataFrame, axis_df: pd.DataFrame) -> str:
    rho = p = n = lo = hi = perm_p = np.nan
    if len(summary_df):
        row = summary_df.iloc[0]
        rho = pd.to_numeric(pd.Series([row.get("spearman_acoustic_vs_clinical_axis", np.nan)]), errors="coerce").iloc[0]
        p = pd.to_numeric(pd.Series([row.get("spearman_p", np.nan)]), errors="coerce").iloc[0]
        n = pd.to_numeric(pd.Series([row.get("n_external_panel_valid", np.nan)]), errors="coerce").iloc[0]
        lo = pd.to_numeric(pd.Series([row.get("spearman_ci95_low", np.nan)]), errors="coerce").iloc[0]
        hi = pd.to_numeric(pd.Series([row.get("spearman_ci95_high", np.nan)]), errors="coerce").iloc[0]
    elif {AXIS_X, AXIS_Y}.issubset(axis_df.columns):
        rho, p, n = safe_spearman(axis_df[AXIS_X], axis_df[AXIS_Y])

    if len(null_df) and "empirical_p_abs_ge_observed" in null_df.columns:
        vals = pd.to_numeric(null_df["empirical_p_abs_ge_observed"], errors="coerce").dropna()
        if len(vals):
            perm_p = float(vals.iloc[0])

    parts: List[str] = []
    if np.isfinite(rho):
        if np.isfinite(lo) and np.isfinite(hi):
            parts.append(f"ρ={rho:.2f} [{lo:.2f}, {hi:.2f}]")
        else:
            parts.append(f"ρ={rho:.2f}")
    if np.isfinite(perm_p):
        parts.append(f"perm. {p_text(perm_p)}")
    elif np.isfinite(p):
        parts.append(p_text(p))
    if np.isfinite(n):
        parts.append(f"n={int(n)}")
    return ", ".join(parts)


def get_endpoint_label(endpoint: str) -> str:
    return PRETTY_LABELS.get(str(endpoint), str(endpoint).replace("_", " "))


def plot_axis_scatter(axis_df: pd.DataFrame, summary_df: pd.DataFrame, null_df: pd.DataFrame, out_path: Path) -> None:
    required = {AXIS_X, AXIS_Y}
    if len(axis_df) == 0 or not required.issubset(axis_df.columns):
        log(f"Skip axis scatter: missing {required - set(axis_df.columns)}")
        return
    d = axis_df.copy()
    d[AXIS_X] = pd.to_numeric(d[AXIS_X], errors="coerce")
    d[AXIS_Y] = pd.to_numeric(d[AXIS_Y], errors="coerce")
    d = d[np.isfinite(d[AXIS_X]) & np.isfinite(d[AXIS_Y])].copy()
    if len(d) < 5:
        log("Skip axis scatter: too few valid patients")
        return

    plt.figure(figsize=(8.0, 6.0))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.regplot(data=d, x=AXIS_X, y=AXIS_Y, scatter_kws={"s": 38, "alpha": 0.34, "edgecolor": "none", "color": "#0072B2"}, line_kws={"color": "#4D4D4D", "lw": 2.8}, ci=None, ax=ax)
    else:
        ax.scatter(d[AXIS_X], d[AXIS_Y], s=38, alpha=0.34, color="#0072B2", edgecolor="none")
        slope, intercept = np.polyfit(d[AXIS_X].to_numpy(float), d[AXIS_Y].to_numpy(float), 1)
        xs = np.linspace(float(d[AXIS_X].min()), float(d[AXIS_X].max()), 120)
        ax.plot(xs, slope * xs + intercept, color="#4D4D4D", lw=2.8)

    # Add decile binned means as in the main CCA scatter figure.
    try:
        q = pd.qcut(d[AXIS_X].rank(method="first"), 10, labels=False, duplicates="drop")
        b = d.assign(axis_bin=q).groupby("axis_bin", as_index=False).agg(x_mean=(AXIS_X, "mean"), y_mean=(AXIS_Y, "mean"))
        ax.plot(b["x_mean"], b["y_mean"], marker="o", markersize=7.0, lw=2.8, color="#CC79A7", label="Binned mean")
        ax.legend(frameon=False, loc="upper left")
    except Exception:
        pass

    ax.axhline(0, color="0.82", lw=1.0)
    ax.axvline(0, color="0.82", lw=1.0)
    ax.set_xlabel("External acoustic CCA axis 1 score")
    ax.set_ylabel("External clinical CCA axis 1 score")
    stats_txt = alignment_stats_text(summary_df, null_df, d)
    if stats_txt:
        ax.text(0.97, 0.04, stats_txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=18, bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="0.25", alpha=0.92))
    savefig(out_path)


def plot_quartile_gradient(axis_df: pd.DataFrame, out_path: Path) -> None:
    if len(axis_df) == 0 or AXIS_X not in axis_df.columns:
        log("Skip quartile gradient: missing axis score table")
        return
    available = [v for v in CLINICAL_PROFILE_VARS if v in axis_df.columns]
    if not available:
        log("Skip quartile gradient: no clinical profile variables available")
        return

    d = axis_df.copy()
    d[AXIS_X] = pd.to_numeric(d[AXIS_X], errors="coerce")
    d = d[np.isfinite(d[AXIS_X])].copy()
    if len(d) < 8:
        log("Skip quartile gradient: too few valid patients")
        return
    try:
        d["axis_quartile"] = pd.qcut(d[AXIS_X].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    except Exception as e:
        log(f"Skip quartile gradient: cannot form quartiles: {e}")
        return

    n = len(available)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.7 * nrows), squeeze=False)
    levels = ["Q1", "Q2", "Q3", "Q4"]

    for ax, var in zip(axes.ravel(), available):
        sub = d[[AXIS_X, "axis_quartile", var]].copy()
        sub[var] = pd.to_numeric(sub[var], errors="coerce")
        sub = sub[np.isfinite(sub[var])].copy()
        sub["axis_quartile"] = pd.Categorical(sub["axis_quartile"].astype(str), categories=levels, ordered=True)
        color = VAR_COLORS.get(var, "#0072B2")
        if len(sub) == 0:
            ax.axis("off")
            continue
        if HAS_SEABORN:
            sns.boxplot(data=sub, x="axis_quartile", y=var, color=color, showfliers=False, linewidth=1.4, boxprops={"alpha": 0.36}, medianprops={"color": "black", "linewidth": 1.55}, ax=ax)
            sns.stripplot(data=sub, x="axis_quartile", y=var, color=color, size=4.5, alpha=0.32, jitter=0.25, ax=ax)
        else:
            rng = np.random.default_rng(123)
            vals = [sub.loc[sub["axis_quartile"].astype(str).eq(q), var].dropna().to_numpy(float) for q in levels]
            ax.boxplot(vals, labels=levels, showfliers=False)
            for i, arr in enumerate(vals, start=1):
                ax.scatter(i + rng.uniform(-0.22, 0.22, len(arr)), arr, s=24, alpha=0.32, color=color)
        rho, pp, nn = safe_spearman(sub[AXIS_X], sub[var])
        if np.isfinite(rho):
            ax.text(0.03, 0.95, f"ρ={rho:.2f}, {p_text(pp)}", transform=ax.transAxes, ha="left", va="top", fontsize=16, bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.55", alpha=0.88))
        ax.set_xlabel("Acoustic CCA axis 1 quartile")
        ax.set_ylabel(unit_label(var, sub[var]))

    for ax in axes.ravel()[len(available):]:
        ax.axis("off")
    savefig(out_path)


def plot_endpoint_auroc_forest(endpoint_df: pd.DataFrame, out_path: Path) -> None:
    if len(endpoint_df) == 0:
        log("Skip endpoint AUROC forest: empty endpoint summary")
        return
    required = {"endpoint", "status", "auroc", "auroc_ci95_low", "auroc_ci95_high"}
    if not required.issubset(endpoint_df.columns):
        log(f"Skip endpoint AUROC forest: missing {required - set(endpoint_df.columns)}")
        return
    d = endpoint_df.copy()
    d = d[d["status"].astype(str).eq("ok")].copy()
    d["auroc"] = pd.to_numeric(d["auroc"], errors="coerce")
    d["auroc_ci95_low"] = pd.to_numeric(d["auroc_ci95_low"], errors="coerce")
    d["auroc_ci95_high"] = pd.to_numeric(d["auroc_ci95_high"], errors="coerce")
    d = d[np.isfinite(d["auroc"])].copy()
    if len(d) == 0:
        log("Skip endpoint AUROC forest: no endpoint with status=ok")
        return
    d["order"] = d["endpoint"].map({e: i for i, e in enumerate(ENDPOINT_ORDER)}).fillna(99)
    d = d.sort_values("order", ascending=True).reset_index(drop=True)
    d["label"] = d["endpoint"].map(get_endpoint_label)
    if "n_positive" in d.columns and "n_negative" in d.columns:
        d["label"] = d.apply(lambda r: f"{r['label']}\n+{int(r['n_positive'])}/-{int(r['n_negative'])}", axis=1)

    plt.figure(figsize=(8.4, max(4.8, 1.0 * len(d) + 1.6)))
    ax = plt.gca()
    y = np.arange(len(d))
    for yi, (_, r) in zip(y, d.iterrows()):
        color = ENDPOINT_COLORS.get(str(r["endpoint"]), "#0072B2")
        lo = r["auroc_ci95_low"]
        hi = r["auroc_ci95_high"]
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [yi, yi], color=color, lw=2.4)
            ax.plot([lo, lo], [yi - 0.08, yi + 0.08], color=color, lw=1.4)
            ax.plot([hi, hi], [yi - 0.08, yi + 0.08], color=color, lw=1.4)
        ax.scatter([r["auroc"]], [yi], s=95, color=color, edgecolor="black", linewidth=0.8, zorder=3)
        ax.text(min(0.98, float(r["auroc"]) + 0.025), yi, f"{float(r['auroc']):.2f}", va="center", ha="left", fontsize=16)
    ax.axvline(0.5, color="black", lw=1.0, ls="--")
    ax.set_xlim(0.45, 1.02)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.invert_yaxis()
    ax.set_xlabel("AUROC using external acoustic CCA axis 1")
    ax.set_ylabel("")
    savefig(out_path)


def plot_permutation_null(null_df: pd.DataFrame, out_path: Path) -> None:
    if len(null_df) == 0 or "spearman" not in null_df.columns:
        log("Skip permutation null: missing permutation table")
        return
    vals = pd.to_numeric(null_df["spearman"], errors="coerce").dropna().to_numpy(float)
    if len(vals) < 10:
        log("Skip permutation null: too few permutation values")
        return
    observed = np.nan
    perm_p = np.nan
    if "observed_spearman" in null_df.columns:
        obs_vals = pd.to_numeric(null_df["observed_spearman"], errors="coerce").dropna()
        if len(obs_vals):
            observed = float(obs_vals.iloc[0])
    if "empirical_p_abs_ge_observed" in null_df.columns:
        p_vals = pd.to_numeric(null_df["empirical_p_abs_ge_observed"], errors="coerce").dropna()
        if len(p_vals):
            perm_p = float(p_vals.iloc[0])

    plt.figure(figsize=(8.2, 5.6))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.histplot(vals, bins=35, color="#A6A6A6", edgecolor="white", ax=ax)
    else:
        ax.hist(vals, bins=35, color="#A6A6A6", edgecolor="white")
    ax.axvline(0, color="black", lw=1.0)
    if np.isfinite(observed):
        ax.axvline(observed, color="#D55E00", lw=3.0, label="Observed")
        ax.axvline(-observed, color="#D55E00", lw=1.7, ls="--")
        txt = f"Observed ρ={observed:.2f}"
        if np.isfinite(perm_p):
            txt += f"\nperm. {p_text(perm_p)}"
        ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top", fontsize=18, bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="0.25", alpha=0.92))
        ax.legend(frameon=False, loc="upper left")
    ax.set_xlabel("Permuted Spearman correlation")
    ax.set_ylabel("Count")
    savefig(out_path)



# =============================================================================
# Figure 6 helpers: locked external vs external-cohort refit
# =============================================================================

def filter_allclinical_refit_axis_scores(axis_df: pd.DataFrame) -> pd.DataFrame:
    """Return external-refit all-clinical unadjusted axis scores."""
    if len(axis_df) == 0:
        return pd.DataFrame()
    d = axis_df.copy()
    if "panel" in d.columns:
        d = d[d["panel"].astype(str).eq("all_clinical")].copy()
    if "adjustment" in d.columns:
        d = d[d["adjustment"].astype(str).eq("none")].copy()
    required = {AXIS_X, AXIS_Y}
    if not required.issubset(d.columns):
        log(f"Refit axis score table missing columns: {required - set(d.columns)}")
        return pd.DataFrame()
    d[AXIS_X] = pd.to_numeric(d[AXIS_X], errors="coerce")
    d[AXIS_Y] = pd.to_numeric(d[AXIS_Y], errors="coerce")
    return d[np.isfinite(d[AXIS_X]) & np.isfinite(d[AXIS_Y])].copy()


def filter_allclinical_refit_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Return external-refit all-clinical unadjusted component-1 summary."""
    if len(summary_df) == 0:
        return pd.DataFrame()
    d = summary_df.copy()
    if "panel" in d.columns:
        d = d[d["panel"].astype(str).eq("all_clinical")].copy()
    if "adjustment" in d.columns:
        d = d[d["adjustment"].astype(str).eq("none")].copy()
    if "component" in d.columns:
        d["component"] = pd.to_numeric(d["component"], errors="coerce")
        d = d[d["component"].eq(1)].copy()
    return d


def load_external_refit_tables(refit_tables_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load tables generated by run_cca_analysis.py on the external cohort."""
    axis_rank = read_csv_optional(refit_tables_dir / "allclinical_patient_axis_scores_ranked.csv")
    if len(axis_rank) == 0:
        axis_rank = read_csv_optional(refit_tables_dir / "oof_cca_axis_scores_by_panel.csv")
    axis_rank = filter_allclinical_refit_axis_scores(axis_rank)

    alignment = read_csv_optional(refit_tables_dir / "cca_panel_alignment_summary.csv")
    alignment = filter_allclinical_refit_summary(alignment)

    endpoint = read_csv_optional(refit_tables_dir / "endpoint_validation_summary.csv")
    return axis_rank, alignment, endpoint


def read_external_alignment_config(alignment_dir: Path) -> Dict:
    config_path = alignment_dir / "config" / "external_alignment_config.json"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"Failed to read external alignment config: {e}")
        return {}


def _numeric_from_row(row: pd.Series, candidates: Sequence[str]) -> float:
    for c in candidates:
        if c in row.index:
            val = pd.to_numeric(pd.Series([row.get(c, np.nan)]), errors="coerce").iloc[0]
            if np.isfinite(val):
                return float(val)
    return np.nan


def alignment_stats_text_general(summary_df: pd.DataFrame, null_df: pd.DataFrame, axis_df: pd.DataFrame) -> str:
    """Stats label compatible with both locked-external and refit output schemas."""
    rho = p = n = lo = hi = perm_p = np.nan
    if len(summary_df):
        row = summary_df.iloc[0]
        rho = _numeric_from_row(row, ["spearman_acoustic_vs_clinical_axis", "spearman"])
        p = _numeric_from_row(row, ["spearman_p", "p"])
        n = _numeric_from_row(row, ["n_external_panel_valid", "n", "n_test"])
        lo = _numeric_from_row(row, ["spearman_ci95_low", "ci95_low"])
        hi = _numeric_from_row(row, ["spearman_ci95_high", "ci95_high"])
    if not np.isfinite(rho) and {AXIS_X, AXIS_Y}.issubset(axis_df.columns):
        rho, p, n = safe_spearman(axis_df[AXIS_X], axis_df[AXIS_Y])
    if len(null_df) and "empirical_p_abs_ge_observed" in null_df.columns:
        vals = pd.to_numeric(null_df["empirical_p_abs_ge_observed"], errors="coerce").dropna()
        if len(vals):
            perm_p = float(vals.iloc[0])
    parts: List[str] = []
    if np.isfinite(rho):
        parts.append(f"ρ={rho:.2f} [{lo:.2f}, {hi:.2f}]" if np.isfinite(lo) and np.isfinite(hi) else f"ρ={rho:.2f}")
    if np.isfinite(perm_p):
        parts.append(f"perm. {p_text(perm_p)}")
    elif np.isfinite(p):
        parts.append(p_text(p))
    if np.isfinite(n):
        parts.append(f"n={int(n)}")
    return ", ".join(parts)


def draw_axis_scatter_panel(
    ax,
    axis_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    null_df: pd.DataFrame,
    scatter_color: str,
    panel_letter: str,
    x_label: str,
    y_label: str,
    stats_loc: str = "lower_right",
) -> None:
    """Draw one Figure 6A/B scatter panel using the Figure 3A visual grammar."""
    required = {AXIS_X, AXIS_Y}
    if len(axis_df) == 0 or not required.issubset(axis_df.columns):
        ax.axis("off")
        ax.text(0.5, 0.5, "Missing axis scores", ha="center", va="center", transform=ax.transAxes)
        return
    d = axis_df.copy()
    d[AXIS_X] = pd.to_numeric(d[AXIS_X], errors="coerce")
    d[AXIS_Y] = pd.to_numeric(d[AXIS_Y], errors="coerce")
    d = d[np.isfinite(d[AXIS_X]) & np.isfinite(d[AXIS_Y])].copy()
    if len(d) < 5:
        ax.axis("off")
        ax.text(0.5, 0.5, "Too few valid patients", ha="center", va="center", transform=ax.transAxes)
        return

    x = d[AXIS_X].to_numpy(float)
    y = d[AXIS_Y].to_numpy(float)
    ax.scatter(x, y, s=38, alpha=0.34, color=scatter_color, edgecolor="none", label="Patients")
    if np.nanstd(x) > 1e-12 and np.nanstd(y) > 1e-12:
        xmin, xmax = np.nanpercentile(x, [1, 99])
        pad = 0.08 * max(1e-9, xmax - xmin)
        xs = np.linspace(xmin - pad, xmax + pad, 160)
        slope, intercept = np.polyfit(x, y, 1)
        ax.plot(xs, slope * xs + intercept, color="#4D4D4D", lw=2.8, label="Linear trend")
        try:
            q = pd.qcut(pd.Series(x).rank(method="first"), 10, labels=False, duplicates="drop")
            b = pd.DataFrame({"x": x, "y": y, "bin": q}).groupby("bin", as_index=False).agg(x_mean=("x", "mean"), y_mean=("y", "mean"))
            ax.plot(b["x_mean"], b["y_mean"], marker="o", markersize=7.0, lw=2.8, color="#CC79A7", label="Binned mean")
        except Exception:
            pass
    ax.axhline(0, color="0.82", lw=1.0)
    ax.axvline(0, color="0.82", lw=1.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    stats_txt = alignment_stats_text_general(summary_df, null_df, d)
    if stats_txt:
        if stats_loc == "upper_left":
            loc_args = dict(x=0.04, y=0.96, ha="left", va="top")
        else:
            loc_args = dict(x=0.96, y=0.04, ha="right", va="bottom")
        ax.text(
            loc_args.pop("x"), loc_args.pop("y"), stats_txt,
            transform=ax.transAxes,
            fontsize=16,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="0.25", alpha=0.92),
            **loc_args,
        )
    ax.text(-0.14, 1.05, panel_letter, transform=ax.transAxes, ha="left", va="top", fontsize=24, fontweight="bold")


def plot_figure6A_locked_external_alignment(axis_df: pd.DataFrame, summary_df: pd.DataFrame, null_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8.0, 8.0))
    ax = plt.gca()
    draw_axis_scatter_panel(
        ax, axis_df, summary_df, null_df,
        # scatter_color="#0072B2",
        scatter_color="#D55E00",
        panel_letter="",
        x_label="Locked external acoustic CCA axis 1 score",
        y_label="Locked external clinical CCA axis 1 score",
    )
    ax.legend(frameon=False, loc="upper left", fontsize=15)
    savefig(out_path)


def plot_figure6B_external_refit_alignment(refit_axis_df: pd.DataFrame, refit_summary_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8.0, 8.0))
    ax = plt.gca()
    draw_axis_scatter_panel(
        ax, refit_axis_df, refit_summary_df, pd.DataFrame(),
        scatter_color="#D55E00",
        panel_letter="",
        x_label="External-refit acoustic CCA axis 1 score",
        y_label="External-refit clinical CCA axis 1 score",
    )
    ax.legend(frameon=False, loc="upper left", fontsize=15)
    savefig(out_path)


def prepare_endpoint_forest_table(locked_df: pd.DataFrame, refit_df: pd.DataFrame) -> pd.DataFrame:
    """Build long endpoint table for locked external vs external-refit forest plot."""
    rows = []
    for label, df in [("Locked external", locked_df), ("External refit", refit_df)]:
        if len(df) == 0:
            continue
        d = df.copy()
        if "status" in d.columns:
            d = d[d["status"].astype(str).eq("ok")].copy()
        if "n_axis_features" in d.columns:
            d["n_axis_features"] = pd.to_numeric(d["n_axis_features"], errors="coerce").fillna(1).astype(int)
            d = d[d["n_axis_features"].eq(1)].copy()
        elif "axis_feature_set" in d.columns:
            d = d[d["axis_feature_set"].astype(str).isin(["axis1", "Axis 1", "1"])].copy()
        d = d[d["endpoint"].astype(str).isin(ENDPOINT_ORDER)].copy() if "endpoint" in d.columns else pd.DataFrame()
        for _, r in d.iterrows():
            endpoint = str(r.get("endpoint", ""))
            n = _numeric_from_row(r, ["n"])
            n_pos = _numeric_from_row(r, ["n_positive"])
            n_neg = _numeric_from_row(r, ["n_negative"])
            if not np.isfinite(n_neg) and np.isfinite(n) and np.isfinite(n_pos):
                n_neg = n - n_pos
            event_text = f"+{int(n_pos)}/-{int(n_neg)}" if np.isfinite(n_pos) and np.isfinite(n) else ""
            rows.append({
                "analysis": label,
                "endpoint": endpoint,
                "endpoint_label": get_endpoint_label(endpoint),
                "auroc": _numeric_from_row(r, ["auroc", "auroc_mean", "auroc_median"]),
                "lo": _numeric_from_row(r, ["auroc_ci95_low", "auroc_p2_5"]),
                "hi": _numeric_from_row(r, ["auroc_ci95_high", "auroc_p97_5"]),
                "n": int(n) if np.isfinite(n) else np.nan,
                "n_positive": int(n_pos) if np.isfinite(n_pos) else np.nan,
                "n_negative": int(n_neg) if np.isfinite(n_neg) else np.nan,
                "event_text": event_text,
                "analysis_order": 0 if label == "Locked external" else 1,
                "endpoint_order": ENDPOINT_ORDER.index(endpoint) if endpoint in ENDPOINT_ORDER else 99,
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out[np.isfinite(out["auroc"])].sort_values(["endpoint_order", "analysis_order"]).reset_index(drop=True)
    return out


def draw_endpoint_forest_panel(ax, forest_df: pd.DataFrame, panel_letter: str = "") -> None:
    if len(forest_df) == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "Missing endpoint AUROC summaries", ha="center", va="center", transform=ax.transAxes)
        return

    d = forest_df.copy()
    d["auroc"] = pd.to_numeric(d["auroc"], errors="coerce")
    d["lo"] = pd.to_numeric(d["lo"], errors="coerce")
    d["hi"] = pd.to_numeric(d["hi"], errors="coerce")
    d = d[np.isfinite(d["auroc"])].copy()
    if len(d) == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No valid endpoint AUROC values", ha="center", va="center", transform=ax.transAxes)
        return

    endpoint_order = [e for e in ENDPOINT_ORDER if e in set(d["endpoint"].astype(str))]
    endpoint_labels = []
    for e in endpoint_order:
        sub = d[d["endpoint"].astype(str).eq(e)].copy()
        endpoint_label = get_endpoint_label(e)
        event_text = ""
        if "event_text" in sub.columns:
            vals = sub["event_text"].dropna().astype(str)
            vals = vals[vals.str.len() > 0]
            if len(vals):
                event_text = vals.iloc[0]
        if event_text:
            endpoint_labels.append(f"{endpoint_label}\n({event_text})")
        else:
            endpoint_labels.append(endpoint_label)

    y_base = {e: i for i, e in enumerate(endpoint_order)}
    offsets = {"Locked external": -0.14, "External refit": 0.14}
    colors = {"Locked external": "#0072B2", "External refit": "#D55E00"}
    markers = {"Locked external": "o", "External refit": "s"}

    used_legend_labels = set()
    for _, r in d.iterrows():
        endpoint = str(r["endpoint"])
        yi = y_base.get(endpoint, np.nan)
        if not np.isfinite(yi):
            continue

        analysis = str(r["analysis"])
        y = yi + offsets.get(analysis, 0.0)
        color = colors.get(analysis, "#0072B2")
        marker = markers.get(analysis, "o")

        lo, hi, val = float(r["lo"]), float(r["hi"]), float(r["auroc"])
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [y, y], color=color, lw=2.5, alpha=0.95)
            ax.plot([lo, lo], [y - 0.06, y + 0.06], color=color, lw=1.3)
            ax.plot([hi, hi], [y - 0.06, y + 0.06], color=color, lw=1.3)

        legend_label = analysis if analysis not in used_legend_labels else "_nolegend_"
        used_legend_labels.add(analysis)
        ax.scatter([val], [y], s=95, color=color, marker=marker, edgecolor="black", linewidth=0.8, zorder=3, label=legend_label)

        if np.isfinite(lo) and np.isfinite(hi):
            value_label = f"{val:.2f} ({lo:.2f}, {hi:.2f})"
        else:
            value_label = f"{val:.2f}"
        text_x = max(val, hi if np.isfinite(hi) else val) + 0.018
        ax.text(text_x, y, value_label, va="center", ha="left", fontsize=13)

    ax.axvline(0.5, color="black", lw=1.0, ls="--", alpha=0.8)
    ax.set_xlim(0.45, 1.05)
    ax.set_yticks(np.arange(len(endpoint_order)))
    ax.set_yticklabels(endpoint_labels)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC")
    ax.set_ylabel("")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=2, fontsize=13, handletextpad=0.5, columnspacing=1.4)

    if panel_letter:
        ax.text(-0.17, 1.10, panel_letter, transform=ax.transAxes, ha="left", va="top", fontsize=24, fontweight="bold")


def plot_figure6C_endpoint_auroc_locked_vs_refit(
    locked_endpoint_df: pd.DataFrame,
    refit_endpoint_df: pd.DataFrame,
    out_path: Path,
    ) -> pd.DataFrame:
    forest_df = prepare_endpoint_forest_table(locked_endpoint_df, refit_endpoint_df)
    plt.figure(figsize=(9, max(6, 0.88 * len(ENDPOINT_ORDER) + 2.6)))
    ax = plt.gca()
    draw_endpoint_forest_panel(ax, forest_df, panel_letter="")
    savefig(out_path)
    return forest_df


def plot_figure6_combined(
    locked_axis_df: pd.DataFrame,
    locked_summary_df: pd.DataFrame,
    locked_null_df: pd.DataFrame,
    refit_axis_df: pd.DataFrame,
    refit_summary_df: pd.DataFrame,
    endpoint_forest_df: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(22.5, 6.2), gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]})
    draw_axis_scatter_panel(
        axes[0], locked_axis_df, locked_summary_df, locked_null_df,
        scatter_color="#0072B2",
        panel_letter="A",
        x_label="Locked external acoustic CCA axis 1 score",
        y_label="Locked external clinical CCA axis 1 score",
    )
    axes[0].legend(frameon=False, loc="upper left", fontsize=13)
    draw_axis_scatter_panel(
        axes[1], refit_axis_df, refit_summary_df, pd.DataFrame(),
        scatter_color="#D55E00",
        panel_letter="B",
        x_label="External-refit acoustic CCA axis 1 score",
        y_label="External-refit clinical CCA axis 1 score",
    )
    axes[1].legend(frameon=False, loc="upper left", fontsize=13)
    draw_endpoint_forest_panel(axes[2], endpoint_forest_df, panel_letter="C")
    savefig(out_path)


# =============================================================================
# Additional external-validation diagnostic plots
# =============================================================================

def load_internal_reference_axis_scores(internal_tables_dir: Path) -> pd.DataFrame:
    """Load internal all-clinical axis scores for internal-vs-external distribution diagnostics."""
    candidates = [
        "allclinical_patient_axis_scores_ranked.csv",
        "oof_cca_axis_scores_by_panel.csv",
        "cca_axis_scores_by_panel.csv",
        "patient_axis_scores.csv",
        "axis_scores.csv",
    ]
    for name in candidates:
        p = Path(internal_tables_dir) / name
        d = read_csv_optional(p)
        if len(d) == 0:
            continue
        d = filter_allclinical_refit_axis_scores(d)
        if len(d):
            return d
    log(f"No usable internal axis-score table found under {internal_tables_dir}; distribution-shift plot will be skipped.")
    return pd.DataFrame()


def _clean_axis_values(df: pd.DataFrame, col: str = AXIS_X) -> np.ndarray:
    if len(df) == 0 or col not in df.columns:
        return np.asarray([], dtype=float)
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    return vals[np.isfinite(vals)]


def _normal_approx_binomial_ci(k: int, n: int) -> Tuple[float, float, float]:
    if n <= 0:
        return np.nan, np.nan, np.nan
    p = float(k / n)
    se = np.sqrt(max(p * (1.0 - p), 0.0) / n)
    return p, max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def plot_external_axis_distribution_shift(
    internal_axis_df: pd.DataFrame,
    locked_axis_df: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    """Density overlap of internal OOF acoustic axis and locked external acoustic axis.

    This panel is intended to show whether the externally projected patients lie
    within the internal acoustic-axis reference range, which is difficult to
    summarize clearly in a table alone.
    """
    x_int = _clean_axis_values(internal_axis_df, AXIS_X)
    x_ext = _clean_axis_values(locked_axis_df, AXIS_X)
    if len(x_int) < 20 or len(x_ext) < 20:
        log("Skip external axis distribution shift: missing internal or external axis values.")
        return pd.DataFrame()

    mu = float(np.mean(x_int))
    sd = float(np.std(x_int, ddof=1))
    if not np.isfinite(sd) or sd < 1e-12:
        log("Skip external axis distribution shift: internal axis SD is near zero.")
        return pd.DataFrame()
    zi = (x_int - mu) / sd
    ze = (x_ext - mu) / sd
    ks_stat, ks_p = stats.ks_2samp(zi, ze)

    source_df = pd.DataFrame({
        "cohort": ["Internal OOF"] * len(zi) + ["Locked external"] * len(ze),
        "acoustic_axis1_internal_z": np.concatenate([zi, ze]),
    })

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    if HAS_SEABORN:
        palette = {
            # "Internal OOF": "#DD8452",
            "Internal OOF": "grey",
            "Locked external": "#4C72B0",
        }

        sns.kdeplot(
            data=source_df,
            x="acoustic_axis1_internal_z",
            hue="cohort",
            hue_order=["Internal OOF", "Locked external"],
            palette=palette,
            fill=True,
            common_norm=False,
            alpha=0.24,
            linewidth=2.4,
            ax=ax,
        )
        sns.rugplot(data=source_df[source_df["cohort"].eq("Locked external")], x="acoustic_axis1_internal_z", height=0.05, color="#0072B2", alpha=0.45, ax=ax)
    else:
        bins = np.linspace(np.nanpercentile(np.concatenate([zi, ze]), 1), np.nanpercentile(np.concatenate([zi, ze]), 99), 32)
        ax.hist(zi, bins=bins, density=True, alpha=0.28, label="Internal OOF")
        ax.hist(ze, bins=bins, density=True, alpha=0.28, label="Locked external")

    med_i = float(np.median(zi))
    med_e = float(np.median(ze))
    ax.axvline(med_i, color="#0072B2", linestyle="--", lw=2.0, alpha=0.9)
    ax.axvline(med_e, color="#D55E00", linestyle="--", lw=2.0, alpha=0.9)
    ax.axvspan(-1.96, 1.96, color="0.88", alpha=0.22, lw=0)
    ax.set_xlabel("Acoustic CCA axis 1 score\n(z-scored to internal OOF distribution)")
    ax.set_ylabel("Density")
    ax.text(
        0.98, 0.96,
        # f"Internal n={len(zi)}; external n={len(ze)}\n"
        f"Median shift={med_e - med_i:.2f} SD\n"
        f"KS D={ks_stat:.2f}, {p_text(ks_p)}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=15,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="0.25", alpha=0.92),
    )
    if not HAS_SEABORN:
        ax.legend(frameon=False, loc="upper left", fontsize=15)
    savefig(out_path)
    return source_df


def build_moving_window_endpoint_enrichment(
    pred_df: pd.DataFrame,
    window_size: int = 30,
    step_size: int = 5,
) -> pd.DataFrame:
    """Build moving-window endpoint prevalence along the locked acoustic axis."""
    if len(pred_df) == 0 or not {"endpoint", "y_true", AXIS_X}.issubset(pred_df.columns):
        return pd.DataFrame()
    rows = []
    for endpoint in ENDPOINT_ORDER:
        d = pred_df[pred_df["endpoint"].astype(str).eq(endpoint)].copy()
        if len(d) == 0:
            continue
        d["y_true"] = pd.to_numeric(d["y_true"], errors="coerce")
        d[AXIS_X] = pd.to_numeric(d[AXIS_X], errors="coerce")
        d = d[np.isfinite(d["y_true"]) & np.isfinite(d[AXIS_X])].copy()
        d = d.sort_values(AXIS_X).reset_index(drop=True)
        n = len(d)
        if n < 12 or d["y_true"].nunique() < 2:
            continue
        local_window = int(min(max(10, window_size), n))
        local_step = int(max(1, min(step_size, local_window)))
        for start in range(0, max(1, n - local_window + 1), local_step):
            end = min(n, start + local_window)
            if end - start < max(8, local_window // 2):
                continue
            sub = d.iloc[start:end]
            k = int(sub["y_true"].sum())
            p, lo, hi = _normal_approx_binomial_ci(k, len(sub))
            center_idx = (start + end - 1) / 2.0
            rows.append({
                "endpoint": endpoint,
                "endpoint_label": get_endpoint_label(endpoint),
                "center_rank": center_idx + 1,
                "center_percentile": 100.0 * (center_idx + 0.5) / n,
                "center_axis_score": float(sub[AXIS_X].median()),
                "window_start_rank": start + 1,
                "window_end_rank": end,
                "window_n": int(len(sub)),
                "n_positive": k,
                "prevalence": p,
                "prevalence_ci95_low": lo,
                "prevalence_ci95_high": hi,
                "overall_prevalence": float(d["y_true"].mean()),
            })
    return pd.DataFrame(rows)


def plot_external_moving_window_endpoint_enrichment(
    pred_df: pd.DataFrame,
    out_path: Path,
    window_size: int = 30,
    step_size: int = 5,
) -> pd.DataFrame:
    """Moving-window endpoint enrichment along the locked external acoustic axis."""
    curve_df = build_moving_window_endpoint_enrichment(pred_df, window_size=window_size, step_size=step_size)
    if len(curve_df) == 0:
        log("Skip moving-window endpoint enrichment: no valid endpoint predictions.")
        return pd.DataFrame()

    plt.figure(figsize=(9.6, 6.2))
    ax = plt.gca()
    for endpoint in [e for e in ENDPOINT_ORDER if e in set(curve_df["endpoint"])]:
        sub = curve_df[curve_df["endpoint"].astype(str).eq(endpoint)].copy()
        color = ENDPOINT_COLORS.get(endpoint, "#0072B2")
        x = sub["center_percentile"].to_numpy(float)
        y = 100.0 * sub["prevalence"].to_numpy(float)
        lo = 100.0 * sub["prevalence_ci95_low"].to_numpy(float)
        hi = 100.0 * sub["prevalence_ci95_high"].to_numpy(float)
        ax.plot(x, y, color=color, lw=2.8, marker="o", markersize=4.8, label=get_endpoint_label(endpoint))
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
        base = 100.0 * float(sub["overall_prevalence"].iloc[0])
        ax.axhline(base, color=color, lw=1.1, linestyle="--", alpha=0.45)

    ax.set_xlim(0, 100)
    ymax = max(10.0, min(100.0, 1.18 * np.nanmax(100.0 * curve_df["prevalence_ci95_high"].to_numpy(float))))
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Percentile along locked external acoustic CCA axis 1")
    ax.set_ylabel("Endpoint prevalence within moving window (%)")
    ax.legend(frameon=False, loc="upper left", fontsize=13, ncol=1)
    savefig(out_path)
    return curve_df


def plot_external_endpoint_axis_distribution_raincloud(
    pred_df: pd.DataFrame,
    endpoint_summary_df: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    """Endpoint-positive vs endpoint-negative acoustic-axis score distributions."""
    if len(pred_df) == 0 or not {"endpoint", "y_true", AXIS_X}.issubset(pred_df.columns):
        log("Skip endpoint axis distribution: missing prediction table.")
        return pd.DataFrame()

    d = pred_df[pred_df["endpoint"].astype(str).isin(ENDPOINT_ORDER)].copy()
    d["y_true"] = pd.to_numeric(d["y_true"], errors="coerce")
    d[AXIS_X] = pd.to_numeric(d[AXIS_X], errors="coerce")
    d = d[np.isfinite(d["y_true"]) & np.isfinite(d[AXIS_X])].copy()
    if len(d) == 0:
        log("Skip endpoint axis distribution: no valid rows.")
        return pd.DataFrame()
    d["endpoint_label"] = d["endpoint"].map(get_endpoint_label)
    d["status_label"] = d["y_true"].astype(int).map({0: "Negative", 1: "Positive"})

    auc_map: Dict[str, Tuple[float, float, float]] = {}
    if len(endpoint_summary_df) and {"endpoint", "auroc"}.issubset(endpoint_summary_df.columns):
        tmp = endpoint_summary_df.copy()
        if "status" in tmp.columns:
            tmp = tmp[tmp["status"].astype(str).eq("ok")].copy()
        for _, r in tmp.iterrows():
            ep = str(r.get("endpoint", ""))
            auc = _numeric_from_row(r, ["auroc"])
            lo = _numeric_from_row(r, ["auroc_ci95_low", "auroc_p2_5"])
            hi = _numeric_from_row(r, ["auroc_ci95_high", "auroc_p97_5"])
            auc_map[ep] = (auc, lo, hi)

    endpoints = [e for e in ENDPOINT_ORDER if e in set(d["endpoint"].astype(str))]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8), squeeze=False, sharey=True)

    # Muted, journal-style endpoint-status colors: cool slate for endpoint-negative
    # patients and warm terracotta for endpoint-positive patients.
    palette = {"Negative": "#8FA3AD", "Positive": "#C65D33"}
    source_rows = []
    for idx, (ax, endpoint) in enumerate(zip(axes.ravel(), endpoints)):
        sub = d[d["endpoint"].astype(str).eq(endpoint)].copy()
        if len(sub) == 0:
            ax.axis("off")
            continue
        if HAS_SEABORN:
            sns.violinplot(
                data=sub, x="status_label", y=AXIS_X,
                order=["Negative", "Positive"], palette=palette, cut=0,
                inner=None, linewidth=1.15, saturation=0.95, ax=ax,
            )
            sns.boxplot(
                data=sub, x="status_label", y=AXIS_X,
                order=["Negative", "Positive"], width=0.26, showcaps=True,
                showfliers=False,
                boxprops={"facecolor": "white", "edgecolor": "black", "alpha": 0.76},
                medianprops={"color": "black", "linewidth": 1.45},
                whiskerprops={"color": "black", "linewidth": 1.05},
                capprops={"color": "black", "linewidth": 1.05},
                ax=ax,
            )
            sns.stripplot(
                data=sub, x="status_label", y=AXIS_X,
                order=["Negative", "Positive"], color="black",
                alpha=0.28, size=3.6, jitter=0.20, ax=ax,
            )
        else:
            vals = [sub.loc[sub["status_label"].eq(k), AXIS_X].to_numpy(float) for k in ["Negative", "Positive"]]
            ax.boxplot(vals, labels=["Negative", "Positive"], showfliers=False)

        neg = sub.loc[sub["status_label"].eq("Negative"), AXIS_X].to_numpy(float)
        pos = sub.loc[sub["status_label"].eq("Positive"), AXIS_X].to_numpy(float)
        pval = np.nan
        if len(neg) >= 3 and len(pos) >= 3:
            try:
                pval = stats.mannwhitneyu(pos, neg, alternative="two-sided").pvalue
            except Exception:
                pval = np.nan

        auc, auc_lo, auc_hi = auc_map.get(endpoint, (np.nan, np.nan, np.nan))
        txt = f"{get_endpoint_label(endpoint)}\n+{len(pos)}/-{len(neg)}"
        if np.isfinite(auc):
            if np.isfinite(auc_lo) and np.isfinite(auc_hi):
                txt += f"\nAUROC={auc:.2f} ({auc_lo:.2f}, {auc_hi:.2f})"
            else:
                txt += f"\nAUROC={auc:.2f}"
        if np.isfinite(pval):
            txt += f"\n{p_text(pval)}"

        ax.text(
            0.04, 0.96, txt,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=12.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.35", alpha=0.92),
        )
        ax.set_xlabel("")
        ax.set_ylabel("Locked external acoustic axis 1" if idx % 2 == 0 else "")
        if idx % 2 == 1:
            ax.tick_params(axis="y", labelleft=False)

        sub = sub.copy()
        sub["endpoint_rank_sum_p"] = pval
        sub["endpoint_rank_sum_test"] = "Mann-Whitney U test, two-sided"
        sub["endpoint_auroc"] = auc
        sub["endpoint_auroc_ci95_low"] = auc_lo
        sub["endpoint_auroc_ci95_high"] = auc_hi
        source_rows.append(sub)

    for ax in axes.ravel()[len(endpoints):]:
        ax.axis("off")
    savefig(out_path)
    return pd.concat(source_rows, ignore_index=True) if source_rows else pd.DataFrame()
def plot_external_rank_percentile_agreement(
    locked_axis_df: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    """Rank-percentile agreement between locked acoustic and clinical axes."""
    if len(locked_axis_df) == 0 or not {AXIS_X, AXIS_Y}.issubset(locked_axis_df.columns):
        log("Skip rank-percentile agreement: missing locked axis scores.")
        return pd.DataFrame()
    d = locked_axis_df.copy()
    d[AXIS_X] = pd.to_numeric(d[AXIS_X], errors="coerce")
    d[AXIS_Y] = pd.to_numeric(d[AXIS_Y], errors="coerce")
    d = d[np.isfinite(d[AXIS_X]) & np.isfinite(d[AXIS_Y])].copy()
    if len(d) < 20:
        log("Skip rank-percentile agreement: too few valid patients.")
        return pd.DataFrame()

    d["acoustic_percentile"] = 100.0 * (d[AXIS_X].rank(method="average") - 0.5) / len(d)
    d["clinical_percentile"] = 100.0 * (d[AXIS_Y].rank(method="average") - 0.5) / len(d)
    d["absolute_percentile_gap"] = (d["acoustic_percentile"] - d["clinical_percentile"]).abs()
    rho, pval, n = safe_spearman(d["acoustic_percentile"], d["clinical_percentile"])
    med_gap = float(d["absolute_percentile_gap"].median())

    plt.figure(figsize=(7.4, 6.8))
    ax = plt.gca()
    x = d["acoustic_percentile"].to_numpy(float)
    y = d["clinical_percentile"].to_numpy(float)
    if len(d) >= 80:
        hb = ax.hexbin(x, y, gridsize=22, mincnt=1, cmap="viridis", linewidths=0, alpha=0.92)
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label("Patient count")
    else:
        ax.scatter(x, y, s=38, alpha=0.45, color="#0072B2", edgecolor="none")
    ax.plot([0, 100], [0, 100], color="black", linestyle="--", lw=1.4, alpha=0.75)

    try:
        bins = pd.qcut(d["acoustic_percentile"].rank(method="first"), 10, labels=False, duplicates="drop")
        b = d.assign(bin=bins).groupby("bin", as_index=False).agg(
            acoustic_percentile=("acoustic_percentile", "mean"),
            clinical_percentile=("clinical_percentile", "mean"),
        )
        ax.plot(b["acoustic_percentile"], b["clinical_percentile"], marker="o", markersize=6.5, lw=2.6, color="#D55E00", label="Binned mean")
        ax.legend(frameon=False, loc="upper left", fontsize=13)
    except Exception:
        pass

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Locked acoustic axis percentile")
    ax.set_ylabel("Locked clinical axis percentile")
    ax.text(
        0.96, 0.04,
        f"Rank ρ={rho:.2f}, {p_text(pval)}\n"
        f"Median |percentile gap|={med_gap:.1f}\n"
        f"n={n}",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="0.25", alpha=0.92),
    )
    savefig(out_path)
    return d[[c for c in ["patient_id", AXIS_X, AXIS_Y, "acoustic_percentile", "clinical_percentile", "absolute_percentile_gap"] if c in d.columns]]



# =============================================================================
# Supplementary tables
# =============================================================================

def normalize_patient_id(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return "".join(s.split())


def read_table_optional(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    path = Path(path)
    try:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
        log(f"Loaded table: {path}, shape={df.shape}")
        return df
    except Exception as e:
        log(f"Failed to read table {path}: {e}")
        return pd.DataFrame()


def infer_patient_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["patient_id", "patient", "pid", "ID", "id", "subject_id", "序号", "病人编码"]:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = str(c).lower()
        if "patient" in lc or lc in {"pid", "id"}:
            return c
    return None


def load_external_clinical_for_tables(args, locked_axis_df: pd.DataFrame, locked_config: Dict) -> pd.DataFrame:
    candidates: List[Path] = []
    if args.external_clinical_table is not None:
        candidates.append(Path(args.external_clinical_table))
    cfg_path = locked_config.get("external_clinical_table")
    if cfg_path:
        candidates.append(Path(cfg_path))
    # Final fallback: the axis table contains a small subset of clinical columns.
    clinical = pd.DataFrame()
    for p in candidates:
        clinical = read_table_optional(p)
        if len(clinical):
            break
    if len(clinical) == 0:
        clinical = locked_axis_df.copy()
        log("Using external_axis_scores.csv as fallback for SI cohort table; clinical-variable coverage may be incomplete.")
    pid_col = infer_patient_col(clinical)
    if pid_col is not None:
        clinical = clinical.copy()
        clinical["patient_id"] = clinical[pid_col].map(normalize_patient_id)
        clinical = clinical.dropna(subset=["patient_id"]).drop_duplicates("patient_id", keep="first")
    if len(locked_axis_df) and "patient_id" in locked_axis_df.columns and "patient_id" in clinical.columns:
        keep_ids = set(locked_axis_df["patient_id"].map(normalize_patient_id).dropna())
        if keep_ids:
            clinical = clinical[clinical["patient_id"].isin(keep_ids)].copy()
    return clinical.reset_index(drop=True)


def _to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _coverage(vals: pd.Series, n_total: int) -> str:
    n = int(vals.notna().sum())
    pct = 100.0 * n / n_total if n_total else np.nan
    return f"{pct:.1f}%" if np.isfinite(pct) else "NA"


def _continuous_summary(vals: pd.Series) -> str:
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    if len(vals) == 0:
        return "NA"
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    return f"{med:.1f} ({q1:.1f}, {q3:.1f})"


def _binary_summary(vals: pd.Series) -> str:
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    if len(vals) == 0:
        return "NA"
    n_pos = int((vals > 0.5).sum())
    pct = 100.0 * n_pos / len(vals)
    return f"{n_pos} ({pct:.1f}%)"


def _add_continuous_row(rows: List[Dict], clinical: pd.DataFrame, domain: str, col: str, label: str, notes: str = "") -> None:
    if col not in clinical.columns:
        return
    n_total = len(clinical)
    vals = _to_numeric_series(clinical, col)
    rows.append({"Domain": domain, "Variable": label, "Summary": _continuous_summary(vals), "Coverage": _coverage(vals, n_total), "Notes": notes})


def _add_binary_row(rows: List[Dict], clinical: pd.DataFrame, domain: str, vals: pd.Series, label: str, notes: str = "") -> None:
    n_total = len(clinical)
    rows.append({"Domain": domain, "Variable": label, "Summary": _binary_summary(vals), "Coverage": _coverage(vals, n_total), "Notes": notes})


def make_ntprobnp_endpoint(clinical: pd.DataFrame) -> pd.Series:
    if "NTproBNP_raw" in clinical.columns:
        vals = _to_numeric_series(clinical, "NTproBNP_raw")
        return (vals >= 900).astype(float).where(vals.notna())
    if "NTproBNP" in clinical.columns:
        vals = _to_numeric_series(clinical, "NTproBNP")
        finite = vals.dropna()
        thr = np.log1p(900.0) if len(finite) and finite.max() <= 25 else 900.0
        return (vals >= thr).astype(float).where(vals.notna())
    return pd.Series(np.nan, index=clinical.index, dtype=float)


def make_lvedd_dilated_endpoint(clinical: pd.DataFrame) -> pd.Series:
    vals = _to_numeric_series(clinical, "LVEDD_mm")
    if "sex_male" in clinical.columns:
        sex = _to_numeric_series(clinical, "sex_male")
    elif "Male sex" in clinical.columns:
        sex = _to_numeric_series(clinical, "Male sex")
    else:
        sex = pd.Series(np.nan, index=clinical.index, dtype=float)
    out = pd.Series(np.nan, index=clinical.index, dtype=float)
    valid = vals.notna() & sex.notna()
    thr = pd.Series(np.where(sex >= 0.5, 58.0, 52.0), index=clinical.index)
    out.loc[valid] = (vals.loc[valid] > thr.loc[valid]).astype(float)
    if valid.sum() == 0:
        out.loc[vals.notna()] = (vals.loc[vals.notna()] > 55.0).astype(float)
    return out


def build_external_cohort_table1_style(clinical: pd.DataFrame, locked_axis_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    n_total = len(clinical) if len(clinical) else len(locked_axis_df)
    rows.append({"Domain": "Cohort", "Variable": "Patients included", "Summary": str(n_total), "Coverage": "100.0%" if n_total else "NA", "Notes": "Patients included in the external acoustic-clinical validation cohort."})
    if len(locked_axis_df):
        rows.append({"Domain": "Recording coverage", "Variable": "Patients with external axis scores", "Summary": str(len(locked_axis_df)), "Coverage": f"{100.0 * len(locked_axis_df) / max(n_total, 1):.1f}%", "Notes": "Patients transformed by the locked internal CCA pipeline."})

    for col, label in [("age_years", "Age, years"), ("Age", "Age, years")]:
        if col in clinical.columns:
            _add_continuous_row(rows, clinical, "Demographics", col, label)
            break
    for col, label in [("sex_male", "Male sex"), ("male", "Male sex"), ("Male sex", "Male sex")]:
        if col in clinical.columns:
            _add_binary_row(rows, clinical, "Demographics", _to_numeric_series(clinical, col), label, "Number and percentage of male patients among non-missing records.")
            break
    for col in ["heart_rate", "Heart rate", "HR"]:
        if col in clinical.columns:
            _add_continuous_row(rows, clinical, "Physiologic covariate", col, "Heart rate, bpm")
            break

    if "NYHA" in clinical.columns:
        _add_binary_row(rows, clinical, "Function", (_to_numeric_series(clinical, "NYHA") >= 3).astype(float).where(_to_numeric_series(clinical, "NYHA").notna()), "NYHA class ≥3", "Ordinal class; NYHA ≥3.")
    _add_continuous_row(rows, clinical, "Function", "EF_Teich", "EF, %")
    if "NTproBNP" in clinical.columns:
        _add_continuous_row(rows, clinical, "Function", "NTproBNP", "NT-proBNP", "Summarized on the scale available in the cleaned table; log-transformed for modeling when applicable.")
    elif "NTproBNP_raw" in clinical.columns:
        _add_continuous_row(rows, clinical, "Function", "NTproBNP_raw", "NT-proBNP, pg/mL")

    for col, label in [("LA_mm", "LA, mm"), ("LVEDD_mm", "LVEDD, mm"), ("IVS_mm", "IVS, mm"), ("LVPW_mm", "LVPW, mm")]:
        _add_continuous_row(rows, clinical, "Structure", col, label)

    for col, label in [("MR_grade", "MR grade ≥2"), ("TR_grade", "TR grade ≥2"), ("AR_grade", "AR grade ≥2"), ("AS_grade", "AS grade ≥2"), ("MS_grade", "MS grade ≥2"), ("PR_grade", "PR grade ≥2")]:
        if col in clinical.columns:
            vals = _to_numeric_series(clinical, col)
            _add_binary_row(rows, clinical, "Valve status", (vals >= 2).astype(float).where(vals.notna()), label, "Ordinal valve grade; moderate-or-greater (grade ≥2).")

    if "EF_Teich" in clinical.columns:
        ef = _to_numeric_series(clinical, "EF_Teich")
        _add_binary_row(rows, clinical, "Endpoint prevalence", (ef < 40).astype(float).where(ef.notna()), "EF <40%", "Prespecified endpoint label derived from EF.")
    if "NTproBNP" in clinical.columns or "NTproBNP_raw" in clinical.columns:
        _add_binary_row(rows, clinical, "Endpoint prevalence", make_ntprobnp_endpoint(clinical), "NT-proBNP ≥900 pg/mL", "Prespecified endpoint label; log1p threshold used when the cleaned column is log-transformed.")
    if "LVEDD_mm" in clinical.columns:
        _add_binary_row(rows, clinical, "Endpoint prevalence", make_lvedd_dilated_endpoint(clinical), "LVEDD dilation", "Male >58 mm or female >52 mm when sex is available.")

    duration_candidates = [
        ("duration_A", "Recording duration at the aortic auscultation position."),
        ("duration_E", "Recording duration at Erb’s point."),
        ("duration_M", "Recording duration at the mitral auscultation position."),
        ("duration_P", "Recording duration at the pulmonic auscultation position."),
        ("duration_T", "Recording duration at the tricuspid auscultation position."),
    ]
    for col, note in duration_candidates:
        if col in clinical.columns:
            _add_continuous_row(rows, clinical, "Recording", col, col, note)
    if all(c in clinical.columns for c, _ in duration_candidates):
        total = sum(_to_numeric_series(clinical, c) for c, _ in duration_candidates)
        rows.append({"Domain": "Recording", "Variable": "Total recording duration, s", "Summary": _continuous_summary(total), "Coverage": _coverage(total, n_total), "Notes": "Sum of recording durations across the five auscultation positions."})
    for col in ["total_windows", "Total windows per patient", "n_windows_total", "total_windows_per_patient"]:
        if col in clinical.columns:
            _add_continuous_row(rows, clinical, "Recording", col, "Total windows per patient")
            break

    return pd.DataFrame(rows)


def _metric_at_youden(y: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    y = y[mask]; score = score[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {}
    thresholds = np.unique(score)
    best = None
    for thr in thresholds:
        pred = (score >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
        youden = sens + spec - 1 if np.isfinite(sens) and np.isfinite(spec) else -np.inf
        if best is None or youden > best[0]:
            ppv = tp / (tp + fp) if tp + fp else np.nan
            npv = tn / (tn + fn) if tn + fn else np.nan
            ba = 0.5 * (sens + spec) if np.isfinite(sens) and np.isfinite(spec) else np.nan
            best = (youden, thr, sens, spec, ppv, npv, ba)
    if best is None:
        return {}
    return {
        "youden_threshold_descriptive": float(best[1]),
        "sensitivity_at_youden_descriptive": float(best[2]),
        "specificity_at_youden_descriptive": float(best[3]),
        "ppv_at_youden_descriptive": float(best[4]),
        "npv_at_youden_descriptive": float(best[5]),
        "balanced_accuracy_at_youden_descriptive": float(best[6]),
    }


def build_locked_external_detailed_metrics(endpoint_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    if len(endpoint_df) == 0:
        return pd.DataFrame()
    rows = []
    for _, r in endpoint_df.iterrows():
        endpoint = str(r.get("endpoint", ""))
        rec = {
            "analysis": "locked_external_transform_only",
            "endpoint": endpoint,
            "endpoint_label": get_endpoint_label(endpoint),
            "source_column": r.get("source_column", ""),
            "rule": r.get("rule", ""),
            "score": r.get("score", AXIS_X),
            "status": r.get("status", ""),
            "n": _numeric_from_row(r, ["n"]),
            "n_positive": _numeric_from_row(r, ["n_positive"]),
            "n_negative": _numeric_from_row(r, ["n_negative"]),
            "positive_rate": _numeric_from_row(r, ["positive_rate"]),
            "auroc": _numeric_from_row(r, ["auroc"]),
            "auroc_ci95_low": _numeric_from_row(r, ["auroc_ci95_low"]),
            "auroc_ci95_high": _numeric_from_row(r, ["auroc_ci95_high"]),
        }
        if len(pred_df) and {"endpoint", "y_true"}.issubset(pred_df.columns):
            sub = pred_df[pred_df["endpoint"].astype(str).eq(endpoint)].copy()
            score_col = "cca_acoustic_axis1" if "cca_acoustic_axis1" in sub.columns else ("y_prob" if "y_prob" in sub.columns else None)
            if score_col is not None and len(sub):
                y = pd.to_numeric(sub["y_true"], errors="coerce").to_numpy(float)
                s = pd.to_numeric(sub[score_col], errors="coerce").to_numpy(float)
                mask = np.isfinite(y) & np.isfinite(s)
                y = y[mask].astype(int); s = s[mask]
                if HAS_SKLEARN_METRICS and len(y) and len(np.unique(y)) == 2:
                    try:
                        rec["average_precision"] = float(average_precision_score(y, s))
                    except Exception:
                        rec["average_precision"] = np.nan
                    rec.update(_metric_at_youden(y, s))
        rec["notes"] = "AUROC uses the locked external acoustic CCA axis. Youden-threshold metrics, if present, are descriptive and thresholded within the external cohort."
        rows.append(rec)
    out = pd.DataFrame(rows)
    order = {e: i for i, e in enumerate(ENDPOINT_ORDER)}
    if len(out):
        out["endpoint_order"] = out["endpoint"].map(order).fillna(99)
        out = out.sort_values("endpoint_order").drop(columns=["endpoint_order"]).reset_index(drop=True)
    return out

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot external validation and external-refit CCA results.")
    parser.add_argument("--alignment-dir", type=Path, default=Path(r"D:\PycharmProjects\HSCA\Outputs\validation\alignment"), help="Locked external validation output directory from run_apply_internal_cca_to_external.py.")
    parser.add_argument("--tables-dir", type=Path, default=None, help="Optional locked external tables directory. Defaults to <alignment-dir>/tables.")
    parser.add_argument("--refit-alignment-dir", type=Path, default=Path(r"D:\PycharmProjects\HSCA\Outputs\validation\alignment_external"), help="External-cohort refit output directory from run_cca_analysis.py.")
    parser.add_argument("--refit-tables-dir", type=Path, default=None, help="Optional external-refit tables directory. Defaults to <refit-alignment-dir>/tables.")
    parser.add_argument("--external-clinical-table", type=Path, default=Path(r"D:\PycharmProjects\HSCA\Outputs\validation\preprocessing\Data_clinic\clinical_clean.csv"), help="Optional cleaned external clinical table for the SI cohort-characteristics table.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Figure output directory. Defaults to <alignment-dir>/figures.")
    parser.add_argument("--table-out-dir", type=Path, default=None, help="Output directory for generated SI tables. Defaults to the locked external tables directory.")
    parser.add_argument("--internal-alignment-dir", type=Path, default=Path(r"D:\PycharmProjects\HSCA\Outputs\alignment\CCA\beats\4_5_4_1"), help="Optional internal CCA output directory for internal-vs-external axis distribution diagnostics.")
    parser.add_argument("--internal-tables-dir", type=Path, default=None, help="Optional internal CCA tables directory. Defaults to <internal-alignment-dir>/tables.")
    parser.add_argument("--moving-window-size", type=int, default=30, help="Window size for moving-window endpoint enrichment along the locked external acoustic axis.")
    parser.add_argument("--moving-window-step", type=int, default=5, help="Step size for moving-window endpoint enrichment along the locked external acoustic axis.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup_plotting()

    locked_tables_dir = args.tables_dir if args.tables_dir is not None else args.alignment_dir / "tables"
    refit_tables_dir = args.refit_tables_dir if args.refit_tables_dir is not None else args.refit_alignment_dir / "tables"
    internal_tables_dir = args.internal_tables_dir if args.internal_tables_dir is not None else args.internal_alignment_dir / "tables"
    out_dir = args.out_dir if args.out_dir is not None else args.alignment_dir / "figures"
    table_out_dir = args.table_out_dir if args.table_out_dir is not None else locked_tables_dir
    ensure_dir(out_dir)
    ensure_dir(table_out_dir)

    locked_axis_df = read_csv_optional(locked_tables_dir / "external_axis_scores.csv")
    locked_summary_df = read_csv_optional(locked_tables_dir / "external_main_alignment_summary.csv")
    locked_endpoint_df = read_csv_optional(locked_tables_dir / "external_endpoint_auroc_summary.csv")
    locked_pred_df = read_csv_optional(locked_tables_dir / "external_endpoint_predictions.csv")
    locked_null_df = read_csv_optional(locked_tables_dir / "external_permutation_null.csv")
    locked_config = read_external_alignment_config(args.alignment_dir)

    refit_axis_df, refit_summary_df, refit_endpoint_df = load_external_refit_tables(refit_tables_dir)
    internal_axis_df = load_internal_reference_axis_scores(internal_tables_dir)

    # Keep the original single-result plots for backward compatibility.
    plot_axis_scatter(locked_axis_df, locked_summary_df, locked_null_df, out_dir / "external_axis_scatter.png")
    plot_quartile_gradient(locked_axis_df, out_dir / "external_axis_quartile_gradient.png")
    plot_endpoint_auroc_forest(locked_endpoint_df, out_dir / "external_endpoint_auroc_forest.png")
    plot_permutation_null(locked_null_df, out_dir / "external_permutation_null.png")

    # New manuscript Figure 6 panels.
    fig6a = out_dir / "figure_06A_locked_external_alignment.png"
    fig6b = out_dir / "figure_06B_external_refit_alignment.png"
    fig6c = out_dir / "figure_06C_endpoint_auroc_locked_vs_refit.png"
    fig6combined = out_dir / "figure_06_external_validation_combined.png"

    plot_figure6A_locked_external_alignment(locked_axis_df, locked_summary_df, locked_null_df, fig6a)
    plot_figure6B_external_refit_alignment(refit_axis_df, refit_summary_df, fig6b)
    endpoint_forest_df = plot_figure6C_endpoint_auroc_locked_vs_refit(locked_endpoint_df, refit_endpoint_df, fig6c)
    if len(endpoint_forest_df):
        endpoint_forest_df.to_csv(table_out_dir / "figure_06C_endpoint_auroc_locked_vs_refit_source_data.csv", index=False, encoding="utf-8-sig")
    plot_figure6_combined(locked_axis_df, locked_summary_df, locked_null_df, refit_axis_df, refit_summary_df, endpoint_forest_df, fig6combined)

    # Additional external-validation diagnostic plots. These are additive and do
    # not alter the original Figure 6 or SI table workflow.
    fig_axis_shift = out_dir / "figure_S_external_axis_distribution_shift.png"
    fig_moving_enrichment = out_dir / "figure_S_external_moving_window_endpoint_enrichment.png"
    fig_endpoint_raincloud = out_dir / "figure_S_external_endpoint_axis_distribution_raincloud.png"
    fig_rank_agreement = out_dir / "figure_S_external_rank_percentile_agreement.png"

    axis_shift_source = plot_external_axis_distribution_shift(internal_axis_df, locked_axis_df, fig_axis_shift)
    if len(axis_shift_source):
        axis_shift_source.to_csv(table_out_dir / "figure_S_external_axis_distribution_shift_source_data.csv", index=False, encoding="utf-8-sig")

    moving_enrichment_source = plot_external_moving_window_endpoint_enrichment(
        locked_pred_df,
        fig_moving_enrichment,
        window_size=args.moving_window_size,
        step_size=args.moving_window_step,
    )
    if len(moving_enrichment_source):
        moving_enrichment_source.to_csv(table_out_dir / "figure_S_external_moving_window_endpoint_enrichment_source_data.csv", index=False, encoding="utf-8-sig")

    endpoint_raincloud_source = plot_external_endpoint_axis_distribution_raincloud(
        locked_pred_df,
        locked_endpoint_df,
        fig_endpoint_raincloud,
    )
    if len(endpoint_raincloud_source):
        endpoint_raincloud_source.to_csv(table_out_dir / "figure_S_external_endpoint_axis_distribution_raincloud_source_data.csv", index=False, encoding="utf-8-sig")

    rank_agreement_source = plot_external_rank_percentile_agreement(locked_axis_df, fig_rank_agreement)
    if len(rank_agreement_source):
        rank_agreement_source.to_csv(table_out_dir / "figure_S_external_rank_percentile_agreement_source_data.csv", index=False, encoding="utf-8-sig")

    # New SI tables.
    external_clinical = load_external_clinical_for_tables(args, locked_axis_df, locked_config)
    external_table1 = build_external_cohort_table1_style(external_clinical, locked_axis_df)
    external_table1_path = table_out_dir / "external_cohort_characteristics_table1_style.csv"
    external_table1.to_csv(external_table1_path, index=False, encoding="utf-8-sig")
    log(f"Saved SI table: {external_table1_path}")

    detailed_metrics = build_locked_external_detailed_metrics(locked_endpoint_df, locked_pred_df)
    detailed_metrics_path = table_out_dir / "locked_external_validation_detailed_metrics.csv"
    detailed_metrics.to_csv(detailed_metrics_path, index=False, encoding="utf-8-sig")
    log(f"Saved SI table: {detailed_metrics_path}")

    log("Done.")
    return {
        "figures_dir": out_dir,
        "table_out_dir": table_out_dir,
        "figure_06A": fig6a,
        "figure_06B": fig6b,
        "figure_06C": fig6c,
        "figure_06_combined": fig6combined,
        "figure_S_external_axis_distribution_shift": fig_axis_shift,
        "figure_S_external_moving_window_endpoint_enrichment": fig_moving_enrichment,
        "figure_S_external_endpoint_axis_distribution_raincloud": fig_endpoint_raincloud,
        "figure_S_external_rank_percentile_agreement": fig_rank_agreement,
        "external_cohort_characteristics_table1_style": external_table1_path,
        "locked_external_validation_detailed_metrics": detailed_metrics_path,
    }


if __name__ == "__main__":
    main()
