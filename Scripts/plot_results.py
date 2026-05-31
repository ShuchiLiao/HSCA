#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone plotting script for clinically anchored acoustic phenotyping outputs.

This script reads result CSV files produced by
run_clinically_anchored_acoustic_phenotyping_clean_v4.py and regenerates all
figures under the figures/ directory. It does not run any modeling, CCA,
endpoint classification, permutation test, or bootstrap-loading analysis.
"""

from __future__ import annotations

import argparse
import re
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
from sklearn.metrics import roc_auc_score, roc_curve

# =============================================================================
# Fixed clinical schema
# =============================================================================

PATIENT_ID_COL = "patient_id"

CLINICAL_PANELS: Dict[str, List[str]] = {
    "functional_impairment_hf_burden": ["EF_Teich", "NTproBNP", "NYHA"],
    "structural_remodeling": ["LA_mm", "LVEDD_mm", "IVS_mm", "LVPW_mm"],
    "valvular_regurgitation": ["MR_grade", "TR_grade", "AR_grade", "PR_grade"],
    "valvular_stenosis": ["AS_grade", "MS_grade"],
}

ALL_CLINICAL_VARS = [
    v for panel in CLINICAL_PANELS.values() for v in panel
]

COVARIATE_SETS: Dict[str, List[str]] = {
    "age_sex_residualized": ["age_years", "sex_male"],
    "age_sex_heart_rate_residualized": ["age_years", "sex_male", "heart_rate"],
}

# Higher values indicate heavier burden except EF.
BURDEN_DIRECTION: Dict[str, int] = {
    "EF_Teich": -1,
    "NTproBNP": +1,
    "NYHA": +1,
    "LA_mm": +1,
    "LVEDD_mm": +1,
    "IVS_mm": +1,
    "LVPW_mm": +1,
    "MR_grade": +1,
    "TR_grade": +1,
    "AR_grade": +1,
    "PR_grade": +1,
    "AS_grade": +1,
    "MS_grade": +1,
}

VARIABLE_DOMAIN: Dict[str, str] = {
    "EF_Teich": "Functional/HF burden",
    "NTproBNP": "Functional/HF burden",
    "NYHA": "Functional/HF burden",
    "LA_mm": "Structural remodeling",
    "LVEDD_mm": "Structural remodeling",
    "IVS_mm": "Structural remodeling",
    "LVPW_mm": "Structural remodeling",
    "MR_grade": "Valvular regurgitation",
    "TR_grade": "Valvular regurgitation",
    "AR_grade": "Valvular regurgitation",
    "PR_grade": "Valvular regurgitation",
    "AS_grade": "Valvular stenosis",
    "MS_grade": "Valvular stenosis",
}

PANEL_DOMAIN: Dict[str, str] = {
    "functional_impairment_hf_burden": "Functional/HF burden",
    "structural_remodeling": "Structural remodeling",
    "valvular_regurgitation": "Valvular regurgitation",
    "valvular_stenosis": "Valvular stenosis",
    "all_clinical": "All domains",
}

PANEL_PRETTY_LABELS: Dict[str, str] = {
    "functional_impairment_hf_burden": "Functional/HF burden",
    "structural_remodeling": "Structural remodeling",
    "valvular_regurgitation": "Valvular regurgitation",
    "valvular_stenosis": "Valvular stenosis",
    "all_clinical": "All clinical",
}

ORDINAL_VARS = {"NYHA", "MR_grade", "TR_grade", "AR_grade", "PR_grade", "AS_grade", "MS_grade"}
VALVE_VARS = {"MR_grade", "TR_grade", "AR_grade", "PR_grade", "AS_grade", "MS_grade"}

PRETTY_LABELS = {
    "EF_Teich": "EF",
    "NTproBNP": "NT-proBNP",
    "NYHA": "NYHA",
    "LA_mm": "LA",
    "LVEDD_mm": "LVEDD",
    "IVS_mm": "IVS",
    "LVPW_mm": "LVPW",
    "MR_grade": "MR grade",
    "TR_grade": "TR grade",
    "AR_grade": "AR grade",
    "PR_grade": "PR grade",
    "AS_grade": "AS grade",
    "MS_grade": "MS grade",
}



CROSS_DOMAIN_TARGETS: Dict[str, List[str]] = {
    "functional_impairment_hf_burden": CLINICAL_PANELS["structural_remodeling"],
    "structural_remodeling": CLINICAL_PANELS["functional_impairment_hf_burden"],
    "valvular_regurgitation": CLINICAL_PANELS["functional_impairment_hf_burden"] + CLINICAL_PANELS["structural_remodeling"],
}

# =============================================================================
# Minimal utilities used by plotting functions
# =============================================================================

def now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def clean_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", str(s))[:180]


def safe_spearman(x, y) -> Tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 5 or np.nanstd(x[mask]) < 1e-12 or np.nanstd(y[mask]) < 1e-12:
        return np.nan, np.nan, n
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p), n


def roc_curve_with_ci(y_true, y_score, n_boot: int, seed: int, grid: Optional[np.ndarray] = None):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if grid is None:
        grid = np.linspace(0, 1, 201)
    if len(y) < 20 or len(np.unique(y)) < 2:
        return grid, np.full_like(grid, np.nan), np.full_like(grid, np.nan), np.full_like(grid, np.nan)
    fpr, tpr, _ = roc_curve(y, s)
    tpr_obs = np.interp(grid, fpr, tpr)
    tpr_obs[0] = 0.0
    tpr_obs[-1] = 1.0
    rng = np.random.default_rng(seed)
    boot_tprs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        bfpr, btpr, _ = roc_curve(y[idx], s[idx])
        interp = np.interp(grid, bfpr, btpr)
        interp[0] = 0.0
        interp[-1] = 1.0
        boot_tprs.append(interp)
    if len(boot_tprs) < 20:
        lo = hi = np.full_like(grid, np.nan)
    else:
        B = np.vstack(boot_tprs)
        lo = np.percentile(B, 2.5, axis=0)
        hi = np.percentile(B, 97.5, axis=0)
    return grid, tpr_obs, lo, hi


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        log(f"Skip missing table: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        log(f"Loaded table: {path.name}, shape={df.shape}")
        return df
    except Exception as e:
        log(f"Failed to read {path}: {e}")
        return pd.DataFrame()


def _lookup_alignment_stats(summary_df: pd.DataFrame, panel: str, component: int = 1) -> str:
    if summary_df is None or len(summary_df) == 0:
        return ""
    if not {"panel", "component", "spearman_acoustic_vs_clinical_axis"}.issubset(summary_df.columns):
        return ""
    sub = summary_df[
        (summary_df["panel"].astype(str) == str(panel))
        & (summary_df["component"].astype(int) == int(component))
    ].copy()
    if "adjustment" in sub.columns:
        sub = sub[sub["adjustment"].astype(str).eq("none")]
    if len(sub) == 0:
        return ""
    row = sub.iloc[0]
    rho = row.get("spearman_acoustic_vs_clinical_axis", np.nan)
    lo = row.get("spearman_ci95_low", np.nan)
    hi = row.get("spearman_ci95_high", np.nan)
    p = row.get("permutation_p_abs", np.nan)
    parts = []
    if np.isfinite(rho):
        if np.isfinite(lo) and np.isfinite(hi):
            parts.append(f"ρ={rho:.2f} [{lo:.2f}, {hi:.2f}]")
        else:
            parts.append(f"ρ={rho:.2f}")
    if np.isfinite(p):
        parts.append(f"perm. p={p:.3g}")
    return ", ".join(parts)


def plot_axis_score_alignment_facets(
    score_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_path: Path,
    component: int = 1,
    include_all_clinical: bool = False,
) -> None:
    """Directly show CCA axis-to-axis alignment.

    Each panel plots the out-of-fold acoustic CCA score against the out-of-fold
    clinical CCA score for the same held-out patients. This is the most direct
    visual counterpart of the primary CCA alignment statistic and should be used
    as a main Results figure rather than only plotting axis-vs-single-variable
    associations.
    """
    if len(score_df) == 0:
        return
    xcol = f"cca_acoustic_axis{component}"
    ycol = f"cca_clinical_axis{component}"
    required = {"panel", "adjustment", xcol, ycol}
    if not required.issubset(score_df.columns):
        log(f"Skip axis alignment facets: missing columns {required - set(score_df.columns)}")
        return

    panels = list(CLINICAL_PANELS.keys())
    if include_all_clinical:
        panels = panels + ["all_clinical"]
    d = score_df[
        (score_df["adjustment"].astype(str) == "none")
        & (score_df["panel"].isin(panels))
    ].copy()
    d = d[np.isfinite(d[xcol]) & np.isfinite(d[ycol])]
    if len(d) == 0:
        return

    n_panels = len([p for p in panels if p in set(d["panel"])])
    n_cols = 2 if n_panels <= 4 else 3
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.1 * n_cols, 5.0 * n_rows), squeeze=False)
    pal = domain_palette()

    used = 0
    for ax, panel in zip(axes.ravel(), [p for p in panels if p in set(d["panel"])]):
        sub = d[d["panel"] == panel].copy()
        color = pal.get(PANEL_DOMAIN.get(panel), "0.45")
        ax.scatter(sub[xcol], sub[ycol], s=18, alpha=0.30, color=color, edgecolor="none")
        # Regression line is descriptive only; rho in title is Spearman.
        if len(sub) >= 5 and np.nanstd(sub[xcol]) > 1e-12 and np.nanstd(sub[ycol]) > 1e-12:
            xs = np.linspace(np.nanpercentile(sub[xcol], 1), np.nanpercentile(sub[xcol], 99), 100)
            slope, intercept = np.polyfit(sub[xcol], sub[ycol], 1)
            ax.plot(xs, slope * xs + intercept, color="black", lw=1.5)
        ax.axhline(0, color="0.80", lw=0.8)
        ax.axvline(0, color="0.80", lw=0.8)
        stats_txt = _lookup_alignment_stats(summary_df, panel, component)
        title = PANEL_PRETTY_LABELS.get(panel, panel)
        ax.set_title(title + (f"\n{stats_txt}" if stats_txt else ""))
        ax.set_xlabel(f"Acoustic CCA axis {component} score")
        ax.set_ylabel(f"Clinical CCA axis {component} score")
        used += 1

    for ax in axes.ravel()[used:]:
        ax.axis("off")
    fig.suptitle(
        f"Out-of-fold acoustic-axis vs. clinical-axis alignment, CCA axis {component}",
        y=1.02,
        fontsize=17,
    )
    savefig(out_path)


def plot_axis_score_alignment_joint_panels(
    score_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    fig_dir: Path,
    component: int = 1,
) -> None:
    """Save one direct acoustic-vs-clinical axis scatter per panel."""
    if len(score_df) == 0:
        return
    xcol = f"cca_acoustic_axis{component}"
    ycol = f"cca_clinical_axis{component}"
    if not {"panel", "adjustment", xcol, ycol}.issubset(score_df.columns):
        return
    pal = domain_palette()
    for panel in list(CLINICAL_PANELS.keys()) + ["all_clinical"]:
        sub = score_df[
            (score_df["panel"].astype(str) == panel)
            & (score_df["adjustment"].astype(str) == "none")
        ].copy()
        sub = sub[np.isfinite(sub[xcol]) & np.isfinite(sub[ycol])]
        if len(sub) == 0:
            continue
        color = pal.get(PANEL_DOMAIN.get(panel), "0.45")
        plt.figure(figsize=(6.8, 6.2))
        ax = plt.gca()
        ax.scatter(sub[xcol], sub[ycol], s=20, alpha=0.28, color=color, edgecolor="none")
        if len(sub) >= 5 and np.nanstd(sub[xcol]) > 1e-12 and np.nanstd(sub[ycol]) > 1e-12:
            xs = np.linspace(np.nanpercentile(sub[xcol], 1), np.nanpercentile(sub[xcol], 99), 100)
            slope, intercept = np.polyfit(sub[xcol], sub[ycol], 1)
            ax.plot(xs, slope * xs + intercept, color="black", lw=1.7)
        ax.axhline(0, color="0.82", lw=0.8)
        ax.axvline(0, color="0.82", lw=0.8)
        stats_txt = _lookup_alignment_stats(summary_df, panel, component)
        ax.set_title(
            f"{PANEL_PRETTY_LABELS.get(panel, panel)}: acoustic vs. clinical axis {component}"
            + (f"\n{stats_txt}" if stats_txt else "")
        )
        ax.set_xlabel(f"Out-of-fold acoustic CCA axis {component} score")
        ax.set_ylabel(f"Out-of-fold clinical CCA axis {component} score")
        savefig(fig_dir / f"{clean_filename(panel)}_axis{component}_acoustic_vs_clinical_axis_scatter.png")


def plot_axis_alignment_by_acoustic_quantile(
    score_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_path: Path,
    component: int = 1,
    include_all_clinical: bool = False,
) -> None:
    """Show how the clinical CCA axis changes across acoustic-axis quartiles.

    This plot is still axis-to-axis: patients are binned by acoustic CCA score,
    and the y-axis is the paired clinical CCA score, not individual clinical
    variables. It visually answers whether high acoustic-axis patients also have
    high clinical-axis scores.
    """
    if len(score_df) == 0:
        return
    xcol = f"cca_acoustic_axis{component}"
    ycol = f"cca_clinical_axis{component}"
    if not {"panel", "adjustment", xcol, ycol}.issubset(score_df.columns):
        return

    panels = list(CLINICAL_PANELS.keys())
    if include_all_clinical:
        panels = panels + ["all_clinical"]
    d = score_df[
        (score_df["adjustment"].astype(str) == "none")
        & (score_df["panel"].isin(panels))
    ].copy()
    d = d[np.isfinite(d[xcol]) & np.isfinite(d[ycol])]
    if len(d) == 0:
        return

    plot_rows = []
    for panel, sub in d.groupby("panel"):
        if len(sub) < 20:
            continue
        ranks = sub[xcol].rank(method="first")
        try:
            qs = pd.qcut(ranks, 4, labels=["Q1", "Q2", "Q3", "Q4"])
        except Exception:
            continue
        tmp = sub.copy()
        tmp["acoustic_axis_quantile"] = qs.astype(str)
        tmp["panel_label"] = tmp["panel"].map(PANEL_PRETTY_LABELS).fillna(tmp["panel"])
        tmp["domain"] = tmp["panel"].map(PANEL_DOMAIN).fillna("Other")
        plot_rows.append(tmp)
    if not plot_rows:
        return
    dd = pd.concat(plot_rows, ignore_index=True)
    panel_labels = [PANEL_PRETTY_LABELS.get(p, p) for p in panels if PANEL_PRETTY_LABELS.get(p, p) in set(dd["panel_label"])]
    pal = domain_palette()
    color_map = {PANEL_PRETTY_LABELS.get(p, p): pal.get(PANEL_DOMAIN.get(p), "0.5") for p in panels}

    plt.figure(figsize=(11.5, 6.0))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.boxplot(
            data=dd,
            x="panel_label",
            y=ycol,
            hue="acoustic_axis_quantile",
            order=panel_labels,
            showfliers=False,
            linewidth=1.1,
            ax=ax,
        )
        ax.legend(title="Acoustic axis\nquartile", frameon=False, loc="best", fontsize=9)
    else:
        # Compact fallback: draw group means with lines.
        for lab in panel_labels:
            sub = dd[dd["panel_label"] == lab]
            means = sub.groupby("acoustic_axis_quantile")[ycol].mean().reindex(["Q1", "Q2", "Q3", "Q4"])
            ax.plot(["Q1", "Q2", "Q3", "Q4"], means.values, marker="o", label=lab)
        ax.legend(frameon=False)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("")
    ax.set_ylabel(f"Clinical CCA axis {component} score")
    ax.set_title(f"Clinical-axis shift across acoustic-axis quartiles, axis {component}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=28, ha="right")
    savefig(out_path)


def plot_axis_alignment_effect_summary(
    score_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_path: Path,
    component: int = 1,
) -> None:
    """Summarize clinical-axis contrast between acoustic-axis Q4 and Q1.

    The y-axis is the standardized mean difference of clinical-axis score between
    patients in the highest vs. lowest acoustic-axis quartile.
    """
    if len(score_df) == 0:
        return
    xcol = f"cca_acoustic_axis{component}"
    ycol = f"cca_clinical_axis{component}"
    if not {"panel", "adjustment", xcol, ycol}.issubset(score_df.columns):
        return
    panels = list(CLINICAL_PANELS.keys()) + ["all_clinical"]
    rows = []
    rng = np.random.default_rng(123)
    for panel in panels:
        sub = score_df[
            (score_df["panel"].astype(str) == panel)
            & (score_df["adjustment"].astype(str) == "none")
        ].copy()
        sub = sub[np.isfinite(sub[xcol]) & np.isfinite(sub[ycol])]
        if len(sub) < 40:
            continue
        try:
            q = pd.qcut(sub[xcol].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
        except Exception:
            continue
        y1 = sub.loc[q.astype(str) == "Q1", ycol].to_numpy(float)
        y4 = sub.loc[q.astype(str) == "Q4", ycol].to_numpy(float)
        pooled = np.nanstd(sub[ycol].to_numpy(float))
        eff = (np.nanmean(y4) - np.nanmean(y1)) / pooled if pooled > 1e-12 else np.nan
        boots = []
        if len(y1) and len(y4):
            for _ in range(1000):
                b1 = rng.choice(y1, size=len(y1), replace=True)
                b4 = rng.choice(y4, size=len(y4), replace=True)
                boots.append((np.nanmean(b4) - np.nanmean(b1)) / pooled if pooled > 1e-12 else np.nan)
        rows.append({
            "panel": panel,
            "panel_label": PANEL_PRETTY_LABELS.get(panel, panel),
            "domain": PANEL_DOMAIN.get(panel, "Other"),
            "clinical_axis_q4_minus_q1_smd": eff,
            "ci_low": float(np.nanpercentile(boots, 2.5)) if boots else np.nan,
            "ci_high": float(np.nanpercentile(boots, 97.5)) if boots else np.nan,
        })
    d = pd.DataFrame(rows)
    if len(d) == 0:
        return
    pal = domain_palette()
    d["order"] = d["panel"].map({p: i for i, p in enumerate(panels)})
    d = d.sort_values("order")
    colors = [pal.get(dom, "0.45") for dom in d["domain"]]
    plt.figure(figsize=(8.8, max(4.6, 0.48 * len(d))))
    ax = plt.gca()
    y = np.arange(len(d))
    x = d["clinical_axis_q4_minus_q1_smd"].to_numpy(float)
    lo = d["ci_low"].to_numpy(float)
    hi = d["ci_high"].to_numpy(float)
    for yi, xi, li, hi_i, color in zip(y, x, lo, hi, colors):
        ax.plot([li, hi_i], [yi, yi], color=color, lw=2.2)
        ax.scatter([xi], [yi], color=color, edgecolor="black", s=85, zorder=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["panel_label"])
    ax.set_xlabel("Clinical-axis shift between acoustic-axis Q4 and Q1\n(standardized mean difference)")
    ax.set_title(f"Axis-to-axis alignment effect size, CCA axis {component}")
    savefig(out_path)

# =============================================================================
# Plotting functions extracted from v4
# =============================================================================

def plot_repeated_cca_alignment(values_df: pd.DataFrame, out_path: Path, component: int = 1) -> None:
    """Plot repeated random-split CCA alignment distributions."""
    if len(values_df) == 0:
        return
    d = values_df[(values_df["status"] == "ok") & (values_df["component"] == component)].copy()
    d = d[np.isfinite(d["spearman_acoustic_vs_clinical_axis"])]
    if len(d) == 0:
        return
    panel_order = list(CLINICAL_PANELS.keys()) + ["all_clinical"]
    d["panel_label"] = d["panel"].map(PANEL_PRETTY_LABELS).fillna(d["panel"])
    order_labels = [PANEL_PRETTY_LABELS.get(p, p) for p in panel_order if p in set(d["panel"])]
    pal = domain_palette()
    color_map = {PANEL_PRETTY_LABELS.get(p, p): pal.get(PANEL_DOMAIN.get(p), "0.5") for p in panel_order}

    plt.figure(figsize=(11.2, 5.8))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.boxplot(
            data=d, x="panel_label", y="spearman_acoustic_vs_clinical_axis",
            order=order_labels, palette=color_map, showfliers=False, linewidth=1.2, ax=ax,
        )
        sns.stripplot(
            data=d, x="panel_label", y="spearman_acoustic_vs_clinical_axis",
            order=order_labels, color="black", size=4, alpha=0.55, jitter=0.18, ax=ax,
        )
    else:
        groups = [d.loc[d["panel_label"] == lab, "spearman_acoustic_vs_clinical_axis"].dropna().to_numpy(float) for lab in order_labels]
        ax.boxplot(groups, labels=order_labels, showfliers=False)
        rng = np.random.default_rng(123)
        for i, vals in enumerate(groups, start=1):
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(vals)), vals, s=16, alpha=0.55, color="black")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("")
    ax.set_ylabel("OOF Spearman correlation")
    ax.set_title(f"Repeated random-split CCA robustness, axis {component}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    savefig(out_path)


def plot_repeated_endpoint_auroc(values_df: pd.DataFrame, out_path: Path) -> None:
    """Plot repeated StratifiedKFold endpoint AUROC distributions."""
    if len(values_df) == 0:
        return
    d = values_df[values_df["status"].eq("ok")].copy()
    d = d[np.isfinite(d["auroc"])]
    if len(d) == 0:
        return
    endpoint_order = ["EF_lt_40", "NTproBNP_ge_300", "NYHA_ge_3", "LA_ge_40", "LVEDD_dilated"]
    endpoint_order = [e for e in endpoint_order if e in set(d["endpoint"])]
    plt.figure(figsize=(10.5, 5.8))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.boxplot(
            data=d, x="endpoint", y="auroc",
            order=endpoint_order, color="white", showfliers=False, linewidth=1.3, ax=ax,
        )
        sns.stripplot(
            data=d, x="endpoint", y="auroc",
            order=endpoint_order, color="black", size=4, alpha=0.60, jitter=0.18, ax=ax,
        )
    else:
        groups = [d.loc[d["endpoint"] == lab, "auroc"].dropna().to_numpy(float) for lab in endpoint_order]
        ax.boxplot(groups, labels=endpoint_order, showfliers=False)
        rng = np.random.default_rng(123)
        for i, vals in enumerate(groups, start=1):
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(vals)), vals, s=16, alpha=0.6, color="black")
    ax.axhline(0.5, color="black", linestyle="--", lw=1)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("AUROC")
    ax.set_title("Repeated StratifiedKFold endpoint robustness")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    savefig(out_path)


# =============================================================================
# Plotting
# =============================================================================

def setup_plotting() -> None:
    if HAS_SEABORN:
        sns.set_theme(style="white", context="talk")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.spines.top"] = True
    plt.rcParams["axes.spines.right"] = True
    plt.rcParams["axes.grid"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"Saved figure: {path}")


def domain_palette():
    order = ["Functional/HF burden", "Structural remodeling", "Valvular regurgitation", "Valvular stenosis"]
    colors = sns.color_palette("Set2", n_colors=len(order)) if HAS_SEABORN else plt.cm.tab10(np.linspace(0, 1, len(order)))
    return dict(zip(order, colors))


def plot_single_variable_lollipop(df: pd.DataFrame, out_path: Path) -> None:
    d = df[df["status"] == "ok"].copy()
    d = d[np.isfinite(d["spearman_pred_true"])]
    if len(d) == 0:
        return
    d["abs_rho"] = d["spearman_pred_true"].abs()
    d = d.sort_values("abs_rho", ascending=False).sort_values("spearman_pred_true")
    pal = domain_palette()
    colors = [pal.get(x, "0.5") for x in d["domain"]]
    plt.figure(figsize=(10, max(5.5, 0.42 * len(d))))
    ax = plt.gca()
    y = np.arange(len(d))
    ax.hlines(y, 0, d["spearman_pred_true"], color=colors, lw=3)
    ax.scatter(d["spearman_pred_true"], y, c=colors, s=80, edgecolor="black", linewidth=0.5, zorder=3)
    lo = d["spearman_ci95_low"].to_numpy(float)
    hi = d["spearman_ci95_high"].to_numpy(float)
    x = d["spearman_pred_true"].to_numpy(float)
    ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt="none", color="black", capsize=2, lw=1)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([PRETTY_LABELS.get(v, v) for v in d["variable"]])
    ax.set_xlabel("Out-of-fold Spearman correlation\n(predicted vs. observed clinical variable)")
    ax.set_title("Single-variable acoustic readout")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=pal[k], markeredgecolor="black", label=k) for k in pal if k in set(d["domain"])]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=10)
    savefig(out_path)


def plot_alignment_forest(df: pd.DataFrame, out_path: Path) -> None:
    """Domain-specific CCA alignment forest plot colored by clinical domain."""
    d = df[(df["adjustment"] == "none") & (df["panel"].isin(list(CLINICAL_PANELS.keys())))].copy()
    d = d[np.isfinite(d["spearman_acoustic_vs_clinical_axis"])]
    if len(d) == 0:
        return
    d["panel_order"] = d["panel"].map({p: i for i, p in enumerate(CLINICAL_PANELS.keys())})
    d["domain"] = d["panel"].map(PANEL_DOMAIN)
    d = d.sort_values(["panel_order", "component"])
    d["label"] = d["panel"].map(PANEL_PRETTY_LABELS) + " axis " + d["component"].astype(str)
    pal = domain_palette()
    colors = [pal.get(x, "0.45") for x in d["domain"]]

    plt.figure(figsize=(10.5, max(5.8, 0.55 * len(d))))
    ax = plt.gca()
    y = np.arange(len(d))
    x = d["spearman_acoustic_vs_clinical_axis"].to_numpy(float)
    lo = d["spearman_ci95_low"].to_numpy(float)
    hi = d["spearman_ci95_high"].to_numpy(float)

    for yi, xi, li, hi_i, color in zip(y, x, lo, hi, colors):
        ax.plot([li, hi_i], [yi, yi], color=color, lw=2.3, alpha=0.95)
        ax.scatter([xi], [yi], s=95, color=color, edgecolor="black", linewidth=0.6, zorder=3)

    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.set_xlabel("Out-of-fold Spearman correlation\n(acoustic axis vs. clinical axis)")
    ax.set_title("Domain-specific CCA alignment")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=pal[k], markeredgecolor="black", markersize=9, label=k)
        for k in pal if k in set(d["domain"])
    ]
    if handles:
        ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=10)
    savefig(out_path)


def plot_axis_association_forest(df: pd.DataFrame, panel: str, out_path: Path, component: int = 1) -> None:
    """Per-panel axis-clinical association forest plot, colored by variable domain."""
    d = df[(df["panel"] == panel) & (df["adjustment"] == "none") & (df["component"] == component)].copy()
    d = d[np.isfinite(d["spearman_axis_variable"])]
    if len(d) == 0:
        return
    d = d.sort_values("spearman_axis_variable")
    pal = domain_palette()
    colors = [pal.get(x, "0.45") for x in d["domain"]]

    plt.figure(figsize=(9.2, max(4.8, 0.48 * len(d))))
    ax = plt.gca()
    y = np.arange(len(d))
    x = d["spearman_axis_variable"].to_numpy(float)
    lo = d["spearman_ci95_low"].to_numpy(float)
    hi = d["spearman_ci95_high"].to_numpy(float)

    for yi, xi, li, hi_i, color in zip(y, x, lo, hi, colors):
        ax.plot([li, hi_i], [yi, yi], color=color, lw=2.2, alpha=0.95)
        ax.scatter([xi], [yi], s=85, color=color, edgecolor="black", linewidth=0.6, zorder=3)

    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([PRETTY_LABELS.get(v, v) for v in d["variable"]])
    ax.set_xlabel("Spearman correlation with acoustic axis score")
    ax.set_title(f"Axis interpretation: {PANEL_PRETTY_LABELS.get(panel, panel)} axis {component}")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=pal[k], markeredgecolor="black", markersize=8, label=k)
        for k in pal if k in set(d["domain"])
    ]
    if handles and len(set(d["domain"])) > 1:
        ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=9)
    savefig(out_path)


def plot_axis_association_combined_forest(df: pd.DataFrame, out_path: Path, component: int = 1) -> None:
    """Combined axis association forest plot for all domain-specific panels."""
    d = df[
        (df["adjustment"] == "none")
        & (df["component"] == component)
        & (df["panel"].isin(list(CLINICAL_PANELS.keys())))
    ].copy()
    d = d[np.isfinite(d["spearman_axis_variable"])]
    if len(d) == 0:
        return
    panel_order = {p: i for i, p in enumerate(CLINICAL_PANELS.keys())}
    d["panel_order"] = d["panel"].map(panel_order)
    d["var_order"] = d["variable"].map({v: i for i, v in enumerate(ALL_CLINICAL_VARS)})
    d = d.sort_values(["panel_order", "var_order"])
    d["row_label"] = d["panel"].map(PANEL_PRETTY_LABELS) + " | " + d["variable"].map(lambda v: PRETTY_LABELS.get(v, v))
    pal = domain_palette()
    colors = [pal.get(x, "0.45") for x in d["domain"]]

    plt.figure(figsize=(10.8, max(6.0, 0.46 * len(d))))
    ax = plt.gca()
    y = np.arange(len(d))
    x = d["spearman_axis_variable"].to_numpy(float)
    lo = d["spearman_ci95_low"].to_numpy(float)
    hi = d["spearman_ci95_high"].to_numpy(float)
    for yi, xi, li, hi_i, color in zip(y, x, lo, hi, colors):
        ax.plot([li, hi_i], [yi, yi], color=color, lw=2.1, alpha=0.95)
        ax.scatter([xi], [yi], s=72, color=color, edgecolor="black", linewidth=0.5, zorder=3)
    # panel separator lines
    for _, sub in d.groupby("panel_order"):
        last = sub.index[-1]
    for pidx in sorted(d["panel_order"].unique())[1:]:
        first_y = int(np.where(d["panel_order"].to_numpy() == pidx)[0][0])
        ax.axhline(first_y - 0.5, color="0.85", lw=1)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["row_label"])
    ax.set_xlabel("Spearman correlation with domain-specific acoustic axis score")
    ax.set_title(f"Axis {component} clinical associations across domain-specific panels")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=pal[k], markeredgecolor="black", markersize=8, label=k)
        for k in pal if k in set(d["domain"])
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=9)
    savefig(out_path)


def plot_axis_association_dot_heatmap(df: pd.DataFrame, out_path: Path, component: int = 1) -> None:
    """Compact dot heatmap: rows are variables, columns are domain panels, color is rho."""
    d = df[
        (df["adjustment"] == "none")
        & (df["component"] == component)
        & (df["panel"].isin(list(CLINICAL_PANELS.keys())))
    ].copy()
    d = d[np.isfinite(d["spearman_axis_variable"])]
    if len(d) == 0:
        return
    panels = list(CLINICAL_PANELS.keys())
    variables = [v for v in ALL_CLINICAL_VARS if v in set(d["variable"])]
    y_map = {v: i for i, v in enumerate(variables)}
    x_map = {p: i for i, p in enumerate(panels)}
    pal = domain_palette()

    plt.figure(figsize=(9.5, max(5.8, 0.42 * len(variables))))
    ax = plt.gca()
    vals = d["spearman_axis_variable"].to_numpy(float)
    vmax = max(0.35, np.nanmax(np.abs(vals)) if len(vals) else 0.35)
    sc = ax.scatter(
        d["panel"].map(x_map),
        d["variable"].map(y_map),
        c=d["spearman_axis_variable"],
        s=np.clip(np.abs(d["spearman_axis_variable"]) * 520, 38, 210),
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        edgecolors=[pal.get(dom, "0.5") for dom in d["domain"]],
        linewidths=1.6,
    )
    ax.set_xticks(range(len(panels)))
    ax.set_xticklabels([PANEL_PRETTY_LABELS.get(p, p) for p in panels], rotation=35, ha="right")
    ax.set_yticks(range(len(variables)))
    ax.set_yticklabels([PRETTY_LABELS.get(v, v) for v in variables])
    # Color y tick labels by domain to match other plots.
    for tick, var in zip(ax.get_yticklabels(), variables):
        tick.set_color(pal.get(VARIABLE_DOMAIN.get(var), "black"))
    ax.set_xlabel("Domain-specific CCA panel")
    ax.set_ylabel("")
    ax.set_title(f"Axis {component} clinical association dot heatmap")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Spearman correlation")
    savefig(out_path)



def _axis_group_levels() -> List[str]:
    return ["Q1", "Q2", "Q3", "Q4"]


def _variable_label_with_direction(var: str) -> str:
    label = PRETTY_LABELS.get(var, var)
    direction = BURDEN_DIRECTION.get(var, 1)
    if direction < 0:
        return f"{label} (lower = higher burden)"
    return f"{label} (higher = higher burden)"


def _profile_values_from_gradient_summary(summary_df: pd.DataFrame, panel: str, component: int = 1) -> pd.DataFrame:
    """Build a domain-level clinical profile matrix from Q1-Q4 gradient summaries.

    Rows are the variables defining one clinical domain. Values are oriented
    row-standardized changes across acoustic-axis quartiles. Therefore, positive
    values always mean movement toward a heavier clinical burden direction
    (EF is reversed; other variables are positive).
    """
    if summary_df is None or len(summary_df) == 0:
        return pd.DataFrame()
    sub = summary_df[
        (summary_df["panel"].astype(str) == str(panel))
        & (summary_df["adjustment"].astype(str) == "none")
        & (summary_df["component"].astype(int) == int(component))
    ].copy()
    if len(sub) == 0:
        return pd.DataFrame()

    rows = []
    for var in [v for v in CLINICAL_PANELS.get(panel, []) if v in set(sub["variable"])]:
        sv = sub[sub["variable"].astype(str) == str(var)].copy()
        vals = []
        raw_vals = []
        for g in _axis_group_levels():
            hit = sv[sv["axis_group"].astype(str) == g]
            if len(hit) == 0:
                vals.append(np.nan)
                raw_vals.append(np.nan)
                continue
            if var in ORDINAL_VARS:
                # One clinically meaningful prevalence per ordinal variable.
                col = "prop_ge_3" if var == "NYHA" else "prop_ge_1"
                val = float(hit[col].iloc[0]) if col in hit.columns else np.nan
            else:
                val = float(hit["mean"].iloc[0]) if "mean" in hit.columns else np.nan
            raw_vals.append(val)
            vals.append(val)
        arr = np.asarray(vals, dtype=float)
        if np.isfinite(arr).sum() >= 2 and np.nanstd(arr) > 1e-12:
            oriented = (arr - np.nanmean(arr)) / np.nanstd(arr)
            oriented = oriented * BURDEN_DIRECTION.get(var, 1)
        else:
            oriented = np.zeros_like(arr, dtype=float)
        row = {
            "variable": var,
            "row_label": _variable_label_with_direction(var),
            "domain": VARIABLE_DOMAIN.get(var, "Other"),
            "trend_spearman": float(sv["trend_spearman"].iloc[0]) if "trend_spearman" in sv.columns and len(sv) else np.nan,
            "trend_spearman_p": float(sv["trend_spearman_p"].iloc[0]) if "trend_spearman_p" in sv.columns and len(sv) else np.nan,
        }
        for i, g in enumerate(_axis_group_levels()):
            row[g] = oriented[i]
            row[f"{g}_raw"] = raw_vals[i]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_multivariable_profile_heatmap(
    summary_df: pd.DataFrame,
    panel: str,
    out_path: Path,
    component: int = 1,
) -> None:
    """Show each domain-specific axis as a coordinated multivariable clinical profile.

    This plot is intended to replace the impression of isolated variable-wise
    correlations. It displays all variables in the anchoring clinical domain
    together, after orienting every row so that red means heavier burden.
    """
    prof = _profile_values_from_gradient_summary(summary_df, panel, component)
    if len(prof) == 0:
        return
    mat = prof[_axis_group_levels()].astype(float)
    vmax = max(1.6, np.nanmax(np.abs(mat.values)) if np.isfinite(mat.values).any() else 1.6)
    plt.figure(figsize=(8.5, max(4.2, 0.72 * len(prof))))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.heatmap(
            mat,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cmap="coolwarm",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            cbar_kws={"label": "Oriented row-z value\n(red = heavier clinical burden)"},
            ax=ax,
        )
        ax.set_yticklabels(prof["row_label"], rotation=0)
    else:
        im = ax.imshow(mat.values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label="Oriented row-z value")
        ax.set_xticks(range(4)); ax.set_xticklabels(_axis_group_levels())
        ax.set_yticks(range(len(prof))); ax.set_yticklabels(prof["row_label"])
    pal = domain_palette()
    for tick, dom in zip(ax.get_yticklabels(), prof["domain"]):
        tick.set_color(pal.get(dom, "black"))
    ax.set_xlabel(f"{PANEL_PRETTY_LABELS.get(panel, panel)} acoustic axis {component} quartile")
    ax.set_ylabel("")
    ax.set_title(
        f"Multivariable clinical profile along the {PANEL_PRETTY_LABELS.get(panel, panel)} acoustic axis {component}"
    )
    # Put trend statistics outside the heatmap; this is optional annotation, not the main visual message.
    for i, rec in prof.iterrows():
        rho = rec.get("trend_spearman", np.nan)
        pval = rec.get("trend_spearman_p", np.nan)
        txt = f"ρ={rho:.2f}, {_p_text(pval)}" if np.isfinite(rho) else ""
        ax.text(4.12, i + 0.5, txt, va="center", fontsize=9)
    savefig(out_path)


def _build_patient_profile_scores(patient_long: pd.DataFrame, panel: str, component: int = 1) -> pd.DataFrame:
    """Build a patient-level multivariable clinical burden/profile score.

    Values are z-scored within variable and oriented by clinical burden direction,
    then averaged across all variables in that domain. This is purely for
    visualization of the clinical profile carried by an acoustic axis.
    """
    if patient_long is None or len(patient_long) == 0:
        return pd.DataFrame()
    vars_ = CLINICAL_PANELS.get(panel, [])
    d = patient_long[
        (patient_long["panel"].astype(str) == str(panel))
        & (patient_long["adjustment"].astype(str) == "none")
        & (patient_long["component"].astype(int) == int(component))
        & (patient_long["variable"].isin(vars_))
    ].copy()
    if len(d) == 0:
        return pd.DataFrame()
    wide = d.pivot_table(index=PATIENT_ID_COL, columns="variable", values="value", aggfunc="mean")
    meta = d.groupby(PATIENT_ID_COL).agg({
        "axis_group": "first",
        "axis_score": "first",
    }).reset_index()
    prof_cols = []
    for var in vars_:
        if var not in wide.columns:
            continue
        x = wide[var].astype(float)
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd <= 1e-12:
            continue
        wide[f"{var}__oriented_z"] = ((x - mu) / sd) * BURDEN_DIRECTION.get(var, 1)
        prof_cols.append(f"{var}__oriented_z")
    if not prof_cols:
        return pd.DataFrame()
    wide["multivariable_clinical_profile_score"] = wide[prof_cols].mean(axis=1)
    out = meta.merge(
        wide[["multivariable_clinical_profile_score"]].reset_index(),
        on=PATIENT_ID_COL,
        how="inner",
    )
    out["panel"] = panel
    out["panel_label"] = PANEL_PRETTY_LABELS.get(panel, panel)
    out["domain"] = PANEL_DOMAIN.get(panel, "Other")
    return out


def plot_multivariable_profile_score_by_axis_quantile(
    patient_long: pd.DataFrame,
    panel: str,
    out_path: Path,
    component: int = 1,
) -> None:
    """Box + patient cloud for a composite clinical-domain profile score.

    Unlike axis-vs-variable plots, this shows the coordinated domain-level
    clinical profile carried by the acoustic axis.
    """
    prof = _build_patient_profile_scores(patient_long, panel, component)
    if len(prof) == 0:
        return
    levels = _axis_group_levels()
    prof["axis_group"] = pd.Categorical(prof["axis_group"].astype(str), categories=levels, ordered=True)
    color = domain_palette().get(PANEL_DOMAIN.get(panel), "0.5")
    plt.figure(figsize=(7.2, 5.6))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.boxplot(
            data=prof,
            x="axis_group",
            y="multivariable_clinical_profile_score",
            color=color,
            showfliers=False,
            linewidth=1.25,
            boxprops={"alpha": 0.38},
            medianprops={"color": "black", "linewidth": 1.35},
            ax=ax,
        )
        sns.stripplot(
            data=prof,
            x="axis_group",
            y="multivariable_clinical_profile_score",
            color=color,
            size=2.4,
            alpha=0.25,
            jitter=0.24,
            ax=ax,
        )
    else:
        rng = np.random.default_rng(123)
        groups = [prof.loc[prof["axis_group"].astype(str) == g, "multivariable_clinical_profile_score"].dropna().to_numpy(float) for g in levels]
        ax.boxplot(groups, labels=levels, showfliers=False)
        for i, vals in enumerate(groups, start=1):
            ax.scatter(i + rng.uniform(-0.22, 0.22, len(vals)), vals, s=8, alpha=0.28, color=color)
    rho, p, _ = safe_spearman(prof["axis_score"], prof["multivariable_clinical_profile_score"])
    ax.axhline(0, color="0.75", lw=0.8)
    ax.set_xlabel(f"{PANEL_PRETTY_LABELS.get(panel, panel)} acoustic axis {component} quartile")
    ax.set_ylabel("Composite clinical-domain profile score\n(mean oriented z-score)")
    title = f"Coordinated clinical profile along the {PANEL_PRETTY_LABELS.get(panel, panel)} acoustic axis {component}"
    if np.isfinite(rho):
        title += f"\nprofile-score trend: ρ={rho:.2f}, {_p_text(p)}"
    ax.set_title(title)
    savefig(out_path)


def plot_multivariable_profile_score_facets(
    patient_long: pd.DataFrame,
    out_path: Path,
    component: int = 1,
) -> None:
    """Small-multiple composite profile score plot across domain-specific axes."""
    pieces = []
    for panel in CLINICAL_PANELS:
        prof = _build_patient_profile_scores(patient_long, panel, component)
        if len(prof):
            pieces.append(prof)
    if not pieces:
        return
    d = pd.concat(pieces, ignore_index=True)
    order = [PANEL_PRETTY_LABELS.get(p, p) for p in CLINICAL_PANELS if PANEL_PRETTY_LABELS.get(p, p) in set(d["panel_label"])]
    pal = domain_palette()
    color_map = {PANEL_PRETTY_LABELS.get(p, p): pal.get(PANEL_DOMAIN.get(p), "0.5") for p in CLINICAL_PANELS}

    plt.figure(figsize=(11.5, 6.2))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.boxplot(
            data=d,
            x="panel_label",
            y="multivariable_clinical_profile_score",
            hue="axis_group",
            order=order,
            showfliers=False,
            linewidth=1.15,
            ax=ax,
        )
        ax.legend(title="Acoustic axis\nquartile", frameon=False, fontsize=9)
    else:
        for lab in order:
            sub = d[d["panel_label"] == lab]
            means = sub.groupby("axis_group")["multivariable_clinical_profile_score"].mean().reindex(_axis_group_levels())
            ax.plot(_axis_group_levels(), means.values, marker="o", label=lab)
        ax.legend(frameon=False)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("Composite clinical-domain profile score\n(mean oriented z-score)")
    ax.set_title(f"Domain-level clinical profiles across acoustic-axis quartiles, axis {component}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=28, ha="right")
    savefig(out_path)


def _p_text(p: float) -> str:
    if not np.isfinite(p):
        return "p=NA"
    if p < 1e-4:
        return "p<1e-4"
    if p < 0.001:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


def plot_gradient_continuous_box_points(
    patient_long: pd.DataFrame,
    summary_df: pd.DataFrame,
    panel: str,
    out_path: Path,
    component: int = 1,
    max_vars: int = 8,
) -> None:
    """Clinical gradient plot for continuous variables only: box + patient point cloud."""
    d = patient_long[
        (patient_long["panel"] == panel)
        & (patient_long["adjustment"] == "none")
        & (patient_long["component"] == component)
        & (~patient_long["is_ordinal"].astype(bool))
    ].copy()
    if len(d) == 0:
        return
    vars_order = [
        v for v in CLINICAL_PANELS.get(panel, ALL_CLINICAL_VARS)
        if v in set(d["variable"]) and v not in ORDINAL_VARS
    ][:max_vars]
    if not vars_order:
        return
    n = len(vars_order)
    n_cols = 2 if n > 2 else 1
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.9 * n_cols, 4.35 * n_rows), squeeze=False)
    pal = domain_palette()
    axis_levels = [f"Q{i+1}" for i in range(4)]

    for ax, var in zip(axes.ravel(), vars_order):
        sub = d[d["variable"] == var].copy()
        sub["axis_group"] = pd.Categorical(sub["axis_group"], categories=axis_levels, ordered=True)
        color = pal.get(VARIABLE_DOMAIN.get(var), "0.5")
        if HAS_SEABORN:
            sns.boxplot(
                data=sub, x="axis_group", y="value", ax=ax,
                color=color, showfliers=False, linewidth=1.25,
                boxprops={"alpha": 0.38}, medianprops={"color": "black", "linewidth": 1.3},
                whiskerprops={"color": "0.25"}, capprops={"color": "0.25"},
            )
            sns.stripplot(
                data=sub, x="axis_group", y="value", ax=ax,
                color=color, size=2.4, alpha=0.26, jitter=0.24
            )
        else:
            rng = np.random.default_rng(123)
            groups = [sub.loc[sub["axis_group"] == g, "value"].dropna().to_numpy(float) for g in axis_levels]
            ax.boxplot(groups, labels=axis_levels, showfliers=False)
            for i, vals in enumerate(groups, start=1):
                ax.scatter(i + rng.uniform(-0.22, 0.22, len(vals)), vals, s=8, alpha=0.28, color=color)
        ss = summary_df[
            (summary_df["panel"] == panel)
            & (summary_df["adjustment"] == "none")
            & (summary_df["component"] == component)
            & (summary_df["variable"] == var)
        ]
        if len(ss):
            rho = ss["trend_spearman"].iloc[0]
            pval = ss["trend_spearman_p"].iloc[0]
            kwp = ss["kruskal_p"].iloc[0]
            subtitle = f"ρ={rho:.2f}, trend {_p_text(pval)}, KW {_p_text(kwp)}" if np.isfinite(rho) else ""
        else:
            rho, _, _ = safe_spearman(sub["axis_score"], sub["value"])
            subtitle = f"ρ={rho:.2f}" if np.isfinite(rho) else ""
        ax.set_title(f"{PRETTY_LABELS.get(var, var)}" + (f"\n{subtitle}" if subtitle else ""))
        ax.set_xlabel(f"CCA axis {component} quantile")
        ax.set_ylabel("Clinical value")

    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Continuous clinical gradients: {PANEL_PRETTY_LABELS.get(panel, panel)} axis {component}", y=1.02, fontsize=16)
    savefig(out_path)


def plot_gradient_ordinal_heatmap(
    summary_df: pd.DataFrame,
    panel: str,
    out_path: Path,
    component: int = 1,
) -> None:
    """Clinical gradient plot for ordinal variables only.

    NYHA is shown as proportion with NYHA ≥3. Valve grades are shown as
    proportion with grade ≥1 and grade ≥2, because medians are often zero.
    """
    d = summary_df[
        (summary_df["panel"] == panel)
        & (summary_df["adjustment"] == "none")
        & (summary_df["component"] == component)
        & (summary_df["variable"].isin(ORDINAL_VARS))
    ].copy()
    if len(d) == 0:
        return
    rows = []
    for var in [v for v in CLINICAL_PANELS.get(panel, ALL_CLINICAL_VARS) if v in set(d["variable"]) and v in ORDINAL_VARS]:
        sub = d[d["variable"] == var].copy()
        if len(sub) == 0:
            continue
        if var == "NYHA":
            metrics = [("NYHA ≥3", "prop_ge_3")]
        else:
            metrics = [("grade ≥1", "prop_ge_1"), ("grade ≥2", "prop_ge_2")]
        for label, col in metrics:
            row_label = f"{PRETTY_LABELS.get(var, var)} {label}"
            rec = {"row_label": row_label, "variable": var, "domain": VARIABLE_DOMAIN.get(var, "Other")}
            for g in [f"Q{i+1}" for i in range(4)]:
                hit = sub[sub["axis_group"].astype(str) == g]
                rec[g] = float(hit[col].iloc[0]) if len(hit) and col in hit.columns else np.nan
            ss = sub.iloc[0]
            rec["trend_spearman"] = ss.get("trend_spearman", np.nan)
            rec["trend_spearman_p"] = ss.get("trend_spearman_p", np.nan)
            rec["kruskal_p"] = ss.get("kruskal_p", np.nan)
            rows.append(rec)
    if not rows:
        return
    mat_df = pd.DataFrame(rows)
    mat = mat_df[[f"Q{i+1}" for i in range(4)]].astype(float)
    plt.figure(figsize=(7.8, max(3.8, 0.52 * len(mat_df))))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.heatmap(
            mat,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            vmin=0,
            vmax=max(0.05, min(1.0, np.nanmax(mat.values) if np.isfinite(mat.values).any() else 1.0)),
            cmap="Blues",
            cbar_kws={"label": "Proportion"},
            ax=ax,
        )
        ax.set_yticklabels(mat_df["row_label"], rotation=0)
    else:
        im = ax.imshow(mat.values, aspect="auto", vmin=0, vmax=max(0.05, np.nanmax(mat.values)))
        plt.colorbar(im, ax=ax, label="Proportion")
        ax.set_xticks(range(4)); ax.set_xticklabels([f"Q{i+1}" for i in range(4)])
        ax.set_yticks(range(len(mat_df))); ax.set_yticklabels(mat_df["row_label"])
    pal = domain_palette()
    for tick, dom in zip(ax.get_yticklabels(), mat_df["domain"]):
        tick.set_color(pal.get(dom, "black"))
    ax.set_xlabel(f"CCA axis {component} quantile")
    ax.set_ylabel("")
    ax.set_title(f"Ordinal clinical gradients: {PANEL_PRETTY_LABELS.get(panel, panel)} axis {component}")
    # Add compact significance text on the right.
    for i, rec in mat_df.iterrows():
        ptxt = f"ρ={rec['trend_spearman']:.2f}, {_p_text(rec['trend_spearman_p'])}" if np.isfinite(rec.get("trend_spearman", np.nan)) else ""
        ax.text(4.08, i + 0.5, ptxt, va="center", fontsize=9)
    savefig(out_path)


def plot_endpoint_roc_curves(pred_df: pd.DataFrame, summary_df: pd.DataFrame, fig_dir: Path, args) -> None:
    if len(pred_df) == 0:
        return
    for endpoint, sub in pred_df.groupby("endpoint"):
        if len(sub) == 0 or sub["y_true"].nunique() < 2:
            continue
        y = sub["y_true"].astype(int).to_numpy()
        p = sub["y_prob"].astype(float).to_numpy()
        fpr, tpr, lo, hi = roc_curve_with_ci(y, p, args.n_bootstrap, args.seed + 9000)
        auc = roc_auc_score(y, p)
        row = summary_df[summary_df["endpoint"] == endpoint]
        if len(row):
            auc_lo = row["auroc_ci95_low"].iloc[0]
            auc_hi = row["auroc_ci95_high"].iloc[0]
        else:
            auc_lo, auc_hi = np.nan, np.nan
        plt.figure(figsize=(6.5, 6))
        ax = plt.gca()
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", lw=1)
        if np.isfinite(lo).any():
            ax.fill_between(fpr, lo, hi, alpha=0.25, label="95% bootstrap CI")
        ax.plot(fpr, tpr, lw=2.5, label=f"AUROC={auc:.3f}" + (f" ({auc_lo:.3f}–{auc_hi:.3f})" if np.isfinite(auc_lo) else ""))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC curve: {endpoint}")
        ax.legend(frameon=False, loc="lower right")
        savefig(fig_dir / f"roc_curve_{clean_filename(endpoint)}.png")


def plot_endpoint_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    d = summary_df[np.isfinite(summary_df.get("auroc", np.nan))].copy()
    if len(d) == 0:
        return
    d = d.sort_values("endpoint")
    plt.figure(figsize=(9, max(4.5, 0.55 * len(d))))
    ax = plt.gca()
    y = np.arange(len(d))
    x = d["auroc"].to_numpy(float)
    lo = d["auroc_ci95_low"].to_numpy(float)
    hi = d["auroc_ci95_high"].to_numpy(float)
    ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt="o", capsize=3)
    ax.axvline(0.5, color="black", linestyle="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["endpoint"])
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("AUROC")
    ax.set_title("Endpoint validation using CCA acoustic axes 1+2")
    savefig(out_path)


def plot_confounder(summary_df: pd.DataFrame, unadjusted: pd.DataFrame, out_path: Path) -> None:
    rows = []
    base = unadjusted[(unadjusted["panel"] == "all_clinical") & (unadjusted["adjustment"] == "none") & (unadjusted["component"] == 1)]
    if len(base):
        row = base.iloc[0].to_dict()
        row["plot_label"] = "Unadjusted"
        rows.append(row)
    for label, nice in [("age_sex_residualized", "Age + sex"), ("age_sex_heart_rate_residualized", "Age + sex + heart rate")]:
        sub = summary_df[(summary_df["adjustment"] == label) & (summary_df["component"] == 1)]
        if len(sub):
            row = sub.iloc[0].to_dict()
            row["plot_label"] = nice
            rows.append(row)
    d = pd.DataFrame(rows)
    if len(d) == 0:
        return
    plt.figure(figsize=(8, 4.8))
    ax = plt.gca()
    y = np.arange(len(d))
    x = d["spearman_acoustic_vs_clinical_axis"].to_numpy(float)
    lo = d["spearman_ci95_low"].to_numpy(float)
    hi = d["spearman_ci95_high"].to_numpy(float)
    ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt="o", capsize=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["plot_label"])
    ax.set_xlabel("OOF Spearman correlation\n(all-clinical acoustic vs. clinical axis)")
    ax.set_title("Confounder-adjusted CCA alignment")
    savefig(out_path)


def plot_lopo(summary_df: pd.DataFrame, unadjusted: pd.DataFrame, out_path: Path) -> None:
    d = summary_df[summary_df["component"] == 1].copy()
    if len(d) == 0:
        return
    d = d.sort_values("left_out_position")
    full = unadjusted[(unadjusted["panel"] == "all_clinical") & (unadjusted["adjustment"] == "none") & (unadjusted["component"] == 1)]
    full_val = full["spearman_acoustic_vs_clinical_axis"].iloc[0] if len(full) else np.nan
    plt.figure(figsize=(8.5, 5))
    ax = plt.gca()
    y = np.arange(len(d))
    x = d["spearman_acoustic_vs_clinical_axis"].to_numpy(float)
    lo = d["spearman_ci95_low"].to_numpy(float)
    hi = d["spearman_ci95_high"].to_numpy(float)
    ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt="o", capsize=3)
    if np.isfinite(full_val):
        ax.axvline(full_val, color="black", linestyle="--", lw=1, label="Full A/E/M/P/T")
        ax.legend(frameon=False)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"Leave {p}" for p in d["left_out_position"]])
    ax.set_xlabel("OOF Spearman correlation\n(all-clinical acoustic vs. clinical axis)")
    ax.set_title("Leave-one-auscultation-position-out control")
    savefig(out_path)


def plot_negative_controls(ctrl_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    d = ctrl_df[ctrl_df["panel"] == "all_clinical"].copy()
    if len(d) == 0:
        return
    observed = summary_df.loc[(summary_df["panel"] == "all_clinical") & (summary_df["control_type"] == "patient_label_permutation"), "observed_spearman"]
    obs = observed.iloc[0] if len(observed) else np.nan
    plt.figure(figsize=(8, 5.5))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.histplot(data=d, x="spearman", hue="control_type", bins=25, element="step", stat="count", common_norm=False, ax=ax)
    else:
        for ct, sub in d.groupby("control_type"):
            ax.hist(sub["spearman"].dropna(), bins=25, alpha=0.5, label=ct)
        ax.legend()
    if np.isfinite(obs):
        ax.axvline(obs, color="black", lw=2, label="Observed")
    ax.set_xlabel("Control OOF Spearman correlation")
    ax.set_title("Negative controls for all-clinical CCA axis 1")
    savefig(out_path)



def plot_leave_one_variable_out_forest(summary_df: pd.DataFrame, out_path: Path, component: int = 1) -> None:
    """Forest plot for leave-one-variable-out CCA held-out-variable correlations."""
    if len(summary_df) == 0:
        return
    d = summary_df[(summary_df["status"] == "ok") & (summary_df["component"] == component)].copy()
    d = d[np.isfinite(d["spearman_acoustic_axis_vs_heldout"])]
    if len(d) == 0:
        return
    panel_order = {p: i for i, p in enumerate(CLINICAL_PANELS.keys())}
    d["panel_order"] = d["source_panel"].map(panel_order)
    d["var_order"] = d["heldout_variable"].map({v: i for i, v in enumerate(ALL_CLINICAL_VARS)})
    d = d.sort_values(["panel_order", "var_order"])
    d["row_label"] = d["source_panel"].map(PANEL_PRETTY_LABELS) + " | leave out " + d["heldout_variable"].map(lambda v: PRETTY_LABELS.get(v, v))
    pal = domain_palette()
    colors = [pal.get(x, "0.45") for x in d["heldout_domain"]]

    plt.figure(figsize=(11.2, max(6, 0.52 * len(d))))
    ax = plt.gca()
    y = np.arange(len(d))
    x = d["spearman_acoustic_axis_vs_heldout"].to_numpy(float)
    lo = d["spearman_ci95_low"].to_numpy(float)
    hi = d["spearman_ci95_high"].to_numpy(float)
    for yi, xi, li, hi_i, color in zip(y, x, lo, hi, colors):
        ax.plot([li, hi_i], [yi, yi], color=color, lw=2.2, alpha=0.95)
        ax.scatter([xi], [yi], color=color, edgecolor="black", linewidth=0.55, s=78, zorder=3)
    for pidx in sorted(d["panel_order"].dropna().unique())[1:]:
        first_y = int(np.where(d["panel_order"].to_numpy() == pidx)[0][0])
        ax.axhline(first_y - 0.5, color="0.86", lw=1)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["row_label"])
    ax.set_xlabel("Out-of-fold Spearman correlation\n(acoustic axis vs. left-out clinical variable)")
    ax.set_title(f"Leave-one-variable-out CCA validation, axis {component}")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=pal[k], markeredgecolor="black", markersize=8, label=k)
        for k in pal if k in set(d["heldout_domain"])
    ]
    if handles:
        ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=9)
    savefig(out_path)


def plot_leave_one_variable_out_gradient_heatmap(gsum_df: pd.DataFrame, out_path: Path, component: int = 1) -> None:
    """Compact gradient heatmap for held-out variables in leave-one-variable-out CCA."""
    if len(gsum_df) == 0:
        return
    rows = []
    for (source_panel, heldout), sub in gsum_df[(gsum_df["component"] == component)].groupby(["source_panel", "heldout_variable"]):
        sub = sub.copy()
        if len(sub) == 0:
            continue
        is_ord = bool(sub["is_ordinal"].iloc[0]) if "is_ordinal" in sub.columns else heldout in ORDINAL_VARS
        if is_ord:
            col = "prop_ge_3" if heldout == "NYHA" else "prop_ge_1"
            metric = "NYHA ≥3" if heldout == "NYHA" else "grade ≥1"
            raw = [float(sub.loc[sub["axis_group"].astype(str) == f"Q{i+1}", col].iloc[0]) if any(sub["axis_group"].astype(str) == f"Q{i+1}") else np.nan for i in range(4)]
            vals = raw
            value_type = "proportion"
        else:
            metric = "mean"
            raw = [float(sub.loc[sub["axis_group"].astype(str) == f"Q{i+1}", "mean"].iloc[0]) if any(sub["axis_group"].astype(str) == f"Q{i+1}") else np.nan for i in range(4)]
            arr = np.asarray(raw, dtype=float)
            sd = np.nanstd(arr)
            vals = ((arr - np.nanmean(arr)) / sd).tolist() if np.isfinite(sd) and sd > 1e-12 else [0, 0, 0, 0]
            value_type = "row_z_mean"
        rows.append({
            "row_label": f"{PANEL_PRETTY_LABELS.get(source_panel, source_panel)} | {PRETTY_LABELS.get(heldout, heldout)} ({metric})",
            "source_panel": source_panel,
            "heldout_variable": heldout,
            "domain": VARIABLE_DOMAIN.get(heldout, "Other"),
            "value_type": value_type,
            "Q1": vals[0], "Q2": vals[1], "Q3": vals[2], "Q4": vals[3],
            "trend_spearman": sub["trend_spearman"].iloc[0] if "trend_spearman" in sub.columns else np.nan,
            "trend_spearman_p": sub["trend_spearman_p"].iloc[0] if "trend_spearman_p" in sub.columns else np.nan,
        })
    if not rows:
        return
    mat_df = pd.DataFrame(rows)
    mat = mat_df[["Q1", "Q2", "Q3", "Q4"]].astype(float)
    plt.figure(figsize=(8.6, max(5.2, 0.52 * len(mat_df))))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.heatmap(
            mat,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cmap="coolwarm",
            center=0,
            cbar_kws={"label": "Row-z mean for continuous; raw proportion for ordinal"},
            ax=ax,
        )
        ax.set_yticklabels(mat_df["row_label"], rotation=0)
    else:
        im = ax.imshow(mat.values, aspect="auto")
        plt.colorbar(im, ax=ax, label="Display value")
        ax.set_xticks(range(4)); ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        ax.set_yticks(range(len(mat_df))); ax.set_yticklabels(mat_df["row_label"])
    pal = domain_palette()
    for tick, dom in zip(ax.get_yticklabels(), mat_df["domain"]):
        tick.set_color(pal.get(dom, "black"))
    ax.set_title(f"Leave-one-variable-out gradients, axis {component}")
    ax.set_xlabel("CCA acoustic-axis quantile")
    ax.set_ylabel("")
    for i, rec in mat_df.iterrows():
        ptxt = f"ρ={rec['trend_spearman']:.2f}, {_p_text(rec['trend_spearman_p'])}" if np.isfinite(rec.get("trend_spearman", np.nan)) else ""
        ax.text(4.08, i + 0.5, ptxt, va="center", fontsize=8.8)
    savefig(out_path)


def plot_cross_domain_forest(cross_df: pd.DataFrame, out_path: Path, component: int = 1) -> None:
    """Forest plot for cross-domain characterization."""
    if len(cross_df) == 0:
        return
    d = cross_df[(cross_df["component"] == component)].copy()
    d = d[np.isfinite(d["spearman_acoustic_axis_vs_target"])]
    if len(d) == 0:
        return
    source_order = {p: i for i, p in enumerate(CROSS_DOMAIN_TARGETS.keys())}
    d["source_order"] = d["source_panel"].map(source_order)
    d["target_order"] = d["target_variable"].map({v: i for i, v in enumerate(ALL_CLINICAL_VARS)})
    d = d.sort_values(["source_order", "target_order"])
    d["row_label"] = d["source_panel"].map(PANEL_PRETTY_LABELS) + " axis → " + d["target_variable"].map(lambda v: PRETTY_LABELS.get(v, v))
    pal = domain_palette()
    colors = [pal.get(x, "0.45") for x in d["target_domain"]]

    plt.figure(figsize=(11.2, max(6, 0.46 * len(d))))
    ax = plt.gca()
    y = np.arange(len(d))
    x = d["spearman_acoustic_axis_vs_target"].to_numpy(float)
    lo = d["spearman_ci95_low"].to_numpy(float)
    hi = d["spearman_ci95_high"].to_numpy(float)
    for yi, xi, li, hi_i, color in zip(y, x, lo, hi, colors):
        ax.plot([li, hi_i], [yi, yi], color=color, lw=2.1, alpha=0.95)
        ax.scatter([xi], [yi], color=color, edgecolor="black", linewidth=0.55, s=72, zorder=3)
    for sidx in sorted(d["source_order"].dropna().unique())[1:]:
        first_y = int(np.where(d["source_order"].to_numpy() == sidx)[0][0])
        ax.axhline(first_y - 0.5, color="0.86", lw=1)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["row_label"])
    ax.set_xlabel("Spearman correlation with non-anchoring clinical variable")
    ax.set_title(f"Cross-domain clinical characterization, axis {component}")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=pal[k], markeredgecolor="black", markersize=8, label=k)
        for k in pal if k in set(d["target_domain"])
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=9)
    savefig(out_path)


def plot_cross_domain_dot_heatmap(cross_df: pd.DataFrame, out_path: Path, component: int = 1) -> None:
    """Dot heatmap for cross-domain characterization."""
    if len(cross_df) == 0:
        return
    d = cross_df[cross_df["component"] == component].copy()
    d = d[np.isfinite(d["spearman_acoustic_axis_vs_target"])]
    if len(d) == 0:
        return
    sources = list(CROSS_DOMAIN_TARGETS.keys())
    targets = [v for v in ALL_CLINICAL_VARS if v in set(d["target_variable"])]
    x_map = {p: i for i, p in enumerate(sources)}
    y_map = {v: i for i, v in enumerate(targets)}
    pal = domain_palette()
    vals = d["spearman_acoustic_axis_vs_target"].to_numpy(float)
    vmax = max(0.35, np.nanmax(np.abs(vals)) if len(vals) else 0.35)

    plt.figure(figsize=(8.8, max(5.6, 0.43 * len(targets))))
    ax = plt.gca()
    sc = ax.scatter(
        d["source_panel"].map(x_map),
        d["target_variable"].map(y_map),
        c=d["spearman_acoustic_axis_vs_target"],
        s=np.clip(np.abs(d["spearman_acoustic_axis_vs_target"]) * 520, 38, 210),
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        edgecolors=[pal.get(dom, "0.5") for dom in d["target_domain"]],
        linewidths=1.5,
    )
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([PANEL_PRETTY_LABELS.get(p, p) for p in sources], rotation=35, ha="right")
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels([PRETTY_LABELS.get(v, v) for v in targets])
    for tick, var in zip(ax.get_yticklabels(), targets):
        tick.set_color(pal.get(VARIABLE_DOMAIN.get(var), "black"))
    ax.set_xlabel("Source CCA acoustic axis")
    ax.set_ylabel("Non-anchoring target clinical variable")
    ax.set_title(f"Cross-domain clinical association dot heatmap, axis {component}")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Spearman correlation")
    savefig(out_path)



# =============================================================================
# Plot orchestration
# =============================================================================

def plot_all_results(table_dir: Path, fig_dir: Path, args) -> None:
    ensure_dir(fig_dir)

    single_df = read_csv_optional(table_dir / "single_variable_readout.csv")
    if len(single_df):
        plot_single_variable_lollipop(single_df, fig_dir / "single_variable_readout_lollipop_by_domain.png")

    alignment_summary = read_csv_optional(table_dir / "cca_panel_alignment_summary.csv")
    if len(alignment_summary):
        plot_alignment_forest(alignment_summary, fig_dir / "domain_specific_cca_alignment_forest.png")

    axis_scores = read_csv_optional(table_dir / "oof_cca_axis_scores_by_panel.csv")
    if len(axis_scores):
        plot_axis_score_alignment_facets(
            axis_scores,
            alignment_summary,
            fig_dir / "axis1_acoustic_vs_clinical_axis_alignment_facets.png",
            component=1,
            include_all_clinical=False,
        )
        plot_axis_score_alignment_facets(
            axis_scores,
            alignment_summary,
            fig_dir / "axis1_acoustic_vs_clinical_axis_alignment_facets_with_all_clinical.png",
            component=1,
            include_all_clinical=True,
        )
        plot_axis_score_alignment_joint_panels(
            axis_scores,
            alignment_summary,
            fig_dir,
            component=1,
        )
        plot_axis_alignment_by_acoustic_quantile(
            axis_scores,
            alignment_summary,
            fig_dir / "axis1_clinical_axis_by_acoustic_axis_quartile.png",
            component=1,
            include_all_clinical=False,
        )
        plot_axis_alignment_effect_summary(
            axis_scores,
            alignment_summary,
            fig_dir / "axis1_alignment_effect_q4_minus_q1_summary.png",
            component=1,
        )

    assoc_all = read_csv_optional(table_dir / "cca_axis_clinical_associations.csv")
    if len(assoc_all):
        for panel in CLINICAL_PANELS:
            plot_axis_association_forest(
                assoc_all,
                panel,
                fig_dir / f"{clean_filename(panel)}_axis1_clinical_association_forest.png",
                component=1,
            )
        plot_axis_association_combined_forest(
            assoc_all,
            fig_dir / "axis1_clinical_association_combined_forest_by_domain.png",
            component=1,
        )
        plot_axis_association_dot_heatmap(
            assoc_all,
            fig_dir / "axis1_clinical_association_dot_heatmap_by_domain.png",
            component=1,
        )

    gradient_patient = read_csv_optional(table_dir / "clinical_gradient_patient_values.csv")
    gradient_summary = read_csv_optional(table_dir / "clinical_gradient_by_axis.csv")
    if len(gradient_patient) and len(gradient_summary):
        for panel in CLINICAL_PANELS:
            plot_gradient_continuous_box_points(
                gradient_patient,
                gradient_summary,
                panel,
                fig_dir / f"{clean_filename(panel)}_axis1_continuous_gradient_box_point.png",
                component=1,
            )
            plot_gradient_ordinal_heatmap(
                gradient_summary,
                panel,
                fig_dir / f"{clean_filename(panel)}_axis1_ordinal_gradient_heatmap.png",
                component=1,
            )
            plot_multivariable_profile_heatmap(
                gradient_summary,
                panel,
                fig_dir / f"{clean_filename(panel)}_axis1_multivariable_clinical_profile_heatmap.png",
                component=1,
            )
            plot_multivariable_profile_score_by_axis_quantile(
                gradient_patient,
                panel,
                fig_dir / f"{clean_filename(panel)}_axis1_multivariable_profile_score_box_point.png",
                component=1,
            )
        plot_multivariable_profile_score_facets(
            gradient_patient,
            fig_dir / "axis1_multivariable_profile_score_facets_by_domain.png",
            component=1,
        )

    endpoint_summary = read_csv_optional(table_dir / "endpoint_validation_summary.csv")
    endpoint_pred = read_csv_optional(table_dir / "endpoint_validation_predictions.csv")
    if len(endpoint_summary):
        plot_endpoint_summary(endpoint_summary, fig_dir / "endpoint_validation_auroc_forest.png")
    if len(endpoint_summary) and len(endpoint_pred):
        plot_endpoint_roc_curves(endpoint_pred, endpoint_summary, fig_dir, args)

    conf_summary = read_csv_optional(table_dir / "confounder_adjusted_alignment_summary.csv")
    if len(conf_summary) and len(alignment_summary):
        plot_confounder(conf_summary, alignment_summary, fig_dir / "confounder_adjusted_alignment.png")

    lopo = read_csv_optional(table_dir / "leave_one_position_out_alignment_summary.csv")
    if len(lopo) and len(alignment_summary):
        plot_lopo(lopo, alignment_summary, fig_dir / "leave_one_position_out_alignment.png")

    neg_ctrl = read_csv_optional(table_dir / "negative_controls.csv")
    neg_summary = read_csv_optional(table_dir / "negative_control_summary.csv")
    if len(neg_ctrl) and len(neg_summary):
        plot_negative_controls(neg_ctrl, neg_summary, fig_dir / "negative_controls_all_clinical.png")

    lovo_summary = read_csv_optional(table_dir / "leave_one_variable_out_cca_summary.csv")
    if len(lovo_summary):
        plot_leave_one_variable_out_forest(
            lovo_summary,
            fig_dir / "leave_one_variable_out_axis1_heldout_variable_forest.png",
            component=1,
        )

    lovo_gradient_summary = read_csv_optional(table_dir / "leave_one_variable_out_gradient_summary.csv")
    if len(lovo_gradient_summary):
        plot_leave_one_variable_out_gradient_heatmap(
            lovo_gradient_summary,
            fig_dir / "leave_one_variable_out_axis1_gradient_heatmap.png",
            component=1,
        )

    cross_summary = read_csv_optional(table_dir / "cross_domain_characterization_summary.csv")
    if len(cross_summary):
        plot_cross_domain_forest(
            cross_summary,
            fig_dir / "cross_domain_axis1_clinical_characterization_forest.png",
            component=1,
        )
        plot_cross_domain_dot_heatmap(
            cross_summary,
            fig_dir / "cross_domain_axis1_clinical_characterization_dot_heatmap.png",
            component=1,
        )

    repeated_cca = read_csv_optional(table_dir / "repeated_cv_cca_alignment_values.csv")
    if len(repeated_cca):
        plot_repeated_cca_alignment(
            repeated_cca,
            fig_dir / "repeated_cv_cca_axis1_alignment_by_split.png",
            component=1,
        )

    repeated_endpoint = read_csv_optional(table_dir / "repeated_cv_endpoint_values.csv")
    if len(repeated_endpoint):
        plot_repeated_endpoint_auroc(
            repeated_endpoint,
            fig_dir / "repeated_cv_endpoint_auroc_by_split.png",
        )


def parse_args():
    p = argparse.ArgumentParser(description="Regenerate figures from clinically anchored acoustic phenotyping result tables")
    p.add_argument(
        "--result-dir",
        type=str,
        default="Clinical_alignment/outputs/clinically_anchored_acoustic_phenotyping_clean_v4",
        help="Output root containing tables/ and figures/. Ignored for table/fig dirs if those are explicitly given.",
    )
    p.add_argument("--table-dir", type=str, default=None, help="Directory containing CSV result tables")
    p.add_argument("--fig-dir", type=str, default=None, help="Directory where figures will be saved")
    p.add_argument("--seed", type=int, default=42, help="Seed used only for bootstrap ROC confidence bands")
    p.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap repeats for ROC confidence bands")
    return p.parse_args()


def main():
    args = parse_args()
    setup_plotting()
    result_dir = Path(args.result_dir)
    table_dir = Path(args.table_dir) if args.table_dir else result_dir / "tables"
    fig_dir = Path(args.fig_dir) if args.fig_dir else result_dir / "figures"
    log(f"Plotting from table_dir={table_dir}")
    log(f"Saving figures to fig_dir={fig_dir}")
    plot_all_results(table_dir, fig_dir, args)
    log("Done plotting.")


if __name__ == "__main__":
    main()
