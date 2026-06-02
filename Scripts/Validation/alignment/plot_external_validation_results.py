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


AXIS_X = "cca_acoustic_axis1"
AXIS_Y = "cca_clinical_axis1"

CLINICAL_PROFILE_VARS = ["EF_Teich", "NTproBNP", "NYHA", "LVEDD_mm"]
ENDPOINT_ORDER = ["EF_lt_40", "NYHA_ge_3", "NTproBNP_ge_900", "LVEDD_dilated"]

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot external validation results from internally fitted BEATs CCA outputs.")
    parser.add_argument("--alignment-dir", type=Path, default=Path(r"D:\PycharmProjects\HSCA\Outputs\validation\alignment"))
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup_plotting()

    tables_dir = args.tables_dir if args.tables_dir is not None else args.alignment_dir / "tables"
    out_dir = args.out_dir if args.out_dir is not None else args.alignment_dir / "figures"
    ensure_dir(out_dir)

    axis_df = read_csv_optional(tables_dir / "external_axis_scores.csv")
    summary_df = read_csv_optional(tables_dir / "external_main_alignment_summary.csv")
    endpoint_df = read_csv_optional(tables_dir / "external_endpoint_auroc_summary.csv")
    null_df = read_csv_optional(tables_dir / "external_permutation_null.csv")

    plot_axis_scatter(axis_df, summary_df, null_df, out_dir / "external_axis_scatter.png")
    plot_quartile_gradient(axis_df, out_dir / "external_axis_quartile_gradient.png")
    plot_endpoint_auroc_forest(endpoint_df, out_dir / "external_endpoint_auroc_forest.png")
    plot_permutation_null(null_df, out_dir / "external_permutation_null.png")

    log("Done.")
    return {
        "figures_dir": out_dir,
        "axis_scatter": out_dir / "external_axis_scatter.png",
        "quartile_gradient": out_dir / "external_axis_quartile_gradient.png",
        "endpoint_auroc_forest": out_dir / "external_endpoint_auroc_forest.png",
        "permutation_null": out_dir / "external_permutation_null.png",
    }


if __name__ == "__main__":
    main()
