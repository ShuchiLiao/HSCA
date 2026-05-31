#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Focused plotting script for all-clinical acoustic-clinical CCA outputs.

This version is intentionally simplified for the revised manuscript story:
1) all-clinical acoustic CCA axis 1 vs all-clinical clinical CCA axis 1;
2) all-clinical acoustic axis 1 interpretation by clinical variables;
3) selected clinical profiles across acoustic-axis quartiles;
4) endpoint ROC curves with CI bands using axis 1 only.

It reads CSV tables produced by run_clinically_anchored_acoustic_phenotyping_clean_v5.py
(or v4 where compatible) and only regenerates figures.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

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


PATIENT_ID_COL = "patient_id"

FUNCTION_VARS = ["EF_Teich", "NTproBNP", "NYHA"]
STRUCTURE_VARS = ["LA_mm", "LVEDD_mm", "IVS_mm", "LVPW_mm"]
VALVE_VARS = ["MR_grade", "TR_grade", "AR_grade", "PR_grade", "AS_grade", "MS_grade"]
ALL_CLINICAL_VARS = FUNCTION_VARS + STRUCTURE_VARS + VALVE_VARS

ORDINAL_VARS = {"NYHA", "MR_grade", "TR_grade", "AR_grade", "PR_grade", "AS_grade", "MS_grade"}

# The user wrote "LP"; this script uses the fixed clinical column LA_mm.
SELECTED_CONTINUOUS_VARS = ["EF_Teich", "NTproBNP", "LA_mm", "LVEDD_mm"]

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

VARIABLE_GROUP: Dict[str, str] = {}
for _v in FUNCTION_VARS:
    VARIABLE_GROUP[_v] = "Function"
for _v in STRUCTURE_VARS:
    VARIABLE_GROUP[_v] = "Structure"
for _v in VALVE_VARS:
    VARIABLE_GROUP[_v] = "Valve"

ENDPOINT_FIGURES = [
    ("EF_lt_40", "EF < 40", "figure_06_roc_EF_lt_40_axis1.png"),
    ("NTproBNP_ge_900", "NT-proBNP ≥ 900", "figure_07_roc_NTproBNP_ge_900_axis1.png"),
    ("LVEDD_dilated", "LVEDD dilated", "figure_08_roc_LVEDD_dilated_axis1.png"),
    ("NYHA_ge_3", "NYHA ≥ 3", "figure_09_roc_NYHA_ge_3_axis1.png"),
]

SELECTED_CONTINUOUS_COLORS = {
    "EF_Teich": "#009E73",    # green
    "NTproBNP": "#CC79A7",    # magenta
    "LA_mm": "#D55E00",       # orange
    "LVEDD_mm": "#0072B2",    # blue
}


def now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def clean_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", str(s))[:180]


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


def safe_spearman(x, y) -> Tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 5 or np.nanstd(x[mask]) < 1e-12 or np.nanstd(y[mask]) < 1e-12:
        return np.nan, np.nan, n
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p), n


def p_text(p: float) -> str:
    if not np.isfinite(p):
        return "p=NA"
    if p < 1e-4:
        return "p<1e-4"
    if p < 0.001:
        return f"p={p:.1e}"
    return f"p={p:.3f}"


def domain_palette() -> Dict[str, tuple]:
    # Colorblind-friendly, non-gray palette.
    return {
        "Function": "#009E73",   # green
        "Structure": "#D55E00",  # vermillion/orange
        "Valve": "#0072B2",      # blue
    }


def unit_label(var: str, values: np.ndarray | pd.Series | None = None) -> str:
    if var == "EF_Teich":
        return "EF (%)"
    if var == "NTproBNP":
        # The prepared table often stores log1p(NT-proBNP). Detect by range.
        if values is not None:
            arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
            mx = np.nanmax(arr) if np.isfinite(arr).any() else np.nan
            if np.isfinite(mx) and mx <= 25:
                return "log1p(NT-proBNP [pg/mL])"
        return "NT-proBNP (pg/mL)"
    if var in {"LA_mm", "LVEDD_mm", "IVS_mm", "LVPW_mm"}:
        return f"{PRETTY_LABELS.get(var, var)} (mm)"
    return PRETTY_LABELS.get(var, var)


def get_all_clinical_axis_scores(axis_df: pd.DataFrame) -> pd.DataFrame:
    required = {"panel", "adjustment", "cca_acoustic_axis1", "cca_clinical_axis1"}
    if len(axis_df) == 0 or not required.issubset(axis_df.columns):
        log(f"Axis score table missing columns: {required - set(axis_df.columns)}")
        return pd.DataFrame()
    d = axis_df[
        axis_df["panel"].astype(str).eq("all_clinical")
        & axis_df["adjustment"].astype(str).eq("none")
    ].copy()
    d = d[np.isfinite(d["cca_acoustic_axis1"]) & np.isfinite(d["cca_clinical_axis1"])]
    return d


def lookup_alignment_stats(summary_df: pd.DataFrame, component: int = 1) -> str:
    if len(summary_df) == 0:
        return ""
    required = {"panel", "component", "spearman_acoustic_vs_clinical_axis"}
    if not required.issubset(summary_df.columns):
        return ""
    sub = summary_df[
        summary_df["panel"].astype(str).eq("all_clinical")
        & summary_df["component"].astype(int).eq(int(component))
    ].copy()
    if "adjustment" in sub.columns:
        sub = sub[sub["adjustment"].astype(str).eq("none")]
    if len(sub) == 0:
        return ""
    row = sub.iloc[0]
    rho = row.get("spearman_acoustic_vs_clinical_axis", np.nan)
    lo = row.get("spearman_ci95_low", np.nan)
    hi = row.get("spearman_ci95_high", np.nan)
    pp = row.get("permutation_p_abs", np.nan)
    txt = []
    if np.isfinite(rho):
        if np.isfinite(lo) and np.isfinite(hi):
            txt.append(f"ρ={rho:.2f} [{lo:.2f}, {hi:.2f}]")
        else:
            txt.append(f"ρ={rho:.2f}")
    if np.isfinite(pp):
        txt.append(f"perm. {p_text(pp)}")
    return ", ".join(txt)


def roc_curve_with_ci(y_true, y_score, n_boot: int, seed: int, grid: np.ndarray | None = None):
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


# 图一：All-clinical acoustic CCA axis 1 vs clinical CCA axis 1 散点+趋势线。
def plot_figure01_axis1_scatter_line(axis_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    """Direct axis-to-axis alignment. The point color is blue, not gray."""
    d = get_all_clinical_axis_scores(axis_df)
    if len(d) == 0:
        return
    x = d["cca_acoustic_axis1"].to_numpy(float)
    y = d["cca_clinical_axis1"].to_numpy(float)
    rho, pp, n = safe_spearman(x, y)

    plt.figure(figsize=(7.2, 6.4))
    ax = plt.gca()
    scatter_color = "#0072B2"
    trend_color = "#D55E00"
    bin_color = "#009E73"
    ax.scatter(x, y, s=24, alpha=0.34, color=scatter_color, edgecolor="none", label="Patients")

    if np.nanstd(x) > 1e-12 and np.nanstd(y) > 1e-12:
        xs = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), 120)
        slope, intercept = np.polyfit(x, y, 1)
        ax.plot(xs, slope * xs + intercept, color=trend_color, lw=2.2, label="Linear trend")
        try:
            q = pd.qcut(pd.Series(x).rank(method="first"), 10, labels=False)
            tmp = pd.DataFrame({"x": x, "y": y, "bin": q})
            b = tmp.groupby("bin").agg(x_mean=("x", "mean"), y_mean=("y", "mean")).dropna()
            ax.plot(b["x_mean"], b["y_mean"], marker="o", lw=2.0, color=bin_color, label="Binned mean")
        except Exception:
            pass

    ax.axhline(0, color="0.82", lw=0.9)
    ax.axvline(0, color="0.82", lw=0.9)
    ax.set_xlabel("All-clinical acoustic CCA axis 1 score")
    ax.set_ylabel("All-clinical clinical CCA axis 1 score")
    stats_txt = lookup_alignment_stats(summary_df, component=1)
    title = "All-clinical acoustic axis aligned with clinical axis"
    if stats_txt:
        title += f"\n{stats_txt}"
    elif np.isfinite(rho):
        title += f"\nSpearman ρ={rho:.2f}, {p_text(pp)}, n={n}"
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    savefig(out_path)


# 图二：All-clinical acoustic CCA axis 1 与各临床变量的 lollipop 相关图（带95%CI）。
def plot_figure02_axis1_variable_lollipop(assoc_df: pd.DataFrame, out_path: Path) -> None:
    """Axis 1 clinical-variable correlation profile; colored by Function/Structure/Valve."""
    if len(assoc_df) == 0:
        return
    required = {"panel", "adjustment", "component", "variable", "spearman_axis_variable"}
    if not required.issubset(assoc_df.columns):
        log(f"Skip figure02: missing {required - set(assoc_df.columns)}")
        return
    d = assoc_df[
        assoc_df["panel"].astype(str).eq("all_clinical")
        & assoc_df["adjustment"].astype(str).eq("none")
        & assoc_df["component"].astype(int).eq(1)
    ].copy()
    d = d[np.isfinite(d["spearman_axis_variable"])]
    if len(d) == 0:
        return
    order = [v for v in ALL_CLINICAL_VARS if v in set(d["variable"])]
    d["order"] = d["variable"].map({v: i for i, v in enumerate(order)})
    d = d.sort_values("order", ascending=False)
    pal = domain_palette()
    colors = [pal.get(VARIABLE_GROUP.get(v, "Valve"), "#0072B2") for v in d["variable"]]

    plt.figure(figsize=(9.8, max(5.8, 0.44 * len(d))))
    ax = plt.gca()
    y = np.arange(len(d))
    x = d["spearman_axis_variable"].to_numpy(float)
    lo = d["spearman_ci95_low"].to_numpy(float) if "spearman_ci95_low" in d.columns else np.full(len(d), np.nan)
    hi = d["spearman_ci95_high"].to_numpy(float) if "spearman_ci95_high" in d.columns else np.full(len(d), np.nan)
    for yi, xi, li, hi_i, color in zip(y, x, lo, hi, colors):
        ax.hlines(yi, 0, xi, color=color, lw=3.0, alpha=0.95)
        if np.isfinite(li) and np.isfinite(hi_i):
            ax.plot([li, hi_i], [yi, yi], color="black", lw=1.05, alpha=0.90)
            ax.plot([li, li], [yi - 0.06, yi + 0.06], color="black", lw=1.05)
            ax.plot([hi_i, hi_i], [yi - 0.06, yi + 0.06], color="black", lw=1.05)
        ax.scatter([xi], [yi], s=90, color=color, edgecolor="black", linewidth=0.6, zorder=3)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([PRETTY_LABELS.get(v, v) for v in d["variable"]])
    for tick, var in zip(ax.get_yticklabels(), d["variable"]):
        tick.set_color(pal.get(VARIABLE_GROUP.get(var, "Valve"), "black"))
    ax.set_xlabel("Spearman correlation with all-clinical acoustic CCA axis 1")
    ax.set_title("Clinical interpretation of all-clinical acoustic CCA axis 1")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=pal[g], markeredgecolor="black", label=g, markersize=9)
        for g in ["Function", "Structure", "Valve"]
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=10)
    savefig(out_path)


def _selected_continuous_data(patient_long: pd.DataFrame, component: int = 1) -> pd.DataFrame:
    required = {"panel", "adjustment", "component", "axis_group", "variable", "value"}
    if len(patient_long) == 0 or not required.issubset(patient_long.columns):
        log(f"Continuous profile table missing columns: {required - set(patient_long.columns)}")
        return pd.DataFrame()
    d = patient_long[
        patient_long["panel"].astype(str).eq("all_clinical")
        & patient_long["adjustment"].astype(str).eq("none")
        & patient_long["component"].astype(int).eq(int(component))
        & patient_long["variable"].isin(SELECTED_CONTINUOUS_VARS)
    ].copy()
    d = d[np.isfinite(d["value"])]
    return d


# 图三：Selected continuous clinical profile，点云+箱线图，仅 EF、NT-proBNP、LA、LVEDD。
def plot_figure03_selected_continuous_box_points(patient_long: pd.DataFrame, out_path: Path, component: int = 1) -> None:
    d = _selected_continuous_data(patient_long, component)
    if len(d) == 0:
        return
    vars_order = [v for v in SELECTED_CONTINUOUS_VARS if v in set(d["variable"])]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), squeeze=False)
    pal = domain_palette()
    levels = ["Q1", "Q2", "Q3", "Q4"]
    for ax, var in zip(axes.ravel(), vars_order):
        sub = d[d["variable"].astype(str).eq(var)].copy()
        sub["axis_group"] = pd.Categorical(sub["axis_group"].astype(str), categories=levels, ordered=True)
        color = SELECTED_CONTINUOUS_COLORS.get(var, pal.get(VARIABLE_GROUP.get(var), "#0072B2"))
        if HAS_SEABORN:
            sns.boxplot(
                data=sub, x="axis_group", y="value", color=color, showfliers=False,
                linewidth=1.25, boxprops={"alpha": 0.36},
                medianprops={"color": "black", "linewidth": 1.35}, ax=ax,
            )
            sns.stripplot(data=sub, x="axis_group", y="value", color=color, size=2.3, alpha=0.24, jitter=0.24, ax=ax)
        else:
            rng = np.random.default_rng(123)
            groups = [sub.loc[sub["axis_group"].astype(str) == g, "value"].dropna().to_numpy(float) for g in levels]
            ax.boxplot(groups, labels=levels, showfliers=False)
            for i, vals in enumerate(groups, start=1):
                ax.scatter(i + rng.uniform(-0.22, 0.22, len(vals)), vals, s=8, alpha=0.28, color=color)
        rho, pp, _ = safe_spearman(sub.get("axis_score", np.arange(len(sub))), sub["value"])
        ax.set_title(f"{PRETTY_LABELS.get(var, var)}" + (f"\nρ={rho:.2f}, {p_text(pp)}" if np.isfinite(rho) else ""))
        ax.set_xlabel("All-clinical acoustic axis 1 quartile")
        ax.set_ylabel(unit_label(var, sub["value"]))
    for ax in axes.ravel()[len(vars_order):]:
        ax.axis("off")
    fig.suptitle("Selected continuous clinical profiles across acoustic-axis quartiles", y=1.02, fontsize=16)
    savefig(out_path)


# 图四：Selected continuous clinical profile，仅箱线图，不显示病人点云。
def plot_figure04_selected_continuous_box_only(patient_long: pd.DataFrame, out_path: Path, component: int = 1) -> None:
    d = _selected_continuous_data(patient_long, component)
    if len(d) == 0:
        return
    vars_order = [v for v in SELECTED_CONTINUOUS_VARS if v in set(d["variable"])]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), squeeze=False)
    pal = domain_palette()
    levels = ["Q1", "Q2", "Q3", "Q4"]
    for ax, var in zip(axes.ravel(), vars_order):
        sub = d[d["variable"].astype(str).eq(var)].copy()
        sub["axis_group"] = pd.Categorical(sub["axis_group"].astype(str), categories=levels, ordered=True)
        color = SELECTED_CONTINUOUS_COLORS.get(var, pal.get(VARIABLE_GROUP.get(var), "#0072B2"))
        if HAS_SEABORN:
            sns.boxplot(
                data=sub, x="axis_group", y="value", color=color, showfliers=False,
                linewidth=1.35, boxprops={"alpha": 0.50},
                medianprops={"color": "black", "linewidth": 1.45}, ax=ax,
            )
        else:
            groups = [sub.loc[sub["axis_group"].astype(str) == g, "value"].dropna().to_numpy(float) for g in levels]
            ax.boxplot(groups, labels=levels, showfliers=False)
        rho, pp, _ = safe_spearman(sub.get("axis_score", np.arange(len(sub))), sub["value"])
        ax.set_title(f"{PRETTY_LABELS.get(var, var)}" + (f"\nρ={rho:.2f}, {p_text(pp)}" if np.isfinite(rho) else ""))
        ax.set_xlabel("All-clinical acoustic axis 1 quartile")
        ax.set_ylabel(unit_label(var, sub["value"]))
    for ax in axes.ravel()[len(vars_order):]:
        ax.axis("off")
    fig.suptitle("Selected continuous clinical profiles across acoustic-axis quartiles", y=1.02, fontsize=16)
    savefig(out_path)


# 图五：Ordinal profile heatmap，仅 NYHA≥3、MR/TR/AR grade ≥2。
def plot_figure05_selected_ordinal_heatmap(summary_df: pd.DataFrame, patient_long: pd.DataFrame, out_path: Path, component: int = 1) -> None:
    targets = [
        ("NYHA", "NYHA ≥3", 3, "Function"),
        ("MR_grade", "MR grade ≥2", 2, "Valve"),
        ("TR_grade", "TR grade ≥2", 2, "Valve"),
        ("AR_grade", "AR grade ≥2", 2, "Valve"),
    ]
    rows = []

    # Preferred source: summary table if it already contains proportion columns.
    if len(summary_df):
        required = {"panel", "adjustment", "component", "axis_group", "variable"}
        if required.issubset(summary_df.columns):
            d = summary_df[
                summary_df["panel"].astype(str).eq("all_clinical")
                & summary_df["adjustment"].astype(str).eq("none")
                & summary_df["component"].astype(int).eq(int(component))
            ].copy()
            for var, label, thr, _group in targets:
                sub = d[d["variable"].astype(str).eq(var)].copy()
                if len(sub) == 0:
                    continue
                col = f"prop_ge_{thr}"
                if col not in sub.columns:
                    continue
                rec = {"row_label": label, "group": _group}
                for q in ["Q1", "Q2", "Q3", "Q4"]:
                    hit = sub[sub["axis_group"].astype(str).eq(q)]
                    rec[q] = float(hit[col].iloc[0]) if len(hit) else np.nan
                rows.append(rec)

    # Fallback: compute proportions directly from patient-level long table.
    if not rows and len(patient_long):
        required = {"panel", "adjustment", "component", "axis_group", "variable", "value"}
        if required.issubset(patient_long.columns):
            d = patient_long[
                patient_long["panel"].astype(str).eq("all_clinical")
                & patient_long["adjustment"].astype(str).eq("none")
                & patient_long["component"].astype(int).eq(int(component))
            ].copy()
            for var, label, thr, _group in targets:
                sub = d[d["variable"].astype(str).eq(var)].copy()
                if len(sub) == 0:
                    continue
                rec = {"row_label": label, "group": _group}
                for q in ["Q1", "Q2", "Q3", "Q4"]:
                    qsub = sub[sub["axis_group"].astype(str).eq(q)].copy()
                    vals = pd.to_numeric(qsub["value"], errors="coerce")
                    rec[q] = float((vals >= thr).mean()) if len(vals.dropna()) else np.nan
                rows.append(rec)

    if not rows:
        return

    mat_df = pd.DataFrame(rows)
    mat = mat_df[["Q1", "Q2", "Q3", "Q4"]].astype(float)
    vmax = max(0.05, min(1.0, np.nanmax(mat.values) if np.isfinite(mat.values).any() else 1.0))
    plt.figure(figsize=(6.8, 4.2))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.heatmap(mat, annot=True, fmt=".2f", linewidths=0.5, vmin=0, vmax=vmax,
                    cmap="Blues", cbar_kws={"label": "Proportion"}, ax=ax)
        ax.set_yticklabels(mat_df["row_label"], rotation=0)
    else:
        im = ax.imshow(mat.values, aspect="auto", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label="Proportion")
        ax.set_xticks(range(4)); ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        ax.set_yticks(range(len(mat_df))); ax.set_yticklabels(mat_df["row_label"])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")
    pal = domain_palette()
    for tick, group in zip(ax.get_yticklabels(), mat_df["group"]):
        tick.set_color(pal.get(group, "black"))
    ax.set_xlabel("All-clinical acoustic axis 1 quartile")
    ax.set_ylabel("")
    ax.set_title("Ordinal clinical burden across all-clinical acoustic-axis quartiles")
    savefig(out_path)


# 图六至图九：Endpoint ROC 曲线，带 bootstrap CI，仅 Axis 1。
def plot_endpoint_roc(pred_df: pd.DataFrame, summary_df: pd.DataFrame, endpoint: str, pretty: str, out_path: Path, args) -> None:
    if len(pred_df) == 0:
        return
    d = pred_df[pred_df["endpoint"].astype(str).eq(endpoint)].copy()
    if len(d) == 0:
        log(f"Skip ROC for {endpoint}: no predictions")
        return
    if "n_axis_features" not in d.columns:
        d["n_axis_features"] = 1
    d = d[d["n_axis_features"].astype(int).eq(1)].copy()
    if len(d) == 0 or d["y_true"].nunique() < 2:
        log(f"Skip ROC for {endpoint}: no valid Axis 1 predictions")
        return

    summary = summary_df.copy() if len(summary_df) else pd.DataFrame()
    if len(summary) and "n_axis_features" not in summary.columns:
        summary["n_axis_features"] = 1

    y = d["y_true"].astype(int).to_numpy()
    p = d["y_prob"].astype(float).to_numpy()
    fpr, tpr, lo, hi = roc_curve_with_ci(y, p, args.n_bootstrap, args.seed + 9001)
    auc = roc_auc_score(y, p)

    row = summary[(summary["endpoint"].astype(str) == endpoint) & (summary["n_axis_features"].astype(int) == 1)] if len(summary) else pd.DataFrame()
    if len(row):
        auc_lo = row["auroc_ci95_low"].iloc[0]
        auc_hi = row["auroc_ci95_high"].iloc[0]
    else:
        auc_lo, auc_hi = np.nan, np.nan

    plt.figure(figsize=(6.6, 6.1))
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", lw=1, label="Chance")
    color = "#0072B2"
    if np.isfinite(lo).any():
        ax.fill_between(fpr, lo, hi, color=color, alpha=0.16)
    lab = f"Axis 1: AUROC={auc:.3f}"
    if np.isfinite(auc_lo) and np.isfinite(auc_hi):
        lab += f" ({auc_lo:.3f}–{auc_hi:.3f})"
    ax.plot(fpr, tpr, lw=2.6, color=color, label=lab)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve: {pretty}\nAxis 1")
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    savefig(out_path)



def _ci_cols(df: pd.DataFrame):
    if {"spearman_ci95_low", "spearman_ci95_high"}.issubset(df.columns):
        return "spearman_ci95_low", "spearman_ci95_high"
    return None, None


def _alignment_value_col(df: pd.DataFrame) -> str:
    if "spearman_acoustic_vs_clinical_axis" in df.columns:
        return "spearman_acoustic_vs_clinical_axis"
    if "spearman" in df.columns:
        return "spearman"
    return ""


def _endpoint_pretty(endpoint: str) -> str:
    mapping = {
        "EF_lt_40": "EF <40",
        "EF_lt_50": "EF <50",
        "NTproBNP_ge_125": "NT-proBNP ≥125",
        "NTproBNP_ge_300": "NT-proBNP ≥300",
        "NTproBNP_ge_900": "NT-proBNP ≥900",
        "NYHA_ge_1": "NYHA ≥1",
        "NYHA_ge_2": "NYHA ≥2",
        "NYHA_ge_3": "NYHA ≥3",
        "LA_ge_40": "LA ≥40",
        "LVEDD_dilated": "LVEDD dilated",
    }
    return mapping.get(str(endpoint), str(endpoint).replace("_", " "))


# 图十：site robustness / site contribution 合并 forest plot + CI。
def plot_figure10_site_robustness_contribution(
    lopo_df: pd.DataFrame,
    contribution_df: pd.DataFrame,
    alignment_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    """Merge leave-one-position-out and only-position contribution into one forest plot.

    Full all-clinical result is shown as a vertical dashed reference line.
    """
    rows = []

    full_val = np.nan
    full_lo = np.nan
    full_hi = np.nan
    if len(alignment_summary):
        base = alignment_summary[
            alignment_summary.get("panel", pd.Series(dtype=str)).astype(str).eq("all_clinical")
            & alignment_summary.get("adjustment", pd.Series(dtype=str)).astype(str).eq("none")
            & alignment_summary.get("component", pd.Series(dtype=float)).astype(float).eq(1)
        ].copy()
        if len(base):
            r = base.iloc[0]
            full_val = r.get("spearman_acoustic_vs_clinical_axis", np.nan)
            full_lo = r.get("spearman_ci95_low", np.nan)
            full_hi = r.get("spearman_ci95_high", np.nan)
            rows.append({
                "group": "Full reference",
                "label": "Full all-clinical",
                "value": full_val,
                "lo": full_lo,
                "hi": full_hi,
                "kind": "Full",
                "order": 0,
            })

    if len(contribution_df):
        d = contribution_df[contribution_df.get("component", pd.Series(dtype=float)).astype(float).eq(1)].copy()
        col = _alignment_value_col(d)
        lo_col, hi_col = _ci_cols(d)
        if col:
            site_order = {"full_position_concat": 1, "only_A": 2, "only_E": 3, "only_M": 4, "only_P": 5, "only_T": 6}
            labels = {
                "full_position_concat": "Position-concat full",
                "only_A": "Only A",
                "only_E": "Only E",
                "only_M": "Only M",
                "only_P": "Only P",
                "only_T": "Only T",
            }
            for _, r in d.iterrows():
                name = str(r.get("position_analysis", ""))
                if name not in site_order:
                    continue
                rows.append({
                    "group": "Site contribution",
                    "label": labels.get(name, name),
                    "value": r.get(col, np.nan),
                    "lo": r.get(lo_col, np.nan) if lo_col else np.nan,
                    "hi": r.get(hi_col, np.nan) if hi_col else np.nan,
                    "kind": "Contribution",
                    "order": site_order[name] + 10,
                })

    if len(lopo_df):
        d = lopo_df[lopo_df.get("component", pd.Series(dtype=float)).astype(float).eq(1)].copy()
        col = _alignment_value_col(d)
        lo_col, hi_col = _ci_cols(d)
        if col and "left_out_position" in d.columns:
            site_order = {"A": 1, "E": 2, "M": 3, "P": 4, "T": 5}
            for _, r in d.iterrows():
                pos = str(r.get("left_out_position", "")).upper()
                rows.append({
                    "group": "Leave-one-site robustness",
                    "label": f"Leave {pos}",
                    "value": r.get(col, np.nan),
                    "lo": r.get(lo_col, np.nan) if lo_col else np.nan,
                    "hi": r.get(hi_col, np.nan) if hi_col else np.nan,
                    "kind": "LOPO",
                    "order": site_order.get(pos, 99) + 30,
                })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return
    df = df[np.isfinite(df["value"])].sort_values("order", ascending=True).reset_index(drop=True)
    if len(df) == 0:
        return

    colors = {
        "Full": "#000000",
        "Contribution": "#0072B2",
        "LOPO": "#D55E00",
    }
    plt.figure(figsize=(9.4, max(5.4, 0.42 * len(df))))
    ax = plt.gca()
    y = np.arange(len(df))
    for yi, (_, r) in zip(y, df.iterrows()):
        color = colors.get(r["kind"], "black")
        lo = r["lo"]
        hi = r["hi"]
        val = r["value"]
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [yi, yi], color=color, lw=2.2, alpha=0.95)
            ax.plot([lo, lo], [yi - 0.07, yi + 0.07], color=color, lw=1.4)
            ax.plot([hi, hi], [yi - 0.07, yi + 0.07], color=color, lw=1.4)
        ax.scatter([val], [yi], color=color, edgecolor="black", linewidth=0.6, s=78, zorder=3)

    if np.isfinite(full_val):
        ax.axvline(full_val, color="black", linestyle="--", lw=1.25, alpha=0.85, label="Full all-clinical reference")
    ax.axvline(0, color="black", lw=1.0, alpha=0.75)
    for split_at in [1, 7]:
        if split_at < len(df):
            ax.axhline(split_at - 0.5, color="0.84", lw=1)

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.invert_yaxis()
    ax.set_xlabel("OOF Spearman correlation\n(all-clinical acoustic axis 1 vs clinical axis 1)")
    ax.set_title("Site contribution and leave-one-site robustness")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["Contribution"], markeredgecolor="black", label="Only-site contribution", markersize=9),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["LOPO"], markeredgecolor="black", label="Leave-one-site robustness", markersize=9),
        plt.Line2D([0], [0], color="black", linestyle="--", label="Full reference"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=9.5)
    savefig(out_path)


# 图十一：CCA axis 1 得分排序图 / ranked score plot / empirical quantile plot。
def plot_figure11_ranked_axis_score(axis_rank_df: pd.DataFrame, axis_scores: pd.DataFrame, out_path: Path) -> None:
    """Rank all patients by all-clinical acoustic CCA axis 1 score."""
    if len(axis_rank_df):
        d = axis_rank_df.copy()
    else:
        d = get_all_clinical_axis_scores(axis_scores).copy()
        if len(d) == 0:
            return
        d = d[np.isfinite(d["cca_acoustic_axis1"])].sort_values("cca_acoustic_axis1").reset_index(drop=True)
        n = len(d)
        d["axis1_rank_low_to_high"] = np.arange(1, n + 1)
        d["axis1_empirical_quantile"] = (d["axis1_rank_low_to_high"] - 0.5) / n
    if len(d) == 0 or "cca_acoustic_axis1" not in d.columns:
        return
    if "axis1_empirical_quantile" not in d.columns:
        d = d.sort_values("cca_acoustic_axis1").reset_index(drop=True)
        n = len(d)
        d["axis1_empirical_quantile"] = (np.arange(1, n + 1) - 0.5) / n
    d = d[np.isfinite(d["axis1_empirical_quantile"]) & np.isfinite(d["cca_acoustic_axis1"])].copy()
    if len(d) == 0:
        return

    color_col = "cca_clinical_axis1" if "cca_clinical_axis1" in d.columns and np.isfinite(d["cca_clinical_axis1"]).any() else None

    plt.figure(figsize=(9.4, 5.4))
    ax = plt.gca()
    ax.plot(d["axis1_empirical_quantile"], d["cca_acoustic_axis1"], color="#1F1F1F", lw=1.6, alpha=0.85)
    if color_col:
        vals = d[color_col].to_numpy(float)
        vmax = np.nanpercentile(np.abs(vals), 98) if np.isfinite(vals).any() else 1
        vmax = max(vmax, 1e-6)
        sc = ax.scatter(
            d["axis1_empirical_quantile"],
            d["cca_acoustic_axis1"],
            c=vals,
            cmap="viridis",
            s=18,
            alpha=0.82,
            edgecolor="none",
            vmin=np.nanpercentile(vals, 2),
            vmax=np.nanpercentile(vals, 98),
        )
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("Clinical CCA axis 1 score")
    else:
        ax.scatter(d["axis1_empirical_quantile"], d["cca_acoustic_axis1"], color="#0072B2", s=18, alpha=0.75, edgecolor="none")
    ax.axhline(0, color="black", lw=1, alpha=0.72)
    for q in [0.25, 0.50, 0.75]:
        ax.axvline(q, color="0.72", linestyle="--", lw=1)
    ax.set_xlabel("Empirical quantile of all-clinical acoustic CCA axis 1")
    ax.set_ylabel("All-clinical acoustic CCA axis 1 score")
    ax.set_title("Ranked patient scores on all-clinical acoustic CCA axis 1")
    savefig(out_path)


# 图S01：repeated random split，boxplot + jitter points，只画 all-clinical axis 1。
def plot_figureS01_repeated_random_split(repeated_df: pd.DataFrame, out_path: Path) -> None:
    if len(repeated_df) == 0:
        return
    required = {"panel", "component", "spearman_acoustic_vs_clinical_axis"}
    if not required.issubset(repeated_df.columns):
        return
    d = repeated_df[
        repeated_df["panel"].astype(str).eq("all_clinical")
        & repeated_df["component"].astype(float).eq(1)
    ].copy()
    if "status" in d.columns:
        d = d[d["status"].astype(str).eq("ok")]
    d = d[np.isfinite(d["spearman_acoustic_vs_clinical_axis"])]
    if len(d) == 0:
        return

    plt.figure(figsize=(5.6, 5.5))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.boxplot(data=d, y="spearman_acoustic_vs_clinical_axis", color="#56B4E9", showfliers=False, width=0.35, ax=ax)
        sns.stripplot(data=d, y="spearman_acoustic_vs_clinical_axis", color="black", alpha=0.65, jitter=0.12, size=5.0, ax=ax)
    else:
        vals = d["spearman_acoustic_vs_clinical_axis"].to_numpy(float)
        ax.boxplot([vals], showfliers=False)
        rng = np.random.default_rng(123)
        ax.scatter(1 + rng.uniform(-0.08, 0.08, len(vals)), vals, color="black", alpha=0.65)
        ax.set_xticks([])
    ax.axhline(0, color="black", lw=1.0)
    ax.set_xlabel("All-clinical CCA axis 1")
    ax.set_ylabel("OOF Spearman correlation")
    ax.set_title("Repeated 5-fold random-split robustness")
    savefig(out_path)


# 图S02：negative controls。
def plot_figureS02_negative_controls(ctrl_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    if len(ctrl_df) == 0:
        return
    d = ctrl_df[ctrl_df.get("panel", pd.Series(dtype=str)).astype(str).eq("all_clinical")].copy()
    d = d[np.isfinite(d.get("spearman", pd.Series(dtype=float)))]
    if len(d) == 0:
        return
    observed = np.nan
    if len(summary_df):
        hit = summary_df[
            summary_df.get("panel", pd.Series(dtype=str)).astype(str).eq("all_clinical")
            & summary_df.get("component", pd.Series(dtype=float)).astype(float).eq(1)
        ]
        if "observed_spearman" in hit.columns and len(hit):
            observed = hit["observed_spearman"].dropna().iloc[0] if len(hit["observed_spearman"].dropna()) else np.nan
    labels = {
        "patient_label_permutation": "Patient-label permutation",
        "random_embedding": "Random embedding",
    }
    colors = {
        "patient_label_permutation": "#0072B2",
        "random_embedding": "#D55E00",
    }
    plt.figure(figsize=(7.4, 5.4))
    ax = plt.gca()
    for ct, sub in d.groupby("control_type"):
        ax.hist(sub["spearman"].dropna(), bins=24, alpha=0.36, color=colors.get(ct, None), label=labels.get(ct, ct), density=False)
        if HAS_SEABORN:
            sns.kdeplot(sub["spearman"].dropna(), color=colors.get(ct, None), lw=2.0, ax=ax)
    if np.isfinite(observed):
        ax.axvline(observed, color="black", lw=2.3, label="Observed")
    ax.axvline(0, color="black", lw=1, alpha=0.7)
    ax.set_xlabel("OOF Spearman correlation under control")
    ax.set_ylabel("Count")
    ax.set_title("Negative controls for all-clinical CCA alignment")
    ax.legend(frameon=False, fontsize=9.5)
    savefig(out_path)


# 图S03：age、sex、heart-rate 单独混杂控制 vs unadjusted。
def plot_figureS03_confounder_forest(conf_df: pd.DataFrame, alignment_summary: pd.DataFrame, out_path: Path) -> None:
    rows = []
    if len(alignment_summary):
        base = alignment_summary[
            alignment_summary.get("panel", pd.Series(dtype=str)).astype(str).eq("all_clinical")
            & alignment_summary.get("adjustment", pd.Series(dtype=str)).astype(str).eq("none")
            & alignment_summary.get("component", pd.Series(dtype=float)).astype(float).eq(1)
        ]
        if len(base):
            r = base.iloc[0]
            rows.append({"label": "Unadjusted", "value": r.get("spearman_acoustic_vs_clinical_axis", np.nan),
                         "lo": r.get("spearman_ci95_low", np.nan), "hi": r.get("spearman_ci95_high", np.nan),
                         "kind": "Unadjusted", "order": 0})
    if len(conf_df):
        labels = {
            "age_residualized": "Age residualized",
            "sex_residualized": "Sex residualized",
            "heart_rate_residualized": "Heart-rate residualized",
            "age_sex_residualized": "Age + sex residualized",
            "age_sex_heart_rate_residualized": "Age + sex + heart-rate residualized",
        }
        order = {"age_residualized": 1, "sex_residualized": 2, "heart_rate_residualized": 3,
                 "age_sex_residualized": 4, "age_sex_heart_rate_residualized": 5}
        d = conf_df[conf_df.get("component", pd.Series(dtype=float)).astype(float).eq(1)].copy()
        for _, r in d.iterrows():
            adj = str(r.get("adjustment", ""))
            if adj not in labels:
                continue
            rows.append({"label": labels[adj], "value": r.get("spearman_acoustic_vs_clinical_axis", np.nan),
                         "lo": r.get("spearman_ci95_low", np.nan), "hi": r.get("spearman_ci95_high", np.nan),
                         "kind": "Adjusted", "order": order.get(adj, 99)})
    df = pd.DataFrame(rows).sort_values("order")
    df = df[np.isfinite(df["value"])]
    if len(df) == 0:
        return
    plt.figure(figsize=(8.4, 4.7))
    ax = plt.gca()
    y = np.arange(len(df))
    colors = ["#000000" if k == "Unadjusted" else "#0072B2" for k in df["kind"]]
    for yi, (_, r), color in zip(y, df.iterrows(), colors):
        if np.isfinite(r["lo"]) and np.isfinite(r["hi"]):
            ax.plot([r["lo"], r["hi"]], [yi, yi], color=color, lw=2.2)
            ax.plot([r["lo"], r["lo"]], [yi - 0.07, yi + 0.07], color=color, lw=1.3)
            ax.plot([r["hi"], r["hi"]], [yi - 0.07, yi + 0.07], color=color, lw=1.3)
        ax.scatter([r["value"]], [yi], color=color, edgecolor="black", s=82, zorder=3)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.invert_yaxis()
    ax.set_xlabel("OOF Spearman correlation\n(all-clinical acoustic axis 1 vs clinical axis 1)")
    ax.set_title("Single-covariate confounder adjustment")
    savefig(out_path)


# 图S04：模型超参数敏感性（endpoint AUROC），heatmap。
def plot_figureS04_hyperparameter_endpoint_heatmap(hyper_ep_df: pd.DataFrame, out_path: Path) -> None:
    if len(hyper_ep_df) == 0 or "auroc" not in hyper_ep_df.columns:
        return
    d = hyper_ep_df[np.isfinite(hyper_ep_df["auroc"])].copy()
    if len(d) == 0:
        return
    if "n_axis_features" in d.columns:
        d = d[d["n_axis_features"].astype(int).eq(1)].copy()
    if len(d) == 0:
        return
    if "setting" in d.columns:
        d["setting_label"] = d["setting"].astype(str).str.replace("pca", "PCA", regex=False).str.replace("_comp", "/C", regex=False)
    elif {"n_pca_setting", "n_components_setting"}.issubset(d.columns):
        d["setting_label"] = "PCA" + d["n_pca_setting"].astype(str) + "/C" + d["n_components_setting"].astype(str)
    else:
        d["setting_label"] = "setting"
    d["endpoint_label"] = d["endpoint"].map(_endpoint_pretty)
    pivot = d.pivot_table(index="endpoint_label", columns="setting_label", values="auroc", aggfunc="mean")
    if pivot.empty:
        return
    plt.figure(figsize=(max(7.6, 0.75 * len(pivot.columns)), max(4.3, 0.43 * len(pivot.index))))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.5, vmax=0.9, linewidths=0.45, cbar_kws={"label": "AUROC"}, ax=ax)
    else:
        im = ax.imshow(pivot.values, aspect="auto", vmin=0.5, vmax=0.9)
        plt.colorbar(im, ax=ax, label="AUROC")
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Hyperparameter setting")
    ax.set_ylabel("")
    ax.set_title("Model hyperparameter sensitivity: endpoint AUROC (Axis 1)")
    savefig(out_path)


# 图S05：endpoint 阈值敏感性，heatmap。
def plot_figureS05_endpoint_threshold_heatmap(threshold_df: pd.DataFrame, out_path: Path) -> None:
    if len(threshold_df) == 0 or "auroc" not in threshold_df.columns:
        return
    d = threshold_df[np.isfinite(threshold_df["auroc"])].copy()
    if len(d) == 0:
        return
    if "n_axis_features" in d.columns:
        # Axis 1 is the manuscript main line; axis1+2 can stay in tables.
        d = d[d["n_axis_features"].astype(int).eq(1)].copy()
    if len(d) == 0:
        return
    d["endpoint_label"] = d["endpoint"].map(_endpoint_pretty)
    if "endpoint_family" not in d.columns:
        d["endpoint_family"] = d["endpoint"].astype(str).str.extract(r"^(EF_lt|NTproBNP_ge|NYHA_ge)", expand=False)
    d["axis_label"] = "Axis 1"
    row_order = [
        "EF <40", "EF <50",
        "NT-proBNP ≥125", "NT-proBNP ≥300", "NT-proBNP ≥900",
        "NYHA ≥1", "NYHA ≥2", "NYHA ≥3",
    ]
    d["row_order"] = d["endpoint_label"].map({v: i for i, v in enumerate(row_order)}).fillna(99)
    d = d.sort_values("row_order")
    pivot = d.pivot_table(index="endpoint_label", columns="axis_label", values="auroc", aggfunc="mean")
    pivot = pivot.reindex([r for r in row_order if r in pivot.index])
    if pivot.empty:
        return
    plt.figure(figsize=(4.4, max(4.8, 0.46 * len(pivot.index))))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0.5, vmax=0.9, linewidths=0.45, cbar_kws={"label": "AUROC"}, ax=ax)
    else:
        im = ax.imshow(pivot.values, aspect="auto", vmin=0.5, vmax=0.9)
        plt.colorbar(im, ax=ax, label="AUROC")
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Endpoint threshold sensitivity: AUROC")
    savefig(out_path)

def plot_all_results(table_dir: Path, fig_dir: Path, args) -> None:
    ensure_dir(fig_dir)

    alignment_summary = read_csv_optional(table_dir / "cca_panel_alignment_summary.csv")
    axis_scores = read_csv_optional(table_dir / "oof_cca_axis_scores_by_panel.csv")
    axis_ranked = read_csv_optional(table_dir / "allclinical_patient_axis_scores_ranked.csv")
    assoc_all = read_csv_optional(table_dir / "cca_axis_clinical_associations.csv")
    gradient_patient = read_csv_optional(table_dir / "clinical_gradient_patient_values.csv")
    gradient_summary = read_csv_optional(table_dir / "clinical_gradient_by_axis.csv")
    endpoint_summary = read_csv_optional(table_dir / "endpoint_validation_summary.csv")
    endpoint_pred = read_csv_optional(table_dir / "endpoint_validation_predictions.csv")

    # Robustness / supplement tables.
    lopo_summary = read_csv_optional(table_dir / "leave_one_position_out_alignment_summary.csv")
    position_contribution = read_csv_optional(table_dir / "position_contribution_alignment_summary.csv")
    repeated_cca_values = read_csv_optional(table_dir / "repeated_cv_cca_alignment_values.csv")
    negative_controls = read_csv_optional(table_dir / "negative_controls.csv")
    negative_control_summary = read_csv_optional(table_dir / "negative_control_summary.csv")
    confounder_summary = read_csv_optional(table_dir / "confounder_adjusted_alignment_summary.csv")
    hyper_endpoint = read_csv_optional(table_dir / "model_hyperparameter_sensitivity_endpoint.csv")
    threshold_summary = read_csv_optional(table_dir / "endpoint_threshold_sensitivity_summary.csv")

    if len(axis_scores):
        plot_figure01_axis1_scatter_line(
            axis_scores,
            alignment_summary,
            fig_dir / "figure_01_allclinical_axis1_acoustic_vs_clinical_scatter_line.png",
        )

    if len(assoc_all):
        plot_figure02_axis1_variable_lollipop(
            assoc_all,
            fig_dir / "figure_02_allclinical_axis1_variable_lollipop_ci.png",
        )

    if len(gradient_patient):
        plot_figure03_selected_continuous_box_points(
            gradient_patient,
            fig_dir / "figure_03_allclinical_axis1_continuous_profile_box_point_selected.png",
            component=1,
        )
        plot_figure04_selected_continuous_box_only(
            gradient_patient,
            fig_dir / "figure_04_allclinical_axis1_continuous_profile_box_selected.png",
            component=1,
        )

    if len(gradient_summary) or len(gradient_patient):
        plot_figure05_selected_ordinal_heatmap(
            gradient_summary,
            gradient_patient,
            fig_dir / "figure_05_allclinical_axis1_selected_ordinal_heatmap.png",
            component=1,
        )

    if len(endpoint_pred):
        for endpoint, pretty, fname in ENDPOINT_FIGURES:
            plot_endpoint_roc(endpoint_pred, endpoint_summary, endpoint, pretty, fig_dir / fname, args)

    # Figure 10: merged site robustness / contribution.
    plot_figure10_site_robustness_contribution(
        lopo_summary,
        position_contribution,
        alignment_summary,
        fig_dir / "figure_10_site_robustness_contribution_forest.png",
    )

    # Figure 11: ranked all-clinical acoustic CCA axis 1 score.
    plot_figure11_ranked_axis_score(
        axis_ranked,
        axis_scores,
        fig_dir / "figure_11_allclinical_axis1_ranked_score_plot.png",
    )

    # Supplementary robustness figures.
    plot_figureS01_repeated_random_split(
        repeated_cca_values,
        fig_dir / "figure_S01_repeated_random_split_allclinical_axis1.png",
    )
    plot_figureS02_negative_controls(
        negative_controls,
        negative_control_summary,
        fig_dir / "figure_S02_negative_controls_allclinical.png",
    )
    plot_figureS03_confounder_forest(
        confounder_summary,
        alignment_summary,
        fig_dir / "figure_S03_single_covariate_confounder_alignment.png",
    )
    plot_figureS04_hyperparameter_endpoint_heatmap(
        hyper_endpoint,
        fig_dir / "figure_S04_hyperparameter_endpoint_auroc_heatmap.png",
    )
    plot_figureS05_endpoint_threshold_heatmap(
        threshold_summary,
        fig_dir / "figure_S05_endpoint_threshold_sensitivity_auroc_heatmap.png",
    )


def parse_args():
    p = argparse.ArgumentParser(description="Regenerate focused all-clinical CCA figures from result tables")
    p.add_argument(
        "--result-dir",
        type=str,
        default="Clinical_alignment/outputs/clinically_anchored_acoustic_phenotyping_clean_v5",
        help="Output root containing tables/ and figures/. Ignored for table/fig dirs if those are explicitly given.",
    )
    p.add_argument("--table-dir", type=str, default=None, help="Directory containing CSV result tables")
    p.add_argument("--fig-dir", type=str, default=None, help="Directory where figures will be saved")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bootstrap", type=int, default=1000)
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
