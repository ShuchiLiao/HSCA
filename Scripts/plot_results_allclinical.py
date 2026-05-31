#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Focused plotting script for all-clinical acoustic-clinical CCA outputs.

This script reads CSV result tables produced by
run_clinically_anchored_acoustic_phenotyping_clean_v4.py and regenerates figures.

The plotting logic is intentionally centered on the all-clinical CCA axis:
1) acoustic CCA axis vs clinical CCA axis alignment;
2) acoustic CCA axis 1/2 two-dimensional map;
3) axis 1/2 correlation profile with clinical variables;
4) clinical profiles across acoustic-axis quartiles.

Variable colors are grouped into three clinical domains:
Function, Structure, and Valve.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

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

VARIABLE_GROUP: Dict[str, str] = {}
for _v in FUNCTION_VARS:
    VARIABLE_GROUP[_v] = "Function"
for _v in STRUCTURE_VARS:
    VARIABLE_GROUP[_v] = "Structure"
for _v in VALVE_VARS:
    VARIABLE_GROUP[_v] = "Valve"


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
    groups = ["Function", "Structure", "Valve"]
    if HAS_SEABORN:
        colors = sns.color_palette("Set2", n_colors=3)
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, 3))
    return dict(zip(groups, colors))


def get_all_clinical_axis_scores(axis_df: pd.DataFrame) -> pd.DataFrame:
    if len(axis_df) == 0:
        return pd.DataFrame()
    required = {"panel", "adjustment", "cca_acoustic_axis1", "cca_clinical_axis1"}
    if not required.issubset(axis_df.columns):
        log(f"Axis score table missing columns: {required - set(axis_df.columns)}")
        return pd.DataFrame()
    d = axis_df[
        axis_df["panel"].astype(str).eq("all_clinical")
        & axis_df["adjustment"].astype(str).eq("none")
    ].copy()
    d = d[np.isfinite(d["cca_acoustic_axis1"]) & np.isfinite(d["cca_clinical_axis1"])]
    return d


def lookup_alignment_stats(summary_df: pd.DataFrame, component: int = 1) -> str:
    if summary_df is None or len(summary_df) == 0:
        return ""
    if not {"panel", "component", "spearman_acoustic_vs_clinical_axis"}.issubset(summary_df.columns):
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


def plot_allclinical_axis1_scatter_line(axis_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    # 图一：All-clinical acoustic CCA axis 1 vs clinical CCA axis 1 散点+趋势线。
    # 目的：直接展示“声学轴”和“临床轴”在测试折病人中的对齐效果，而不是展示单变量相关。
    d = get_all_clinical_axis_scores(axis_df)
    if len(d) == 0:
        return
    x = d["cca_acoustic_axis1"].to_numpy(float)
    y = d["cca_clinical_axis1"].to_numpy(float)
    rho, pp, n = safe_spearman(x, y)

    plt.figure(figsize=(7.2, 6.4))
    ax = plt.gca()
    ax.scatter(x, y, s=22, alpha=0.28, color="0.25", edgecolor="none")

    if np.nanstd(x) > 1e-12 and np.nanstd(y) > 1e-12:
        xs = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), 120)
        slope, intercept = np.polyfit(x, y, 1)
        ax.plot(xs, slope * xs + intercept, color="black", lw=2.1, label="Linear trend")

        try:
            q = pd.qcut(pd.Series(x).rank(method="first"), 10, labels=False)
            tmp = pd.DataFrame({"x": x, "y": y, "bin": q})
            b = tmp.groupby("bin").agg(x_mean=("x", "mean"), y_mean=("y", "mean")).dropna()
            ax.plot(b["x_mean"], b["y_mean"], marker="o", lw=2.0, color="#D55E00", label="Binned mean")
        except Exception:
            pass

    ax.axhline(0, color="0.80", lw=0.9)
    ax.axvline(0, color="0.80", lw=0.9)
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


def plot_allclinical_acoustic_2d_map(axis_df: pd.DataFrame, out_path: Path) -> None:
    # 图二：All-clinical acoustic CCA axis 1/2 二维图。
    # 目的：展示病人在二维声学 CCA 空间中的分布，并用 clinical axis 1 给点着色，
    # 让读者看到二维声学空间与综合临床状态的关系。
    d = get_all_clinical_axis_scores(axis_df)
    required = {"cca_acoustic_axis1", "cca_acoustic_axis2", "cca_clinical_axis1"}
    if len(d) == 0 or not required.issubset(d.columns):
        log(f"Skip 2D acoustic map: missing {required - set(d.columns)}")
        return
    d = d[np.isfinite(d["cca_acoustic_axis1"]) & np.isfinite(d["cca_acoustic_axis2"]) & np.isfinite(d["cca_clinical_axis1"])]
    if len(d) == 0:
        return

    plt.figure(figsize=(7.4, 6.4))
    ax = plt.gca()
    sc = ax.scatter(
        d["cca_acoustic_axis1"],
        d["cca_acoustic_axis2"],
        c=d["cca_clinical_axis1"],
        cmap="coolwarm",
        s=24,
        alpha=0.72,
        edgecolor="none",
    )
    ax.axhline(0, color="0.80", lw=0.9)
    ax.axvline(0, color="0.80", lw=0.9)
    ax.set_xlabel("All-clinical acoustic CCA axis 1")
    ax.set_ylabel("All-clinical acoustic CCA axis 2")
    ax.set_title("Two-dimensional acoustic CCA space\ncolored by clinical CCA axis 1")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Clinical CCA axis 1 score")
    savefig(out_path)


def plot_allclinical_axis_variable_correlations(assoc_df: pd.DataFrame, out_path: Path) -> None:
    # 图三：All-clinical acoustic CCA axis 1/2 与各临床变量的相关系数图。
    # 目的：解释 all-clinical 声学轴 1 和 2 分别对应哪些临床变量方向。
    # 注意：这是轴解释图，不是主对齐证据；主对齐证据来自图一。
    if len(assoc_df) == 0:
        return
    required = {"panel", "adjustment", "component", "variable", "spearman_axis_variable"}
    if not required.issubset(assoc_df.columns):
        log(f"Skip variable correlation heatmap: missing {required - set(assoc_df.columns)}")
        return
    d = assoc_df[
        assoc_df["panel"].astype(str).eq("all_clinical")
        & assoc_df["adjustment"].astype(str).eq("none")
        & assoc_df["component"].astype(int).isin([1, 2])
    ].copy()
    d = d[np.isfinite(d["spearman_axis_variable"])]
    if len(d) == 0:
        return

    variables = [v for v in ALL_CLINICAL_VARS if v in set(d["variable"])]
    mat = pd.DataFrame(index=[PRETTY_LABELS.get(v, v) for v in variables], columns=["Axis 1", "Axis 2"], dtype=float)
    for v in variables:
        for c in [1, 2]:
            hit = d[(d["variable"].astype(str) == v) & (d["component"].astype(int) == c)]
            if len(hit):
                mat.loc[PRETTY_LABELS.get(v, v), f"Axis {c}"] = hit["spearman_axis_variable"].iloc[0]

    vmax = max(0.35, np.nanmax(np.abs(mat.values)) if np.isfinite(mat.values).any() else 0.35)
    plt.figure(figsize=(6.8, max(6.2, 0.46 * len(mat))))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.heatmap(
            mat,
            annot=True,
            fmt=".2f",
            linewidths=0.6,
            cmap="coolwarm",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            cbar_kws={"label": "Spearman correlation"},
            ax=ax,
        )
    else:
        im = ax.imshow(mat.values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label="Spearman correlation")
        ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns)
        ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index)

    pal = domain_palette()
    for tick, var in zip(ax.get_yticklabels(), variables):
        tick.set_color(pal.get(VARIABLE_GROUP.get(var), "black"))

    group_boundaries = []
    last_group = None
    for i, v in enumerate(variables):
        g = VARIABLE_GROUP.get(v)
        if last_group is not None and g != last_group:
            group_boundaries.append(i)
        last_group = g
    for b in group_boundaries:
        ax.axhline(b, color="black", lw=1.0)

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=pal[g], label=g, markersize=9)
        for g in ["Function", "Structure", "Valve"]
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.25, 1.0), fontsize=10)
    ax.set_title("Clinical interpretation of all-clinical acoustic CCA axes")
    ax.set_xlabel("Acoustic CCA axis")
    ax.set_ylabel("")
    savefig(out_path)


def plot_allclinical_continuous_profile_box_points(patient_long: pd.DataFrame, out_path: Path, component: int = 1) -> None:
    # 图四：All-clinical acoustic axis 四分位下连续临床变量的点云+箱线图。
    # 目的：把“临床轴高低”落到具体连续临床变量上，展示 EF/BNP/LA/LVEDD 等随声学轴的趋势。
    if len(patient_long) == 0:
        return
    required = {"panel", "adjustment", "component", "axis_group", "variable", "value"}
    if not required.issubset(patient_long.columns):
        log(f"Skip continuous clinical profile: missing {required - set(patient_long.columns)}")
        return
    d = patient_long[
        patient_long["panel"].astype(str).eq("all_clinical")
        & patient_long["adjustment"].astype(str).eq("none")
        & patient_long["component"].astype(int).eq(int(component))
        & (~patient_long["variable"].isin(ORDINAL_VARS))
    ].copy()
    d = d[np.isfinite(d["value"])]
    if len(d) == 0:
        return

    vars_order = [v for v in FUNCTION_VARS + STRUCTURE_VARS if v in set(d["variable"]) and v not in ORDINAL_VARS]
    n = len(vars_order)
    if n == 0:
        return
    n_cols = 2
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.1 * n_cols, 4.35 * n_rows), squeeze=False)
    pal = domain_palette()
    levels = ["Q1", "Q2", "Q3", "Q4"]

    for ax, var in zip(axes.ravel(), vars_order):
        sub = d[d["variable"].astype(str).eq(var)].copy()
        sub["axis_group"] = pd.Categorical(sub["axis_group"].astype(str), categories=levels, ordered=True)
        color = pal.get(VARIABLE_GROUP.get(var), "0.5")
        if HAS_SEABORN:
            sns.boxplot(
                data=sub,
                x="axis_group",
                y="value",
                color=color,
                showfliers=False,
                linewidth=1.25,
                boxprops={"alpha": 0.38},
                medianprops={"color": "black", "linewidth": 1.35},
                ax=ax,
            )
            sns.stripplot(
                data=sub,
                x="axis_group",
                y="value",
                color=color,
                size=2.4,
                alpha=0.25,
                jitter=0.24,
                ax=ax,
            )
        else:
            rng = np.random.default_rng(123)
            groups = [sub.loc[sub["axis_group"].astype(str) == g, "value"].dropna().to_numpy(float) for g in levels]
            ax.boxplot(groups, labels=levels, showfliers=False)
            for i, vals in enumerate(groups, start=1):
                ax.scatter(i + rng.uniform(-0.22, 0.22, len(vals)), vals, s=8, alpha=0.28, color=color)

        rho, pp, _ = safe_spearman(sub["axis_score"], sub["value"]) if "axis_score" in sub.columns else (np.nan, np.nan, 0)
        title = PRETTY_LABELS.get(var, var)
        if np.isfinite(rho):
            title += f"\nρ={rho:.2f}, {p_text(pp)}"
        ax.set_title(title)
        ax.set_xlabel("All-clinical acoustic axis 1 quartile")
        ax.set_ylabel("Clinical value")

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=pal[g], label=g, markersize=10)
        for g in ["Function", "Structure", "Valve"]
    ]
    fig.legend(handles=handles, frameon=False, loc="upper right", bbox_to_anchor=(0.98, 1.02), fontsize=10)
    fig.suptitle("Continuous clinical profile across all-clinical acoustic-axis quartiles", y=1.03, fontsize=16)
    savefig(out_path)


def plot_allclinical_ordinal_profile_heatmap(summary_df: pd.DataFrame, out_path: Path, component: int = 1) -> None:
    # 图五：All-clinical acoustic axis 四分位下等级变量的比例 heatmap。
    # 目的：等级变量不适合箱线图，改用 NYHA≥3 或 valve grade≥1/≥2 的比例展示临床负担趋势。
    if len(summary_df) == 0:
        return
    required = {"panel", "adjustment", "component", "axis_group", "variable"}
    if not required.issubset(summary_df.columns):
        log(f"Skip ordinal heatmap: missing {required - set(summary_df.columns)}")
        return
    d = summary_df[
        summary_df["panel"].astype(str).eq("all_clinical")
        & summary_df["adjustment"].astype(str).eq("none")
        & summary_df["component"].astype(int).eq(int(component))
        & summary_df["variable"].isin(ORDINAL_VARS)
    ].copy()
    if len(d) == 0:
        return

    rows = []
    for var in [v for v in FUNCTION_VARS + VALVE_VARS if v in set(d["variable"]) and v in ORDINAL_VARS]:
        sub = d[d["variable"].astype(str).eq(var)].copy()
        if len(sub) == 0:
            continue
        if var == "NYHA":
            metrics = [("NYHA ≥3", "prop_ge_3")]
        else:
            metrics = [(f"{PRETTY_LABELS.get(var, var)} ≥1", "prop_ge_1"), (f"{PRETTY_LABELS.get(var, var)} ≥2", "prop_ge_2")]
        for label, col in metrics:
            if col not in sub.columns:
                continue
            rec = {"row_label": label, "variable": var, "group": VARIABLE_GROUP.get(var, "Valve")}
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                hit = sub[sub["axis_group"].astype(str).eq(q)]
                rec[q] = float(hit[col].iloc[0]) if len(hit) else np.nan
            rows.append(rec)
    if not rows:
        return

    mat_df = pd.DataFrame(rows)
    mat = mat_df[["Q1", "Q2", "Q3", "Q4"]].astype(float)
    vmax = max(0.05, min(1.0, np.nanmax(mat.values) if np.isfinite(mat.values).any() else 1.0))
    plt.figure(figsize=(7.8, max(4.8, 0.48 * len(mat_df))))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.heatmap(
            mat,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            vmin=0,
            vmax=vmax,
            cmap="Blues",
            cbar_kws={"label": "Proportion"},
            ax=ax,
        )
        ax.set_yticklabels(mat_df["row_label"], rotation=0)
    else:
        im = ax.imshow(mat.values, aspect="auto", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label="Proportion")
        ax.set_xticks(range(4)); ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        ax.set_yticks(range(len(mat_df))); ax.set_yticklabels(mat_df["row_label"])

    pal = domain_palette()
    for tick, group in zip(ax.get_yticklabels(), mat_df["group"]):
        tick.set_color(pal.get(group, "black"))

    for b in np.where(mat_df["group"].ne(mat_df["group"].shift()).to_numpy())[0][1:]:
        ax.axhline(b, color="black", lw=1.0)

    ax.set_xlabel("All-clinical acoustic axis 1 quartile")
    ax.set_ylabel("")
    ax.set_title("Ordinal clinical profile across all-clinical acoustic-axis quartiles")
    savefig(out_path)


def plot_endpoint_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    # 图六：Endpoint AUROC 汇总图。
    if len(summary_df) == 0 or "auroc" not in summary_df.columns:
        return
    d = summary_df[np.isfinite(summary_df["auroc"])].copy()
    if len(d) == 0:
        return
    order = ["EF_lt_40", "NTproBNP_ge_300", "NYHA_ge_3", "LA_ge_40", "LVEDD_dilated"]
    d["order"] = d["endpoint"].map({e: i for i, e in enumerate(order)})
    d = d.sort_values("order")
    plt.figure(figsize=(8.8, max(4.4, 0.52 * len(d))))
    ax = plt.gca()
    y = np.arange(len(d))
    x = d["auroc"].to_numpy(float)
    lo = d["auroc_ci95_low"].to_numpy(float)
    hi = d["auroc_ci95_high"].to_numpy(float)
    ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt="o", capsize=3, color="black")
    ax.axvline(0.5, color="black", linestyle="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["endpoint"])
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("AUROC")
    ax.set_title("Endpoint readability using all-clinical CCA acoustic axes 1+2")
    savefig(out_path)


def plot_repeated_cca_alignment(values_df: pd.DataFrame, out_path: Path, component: int = 1) -> None:
    # 图七：Repeated 5-fold random-split robustness。
    if len(values_df) == 0:
        return
    d = values_df[
        values_df["status"].astype(str).eq("ok")
        & values_df["panel"].astype(str).eq("all_clinical")
        & values_df["component"].astype(int).eq(int(component))
    ].copy()
    d = d[np.isfinite(d["spearman_acoustic_vs_clinical_axis"])]
    if len(d) == 0:
        return
    plt.figure(figsize=(5.8, 5.4))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.boxplot(data=d, y="spearman_acoustic_vs_clinical_axis", color="white", showfliers=False, linewidth=1.3, ax=ax)
        sns.stripplot(data=d, y="spearman_acoustic_vs_clinical_axis", color="black", size=5, alpha=0.65, jitter=0.12, ax=ax)
    else:
        vals = d["spearman_acoustic_vs_clinical_axis"].dropna().to_numpy(float)
        ax.boxplot([vals], labels=["All clinical"], showfliers=False)
        ax.scatter(np.ones(len(vals)), vals, s=18, alpha=0.6, color="black")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("OOF Spearman correlation")
    ax.set_title(f"Repeated random-split robustness\nall-clinical CCA axis {component}")
    ax.set_xticks([])
    savefig(out_path)


def plot_all_results(table_dir: Path, fig_dir: Path, args) -> None:
    ensure_dir(fig_dir)

    alignment_summary = read_csv_optional(table_dir / "cca_panel_alignment_summary.csv")
    axis_scores = read_csv_optional(table_dir / "oof_cca_axis_scores_by_panel.csv")
    assoc_all = read_csv_optional(table_dir / "cca_axis_clinical_associations.csv")
    gradient_patient = read_csv_optional(table_dir / "clinical_gradient_patient_values.csv")
    gradient_summary = read_csv_optional(table_dir / "clinical_gradient_by_axis.csv")

    if len(axis_scores):
        plot_allclinical_axis1_scatter_line(
            axis_scores,
            alignment_summary,
            fig_dir / "figure_01_allclinical_axis1_acoustic_vs_clinical_scatter_line.png",
        )
        plot_allclinical_acoustic_2d_map(
            axis_scores,
            fig_dir / "figure_02_allclinical_acoustic_cca_axis1_axis2_2d_map.png",
        )

    if len(assoc_all):
        plot_allclinical_axis_variable_correlations(
            assoc_all,
            fig_dir / "figure_03_allclinical_axis1_axis2_clinical_variable_correlations.png",
        )

    if len(gradient_patient):
        plot_allclinical_continuous_profile_box_points(
            gradient_patient,
            fig_dir / "figure_04_allclinical_axis1_continuous_clinical_profile_box_point.png",
            component=1,
        )

    if len(gradient_summary):
        plot_allclinical_ordinal_profile_heatmap(
            gradient_summary,
            fig_dir / "figure_05_allclinical_axis1_ordinal_clinical_profile_heatmap.png",
            component=1,
        )

    endpoint_summary = read_csv_optional(table_dir / "endpoint_validation_summary.csv")
    if len(endpoint_summary):
        plot_endpoint_summary(endpoint_summary, fig_dir / "figure_06_endpoint_auroc_summary.png")

    repeated_cca = read_csv_optional(table_dir / "repeated_cv_cca_alignment_values.csv")
    if len(repeated_cca):
        plot_repeated_cca_alignment(
            repeated_cca,
            fig_dir / "figure_07_repeated_cv_allclinical_axis1_alignment.png",
            component=1,
        )


def parse_args():
    p = argparse.ArgumentParser(description="Regenerate focused all-clinical CCA figures from result tables")
    p.add_argument(
        "--result-dir",
        type=str,
        default="Clinical_alignment/outputs/clinically_anchored_acoustic_phenotyping_clean_v4",
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
