#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Focused plotting script for acoustic-clinical CCA outputs.

This version is intentionally simplified for the revised manuscript story:
1) Acoustic CCA axis 1 vs all-clinical clinical CCA axis 1;
2) Acoustic axis 1 interpretation by clinical variables;
3) selected clinical profiles across acoustic-axis quartiles;
4) endpoint ROC curves with CI bands using axis 1 only.

It reads CSV tables produced by run_clinically_anchored_acoustic_phenotyping_clean_v5.py
(or v4 where compatible) and only regenerates figures.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

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


FUNCTION_VARS = ["EF_Teich", "NTproBNP", "NYHA"]
STRUCTURE_VARS = ["LA_mm", "LVEDD_mm", "IVS_mm", "LVPW_mm"]
VALVE_VARS = ["MR_grade", "TR_grade", "AR_grade", "PR_grade", "AS_grade", "MS_grade"]
ALL_CLINICAL_VARS = FUNCTION_VARS + STRUCTURE_VARS + VALVE_VARS

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
    ("EF_lt_40", "EF <40%", "figure_06_roc_EF_lt_40_axis1.png"),
    ("NTproBNP_ge_900", "NT-proBNP ≥900 pg/mL", "figure_07_roc_NTproBNP_ge_900_axis1.png"),
    ("LVEDD_dilated", "LVEDD dilation", "figure_08_roc_LVEDD_dilated_axis1.png"),
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
    if not rows:
        return
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
    ax.set_xlabel("OOF Spearman correlation\n(acoustic axis 1 vs clinical axis 1)")
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
    plt.figure(figsize=(max(11.0, 1.05 * len(pivot.columns)), max(6.2, 0.72 * len(pivot.index))))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.5, vmax=0.9, linewidths=0.55, annot_kws={"fontsize": 18}, cbar_kws={"label": "AUROC"}, ax=ax)
    else:
        im = ax.imshow(pivot.values, aspect="auto", vmin=0.5, vmax=0.9)
        plt.colorbar(im, ax=ax, label="AUROC")
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Hyperparameter setting")
    ax.set_ylabel("")
    ax.set_title("Model hyperparameter sensitivity: endpoint AUROC (Axis 1)")
    savefig(out_path)


# 图S05：endpoint 阈值敏感性，heatmap，两列分别为 Axis 1 与 Axis 1+2。
def plot_figureS05_endpoint_threshold_heatmap(threshold_df: pd.DataFrame, out_path: Path) -> None:
    if len(threshold_df) == 0 or "auroc" not in threshold_df.columns:
        return
    d = threshold_df[np.isfinite(threshold_df["auroc"])].copy()
    if len(d) == 0:
        return

    d["endpoint_label"] = d["endpoint"].map(_endpoint_pretty)
    if "endpoint_family" not in d.columns:
        d["endpoint_family"] = d["endpoint"].astype(str).str.extract(r"^(EF_lt|NTproBNP_ge|NYHA_ge)", expand=False)

    # Keep both feature settings. If an older table lacks n_axis_features, treat it as Axis 1.
    if "n_axis_features" in d.columns:
        d["n_axis_features"] = pd.to_numeric(d["n_axis_features"], errors="coerce").fillna(1).astype(int)
        d = d[d["n_axis_features"].isin([1, 2])].copy()
        d["axis_label"] = d["n_axis_features"].map({1: "Axis 1", 2: "Axis 1+2"})
    elif "axis_feature_set" in d.columns:
        d["axis_label"] = (
            d["axis_feature_set"]
            .astype(str)
            .replace({"axis1": "Axis 1", "axis1_2": "Axis 1+2", "Axis 1": "Axis 1", "Axis 1+2": "Axis 1+2"})
        )
        d = d[d["axis_label"].isin(["Axis 1", "Axis 1+2"])].copy()
    else:
        d["axis_label"] = "Axis 1"

    if len(d) == 0:
        return

    row_order = [
        "EF <40", "EF <50",
        "NT-proBNP ≥125", "NT-proBNP ≥300", "NT-proBNP ≥900",
        "NYHA ≥1", "NYHA ≥2", "NYHA ≥3",
    ]
    axis_order = ["Axis 1", "Axis 1+2"]

    d["row_order"] = d["endpoint_label"].map({v: i for i, v in enumerate(row_order)}).fillna(99)
    d = d.sort_values(["row_order", "axis_label"])
    pivot = d.pivot_table(index="endpoint_label", columns="axis_label", values="auroc", aggfunc="mean")
    pivot = pivot.reindex([r for r in row_order if r in pivot.index])
    pivot = pivot.reindex(columns=[c for c in axis_order if c in pivot.columns])

    if pivot.empty:
        return

    plt.figure(figsize=(7.2, max(7.0, 0.82 * len(pivot.index))))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            vmin=0.5,
            vmax=0.9,
            linewidths=0.55,
            annot_kws={"fontsize": 18},
            cbar_kws={"label": "AUROC"},
            ax=ax,
        )
    else:
        im = ax.imshow(pivot.values, aspect="auto", vmin=0.5, vmax=0.9)
        plt.colorbar(im, ax=ax, label="AUROC")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=18)
    ax.set_xlabel("CCA acoustic features")
    ax.set_ylabel("")
    ax.set_title("Endpoint threshold sensitivity: AUROC")
    savefig(out_path)




# =============================================================================
# Publication plotting style
# =============================================================================

def setup_plotting() -> None:
    if HAS_SEABORN:
        sns.set_theme(style="white", context="talk")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["font.size"] = 21
    plt.rcParams["axes.labelsize"] = 21
    plt.rcParams["xtick.labelsize"] = 21
    plt.rcParams["ytick.labelsize"] = 21
    plt.rcParams["legend.fontsize"] = 21
    plt.rcParams["axes.titlesize"] = 21
    plt.rcParams["figure.titlesize"] = 21
    plt.rcParams["axes.spines.top"] = True
    plt.rcParams["axes.spines.right"] = True
    plt.rcParams["axes.grid"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300


def savefig(path: Path) -> None:
    fig = plt.gcf()
    # No titles in saved manuscript figures.
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")
    for ax in fig.axes:
        ax.set_title("")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    log(f"Saved figure: {path}")


def endpoint_color(endpoint: str) -> str:
    endpoint = str(endpoint)
    if endpoint.startswith("EF_"):
        return SELECTED_CONTINUOUS_COLORS["EF_Teich"]
    if endpoint.startswith("NTproBNP"):
        return SELECTED_CONTINUOUS_COLORS["NTproBNP"]
    if endpoint == "LVEDD_dilated":
        return SELECTED_CONTINUOUS_COLORS["LVEDD_mm"]
    if endpoint.startswith("NYHA"):
        return "#56B4E9"
    return "#0072B2"


def endpoint_panel_text(endpoint: str, pretty: str) -> str:
    """Label shown in the upper-left of individual ROC panels."""
    mapping = {
        "EF_lt_40": "EF <40%",
        "NTproBNP_ge_900": "NT-proBNP ≥900 pg/mL",
        "LVEDD_dilated": "LVEDD dilation",
        "NYHA_ge_3": "NYHA ≥3",
    }
    return mapping.get(str(endpoint), str(pretty))


# 图一：Acoustic CCA axis 1 vs clinical CCA axis 1 散点+趋势线。
def plot_figure01_axis1_scatter_line(axis_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    d = get_all_clinical_axis_scores(axis_df)
    if len(d) == 0:
        return
    x = d["cca_acoustic_axis1"].to_numpy(float)
    y = d["cca_clinical_axis1"].to_numpy(float)
    rho, pp, n = safe_spearman(x, y)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    scatter_color = "#0072B2"
    trend_color = "#4D4D4D"
    bin_color = "#CC79A7"
    ax.scatter(x, y, s=34, alpha=0.30, color=scatter_color, edgecolor="none", label="Patients")

    if np.nanstd(x) > 1e-12 and np.nanstd(y) > 1e-12:
        xs = np.linspace(-20, 25, 140)
        slope, intercept = np.polyfit(x, y, 1)
        ax.plot(xs, slope * xs + intercept, color=trend_color, lw=2.8, label="Linear trend")
        try:
            q = pd.qcut(pd.Series(x).rank(method="first"), 10, labels=False)
            tmp = pd.DataFrame({"x": x, "y": y, "bin": q})
            b = tmp.groupby("bin").agg(x_mean=("x", "mean"), y_mean=("y", "mean")).dropna()
            ax.plot(b["x_mean"], b["y_mean"], marker="o", markersize=7.0, lw=2.8, color=bin_color, label="Binned mean")
        except Exception:
            pass

    ax.axhline(0, color="0.82", lw=1.0)
    ax.axvline(0, color="0.82", lw=1.0)
    ax.set_xlim(-20, 25)
    ax.set_xlabel("Acoustic CCA axis 1 score")
    ax.set_ylabel("Clinical CCA axis 1 score")
    stats_txt = lookup_alignment_stats(summary_df, component=1)
    if not stats_txt and np.isfinite(rho):
        stats_txt = f"ρ={rho:.2f}, {p_text(pp)}, n={n}"
    if stats_txt:
        ax.text(
            0.97, 0.04, stats_txt,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=21,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="0.25", alpha=0.92),
        )
    ax.legend(frameon=False, loc="upper left")
    savefig(out_path)


# 图二：Acoustic CCA axis 1 与各临床变量的 lollipop 相关图（带95%CI）。
def plot_figure02_axis1_variable_lollipop(assoc_df: pd.DataFrame, out_path: Path) -> None:
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

    # plt.figure(figsize=(8, max(6, 0.55 * len(d))))
    plt.figure(figsize=(8, 10))
    ax = plt.gca()
    y = np.arange(len(d))
    x = d["spearman_axis_variable"].to_numpy(float)
    lo = d["spearman_ci95_low"].to_numpy(float) if "spearman_ci95_low" in d.columns else np.full(len(d), np.nan)
    hi = d["spearman_ci95_high"].to_numpy(float) if "spearman_ci95_high" in d.columns else np.full(len(d), np.nan)
    for yi, xi, li, hi_i, color in zip(y, x, lo, hi, colors):
        ax.hlines(yi, 0, xi, color=color, lw=3.4, alpha=0.95)
        if np.isfinite(li) and np.isfinite(hi_i):
            ax.plot([li, hi_i], [yi, yi], color="black", lw=1.25, alpha=0.90)
            ax.plot([li, li], [yi - 0.07, yi + 0.07], color="black", lw=1.25)
            ax.plot([hi_i, hi_i], [yi - 0.07, yi + 0.07], color="black", lw=1.25)
        ax.scatter([xi], [yi], s=110, color=color, edgecolor="black", linewidth=0.7, zorder=3)
    ax.axvline(0, color="black", lw=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels([PRETTY_LABELS.get(v, v) for v in d["variable"]])
    for tick, var in zip(ax.get_yticklabels(), d["variable"]):
        tick.set_color(pal.get(VARIABLE_GROUP.get(var, "Valve"), "black"))
    ax.set_xlabel("Spearman correlation with acoustic CCA axis 1")
    legend_name = {"Function": "Functional", "Structure": "Structural", "Valve": "Valvular"}
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=pal[g], markeredgecolor="black", label=legend_name[g], markersize=11)
        for g in ["Function", "Structure", "Valve"]
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left", fontsize=18)
    savefig(out_path)


# 图三：Selected continuous clinical profile，点云+箱线图，仅 EF、NT-proBNP、LA、LVEDD。
def plot_figure03_selected_continuous_box_points(patient_long: pd.DataFrame, out_path: Path, component: int = 1) -> None:
    d = _selected_continuous_data(patient_long, component)
    if len(d) == 0:
        return
    vars_order = [v for v in SELECTED_CONTINUOUS_VARS if v in set(d["variable"])]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.2), squeeze=False)
    pal = domain_palette()
    levels = ["Q1", "Q2", "Q3", "Q4"]
    for ax, var in zip(axes.ravel(), vars_order):
        sub = d[d["variable"].astype(str).eq(var)].copy()
        sub["axis_group"] = pd.Categorical(sub["axis_group"].astype(str), categories=levels, ordered=True)
        color = SELECTED_CONTINUOUS_COLORS.get(var, pal.get(VARIABLE_GROUP.get(var), "#0072B2"))
        if HAS_SEABORN:
            sns.boxplot(
                data=sub, x="axis_group", y="value", color=color, showfliers=False,
                linewidth=1.4, boxprops={"alpha": 0.36},
                medianprops={"color": "black", "linewidth": 1.55}, ax=ax,
            )
            sns.stripplot(data=sub, x="axis_group", y="value", color=color, size=4.6, alpha=0.30, jitter=0.25, ax=ax)
        else:
            rng = np.random.default_rng(123)
            groups = [sub.loc[sub["axis_group"].astype(str) == g, "value"].dropna().to_numpy(float) for g in levels]
            ax.boxplot(groups, labels=levels, showfliers=False)
            for i, vals in enumerate(groups, start=1):
                ax.scatter(i + rng.uniform(-0.22, 0.22, len(vals)), vals, s=24, alpha=0.30, color=color)
        rho, pp, _ = safe_spearman(sub.get("axis_score", np.arange(len(sub))), sub["value"])
        if np.isfinite(rho):
            # EF panel: lower-left; other panels: upper-left.
            if var == "EF_Teich":
                text_x, text_y, text_va = 0.04, 0.06, "bottom"
            else:
                text_x, text_y, text_va = 0.04, 0.94, "top"
            ax.text(
                text_x, text_y, f"ρ={rho:.2f}\n{p_text(pp)}",
                transform=ax.transAxes,
                ha="left", va=text_va,
                fontsize=21,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.25", alpha=0.90),
            )
        ax.set_xlabel("Acoustic axis 1 quartile")
        ax.set_ylabel(unit_label(var, sub["value"]))
    for ax in axes.ravel()[len(vars_order):]:
        ax.axis("off")
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
                col = f"prop_ge_{thr}"
                if len(sub) == 0 or col not in sub.columns:
                    continue
                rec = {"row_label": label, "group": _group}
                for q in ["Q1", "Q2", "Q3", "Q4"]:
                    hit = sub[sub["axis_group"].astype(str).eq(q)]
                    rec[q] = float(hit[col].iloc[0]) if len(hit) else np.nan
                rows.append(rec)

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
    plt.figure(figsize=(7.3, 4.6))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.heatmap(
            mat, annot=True, fmt=".2f", linewidths=0.85, linecolor="white",
            vmin=0, vmax=vmax, cmap="magma_r",
            annot_kws={"fontsize": 21},
            cbar_kws={"label": "Proportion"}, ax=ax
        )
        ax.set_yticklabels(mat_df["row_label"], rotation=0)
    else:
        im = ax.imshow(mat.values, aspect="auto", vmin=0, vmax=vmax, cmap="magma_r")
        plt.colorbar(im, ax=ax, label="Proportion")
        ax.set_xticks(range(4)); ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        ax.set_yticks(range(len(mat_df))); ax.set_yticklabels(mat_df["row_label"])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=21)
    pal = domain_palette()
    for tick, group in zip(ax.get_yticklabels(), mat_df["group"]):
        tick.set_color(pal.get(group, "black"))
    ax.set_xlabel("Acoustic axis 1 quartile")
    ax.set_ylabel("")
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

    color = endpoint_color(endpoint)
    plt.figure(figsize=(6.6, 6.1))
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", lw=1.2, label="_nolegend_")
    if np.isfinite(lo).any():
        ax.fill_between(fpr, lo, hi, color=color, alpha=0.18)
    lab = f"AUROC={auc:.3f}"
    if np.isfinite(auc_lo) and np.isfinite(auc_hi):
        lab += f" ({auc_lo:.3f}–{auc_hi:.3f})"
    ax.plot(fpr, tpr, lw=3.0, color=color, label=lab)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.text(
        0.04, 0.96, endpoint_panel_text(endpoint, pretty),
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=18,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.25", alpha=0.92),
    )
    ax.legend(frameon=False, loc="lower right", fontsize=18)
    savefig(out_path)


# 图十：site robustness / site contribution 合并 forest plot + CI。
def plot_figure10_site_robustness_contribution(
    lopo_df: pd.DataFrame,
    contribution_df: pd.DataFrame,
    alignment_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    rows = []
    full_val = np.nan
    if len(contribution_df):
        d = contribution_df[contribution_df.get("component", pd.Series(dtype=float)).astype(float).eq(1)].copy()
        col = _alignment_value_col(d)
        lo_col, hi_col = _ci_cols(d)
        if col:
            site_order = {"full_position_concat": 0, "only_A": 1, "only_E": 2, "only_M": 3, "only_P": 4, "only_T": 5}
            labels = {
                "full_position_concat": "full",
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
                if name == "full_position_concat":
                    full_val = r.get(col, np.nan)
                rows.append({
                    "label": labels.get(name, name),
                    "value": r.get(col, np.nan),
                    "lo": r.get(lo_col, np.nan) if lo_col else np.nan,
                    "hi": r.get(hi_col, np.nan) if hi_col else np.nan,
                    "kind": "Full" if name == "full_position_concat" else "Only-site",
                    "order": site_order[name],
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
                    "label": f"Leave {pos}",
                    "value": r.get(col, np.nan),
                    "lo": r.get(lo_col, np.nan) if lo_col else np.nan,
                    "hi": r.get(hi_col, np.nan) if hi_col else np.nan,
                    "kind": "",
                    "order": 10 + site_order.get(pos, 99),
                })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return
    df = df[np.isfinite(df["value"])].sort_values("order", ascending=True).reset_index(drop=True)
    if len(df) == 0:
        return
    if not np.isfinite(full_val):
        full_row = df[df["kind"].astype(str).eq("Full")]
        if len(full_row):
            full_val = float(full_row["value"].iloc[0])

    colors = {"Full": "#000000", "Only-site": "#0072B2", "Leave-one-site": "#D55E00"}
    plt.figure(figsize=(9.8, max(5.8, 0.52 * len(df))))
    ax = plt.gca()
    y = np.arange(len(df))
    for yi, (_, r) in zip(y, df.iterrows()):
        color = colors.get(r["kind"], "black")
        if np.isfinite(r["lo"]) and np.isfinite(r["hi"]):
            ax.plot([r["lo"], r["hi"]], [yi, yi], color=color, lw=2.6, alpha=0.95)
            ax.plot([r["lo"], r["lo"]], [yi - 0.08, yi + 0.08], color=color, lw=1.5)
            ax.plot([r["hi"], r["hi"]], [yi - 0.08, yi + 0.08], color=color, lw=1.5)
        ax.scatter([r["value"]], [yi], color=color, edgecolor="black", linewidth=0.8, s=96, zorder=3)

    if np.isfinite(full_val):
        ax.axvline(full_val, color="black", linestyle="--", lw=1.5, alpha=0.85)
    ax.axvline(0, color="black", lw=1.0, alpha=0.75)
    if len(df) > 6:
        ax.axhline(5.5, color="0.82", lw=1.2)

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.invert_yaxis()
    ax.set_xlabel("OOF Spearman correlation\n(acoustic axis 1 vs clinical axis 1)")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["Only-site"], markeredgecolor="black",  markersize=12),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["Leave-one-site"], markeredgecolor="black",  markersize=12),
        plt.Line2D([0], [0], color="black", linestyle="--", label="Position-concat full"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left")
    savefig(out_path)


# 图十一：CCA axis 1 得分排序图 / ranked score plot / empirical quantile plot。
def plot_figure11_ranked_axis_score(axis_rank_df: pd.DataFrame, axis_scores: pd.DataFrame, out_path: Path) -> None:
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
    plt.figure(figsize=(9.6, 5.6))
    ax = plt.gca()
    ax.plot(d["axis1_empirical_quantile"], d["cca_acoustic_axis1"], color="#1F1F1F", lw=1.8, alpha=0.85)
    if color_col:
        vals = d[color_col].to_numpy(float)
        sc = ax.scatter(
            d["axis1_empirical_quantile"],
            d["cca_acoustic_axis1"],
            c=vals,
            cmap="viridis",
            s=48,
            alpha=0.86,
            edgecolor="none",
            vmin=np.nanpercentile(vals, 2),
            vmax=np.nanpercentile(vals, 98),
        )
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("Clinical CCA axis 1 score")
    else:
        ax.scatter(d["axis1_empirical_quantile"], d["cca_acoustic_axis1"], color="#0072B2", s=48, alpha=0.78, edgecolor="none")
    ax.axhline(0, color="black", lw=1, alpha=0.72)
    for q in [0.25, 0.50, 0.75]:
        ax.axvline(q, color="0.72", linestyle="--", lw=1)
    ax.set_xlabel("Empirical quantile of acoustic CCA axis 1")
    ax.set_ylabel("Acoustic CCA axis 1 score")
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
    plt.figure(figsize=(5.8, 5.5))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.boxplot(data=d, y="spearman_acoustic_vs_clinical_axis", color="#56B4E9", showfliers=False, width=0.35, ax=ax)
        sns.stripplot(data=d, y="spearman_acoustic_vs_clinical_axis", color="black", alpha=0.65, jitter=0.12, size=6.0, ax=ax)
    else:
        vals = d["spearman_acoustic_vs_clinical_axis"].to_numpy(float)
        ax.boxplot([vals], showfliers=False)
        rng = np.random.default_rng(123)
        ax.scatter(1 + rng.uniform(-0.08, 0.08, len(vals)), vals, color="black", alpha=0.65, s=24)
        ax.set_xticks([])
    ax.axhline(0, color="black", lw=1.0)
    ax.set_ylim(0.3, 0.4)
    ax.set_xlabel("All-clinical CCA axis 1")
    ax.set_ylabel("OOF Spearman correlation")
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
    plt.figure(figsize=(7.6, 5.6))
    ax = plt.gca()
    for ct, sub in d.groupby("control_type"):
        ax.hist(sub["spearman"].dropna(), bins=24, alpha=0.36, color=colors.get(ct, None), label=labels.get(ct, ct), density=False)
        if HAS_SEABORN:
            sns.kdeplot(sub["spearman"].dropna(), color=colors.get(ct, None), lw=2.2, ax=ax)
    if np.isfinite(observed):
        ax.axvline(observed, color="#D62728", lw=3.0, label="Observed")
    ax.axvline(0, color="black", lw=1, alpha=0.7)
    ax.set_xlabel("OOF Spearman correlation under control")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=18)
    savefig(out_path)


# 图S06：endpoint validation 的 repeated-CV AUROC by split，箱线图。
def plot_figureS06_repeated_endpoint_auroc_by_split(repeated_endpoint_df: pd.DataFrame, out_path: Path) -> None:
    if len(repeated_endpoint_df) == 0 or "auroc" not in repeated_endpoint_df.columns:
        return
    d = repeated_endpoint_df[np.isfinite(repeated_endpoint_df["auroc"])].copy()
    if len(d) == 0:
        return
    if "n_axis_features" in d.columns:
        d["n_axis_features"] = pd.to_numeric(d["n_axis_features"], errors="coerce").fillna(1).astype(int)
        d = d[d["n_axis_features"].eq(1)].copy()
    elif "axis_feature_set" in d.columns:
        axis = d["axis_feature_set"].astype(str)
        d = d[axis.isin(["axis1", "Axis 1", "1"])].copy()
    if len(d) == 0:
        return
    endpoint_order = ["EF_lt_40", "NTproBNP_ge_900", "NTproBNP_ge_300", "NYHA_ge_3", "LVEDD_dilated"]
    d = d[d["endpoint"].astype(str).isin(set(endpoint_order))].copy()
    if len(d) == 0:
        return
    if "NTproBNP_ge_900" in set(d["endpoint"].astype(str)):
        d = d[~d["endpoint"].astype(str).eq("NTproBNP_ge_300")].copy()
        endpoint_order = ["EF_lt_40", "NTproBNP_ge_900", "NYHA_ge_3", "LVEDD_dilated"]
    else:
        endpoint_order = ["EF_lt_40", "NTproBNP_ge_300", "NYHA_ge_3", "LVEDD_dilated"]
    d["endpoint_label"] = d["endpoint"].map(_endpoint_pretty)
    ordered_labels = [_endpoint_pretty(e) for e in endpoint_order if e in set(d["endpoint"].astype(str))]
    d["endpoint_label"] = pd.Categorical(d["endpoint_label"], categories=ordered_labels, ordered=True)
    d = d.dropna(subset=["endpoint_label"]).copy()
    if len(d) == 0:
        return
    plt.figure(figsize=(8.6, 5.5))
    ax = plt.gca()
    palette = {
        _endpoint_pretty("EF_lt_40"): endpoint_color("EF_lt_40"),
        _endpoint_pretty("NTproBNP_ge_900"): endpoint_color("NTproBNP_ge_900"),
        _endpoint_pretty("NTproBNP_ge_300"): endpoint_color("NTproBNP_ge_300"),
        _endpoint_pretty("NYHA_ge_3"): endpoint_color("NYHA_ge_3"),
        _endpoint_pretty("LVEDD_dilated"): endpoint_color("LVEDD_dilated"),
    }
    if HAS_SEABORN:
        sns.boxplot(
            data=d, x="endpoint_label", y="auroc", order=ordered_labels,
            palette=[palette.get(x, "#0072B2") for x in ordered_labels],
            showfliers=False, width=0.56, linewidth=1.3, ax=ax,
        )
        sns.stripplot(
            data=d, x="endpoint_label", y="auroc", order=ordered_labels,
            color="black", alpha=0.58, jitter=0.18, size=5.2, ax=ax,
        )
    else:
        groups = [d.loc[d["endpoint_label"].astype(str).eq(lbl), "auroc"].dropna().to_numpy(float) for lbl in ordered_labels]
        ax.boxplot(groups, labels=ordered_labels, showfliers=False)
        rng = np.random.default_rng(42)
        for i, vals in enumerate(groups, start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, color="black", alpha=0.58, s=24)
    ax.axhline(0.5, color="black", linestyle="--", lw=1.05, alpha=0.75)
    ax.set_ylim(0.65, 0.8)
    ax.set_xlabel("")
    ax.set_ylabel("AUROC across repeated CV splits")
    ax.tick_params(axis="x", rotation=18)
    savefig(out_path)



# 图S07：All-clinical acoustic / clinical CCA axis 1 的成对距离可视化。
def plot_figureS07_pairwise_axis_distance(axis_df: pd.DataFrame, out_path: Path) -> None:
    """
    Pairwise absolute distances along held-out CCA axis 1.

    x-axis: |acoustic CCA axis 1_i - acoustic CCA axis 1_j|
    y-axis: |clinical CCA axis 1_i - clinical CCA axis 1_j|

    This is a supplementary visualization of pairwise ordering structure.
    It uses out-of-fold all-clinical axis scores only.
    """
    d = get_all_clinical_axis_scores(axis_df)
    needed = {"cca_acoustic_axis1", "cca_clinical_axis1"}
    if len(d) == 0 or not needed.issubset(d.columns):
        return

    d = d[np.isfinite(d["cca_acoustic_axis1"]) & np.isfinite(d["cca_clinical_axis1"])].copy()
    if len(d) < 3:
        return

    xa = d["cca_acoustic_axis1"].to_numpy(float)
    yc = d["cca_clinical_axis1"].to_numpy(float)
    n = len(d)

    # Upper-triangular patient pairs.
    ii, jj = np.triu_indices(n, k=1)
    dx = np.abs(xa[ii] - xa[jj])
    dy = np.abs(yc[ii] - yc[jj])
    keep = np.isfinite(dx) & np.isfinite(dy)
    dx = dx[keep]
    dy = dy[keep]
    if len(dx) == 0:
        return

    rho, pp, _ = safe_spearman(dx, dy)

    plt.figure(figsize=(8.8, 7.0))
    ax = plt.gca()

    hb = ax.hexbin(
        dx,
        dy,
        gridsize=42,
        mincnt=1,
        cmap="viridis",
        linewidths=0,
        bins=None,
    )
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label("Pair count")

    # Binned median trend with IQR ribbon.
    n_bins = 16
    try:
        edges = np.quantile(dx, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) >= 4:
            mids, meds, q1s, q3s = [], [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = (dx >= lo) & (dx <= hi if hi == edges[-1] else dx < hi)
                vals = dy[sel]
                xvals = dx[sel]
                if len(vals) < 15:
                    continue
                mids.append(np.median(xvals))
                meds.append(np.median(vals))
                q1s.append(np.quantile(vals, 0.25))
                q3s.append(np.quantile(vals, 0.75))
            if len(mids) >= 2:
                mids = np.asarray(mids, dtype=float)
                meds = np.asarray(meds, dtype=float)
                q1s = np.asarray(q1s, dtype=float)
                q3s = np.asarray(q3s, dtype=float)
                ax.fill_between(mids, q1s, q3s, color="#D55E00", alpha=0.20, linewidth=0)
                ax.plot(mids, meds, color="#B24A3A", lw=3.0, label="Binned median trend")
                ax.legend(frameon=False, loc="upper right")
    except Exception:
        pass

    ax.set_xlabel("Pairwise distance in acoustic CCA axis 1")
    ax.set_ylabel("Pairwise distance in clinical CCA axis 1")

    if np.isfinite(rho):
        txt = f"Pairwise Spearman ρ={rho:.2f}\n{p_text(pp)}\nPairs={len(dx):,}"
        ax.text(
            0.97, 0.04, txt,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=21,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="0.25", alpha=0.92),
        )

    savefig(out_path)


# =============================================================================

def plot_figureS12_endpoint_auroc_by_site_configuration_axis1(position_endpoint_summary: pd.DataFrame, out_path: Path) -> None:
    """Forest plot of endpoint AUROC across auscultation-site configurations.

    Uses position_endpoint_validation_summary.csv generated by the site endpoint
    validation run. Only the four manuscript endpoints are included:
    EF <40%, NT-proBNP >=900, NYHA >=3, and LVEDD dilation.
    """
    if len(position_endpoint_summary) == 0:
        return

    required = {"endpoint", "position_analysis", "auroc"}
    if not required.issubset(position_endpoint_summary.columns):
        log(f"Skip figureS12: missing {required - set(position_endpoint_summary.columns)}")
        return

    d = position_endpoint_summary.copy()

    # Keep axis 1 only. New tables have axis_feature_set; older compatible
    # tables may only have n_axis_features.
    if "axis_feature_set" in d.columns:
        d = d[d["axis_feature_set"].astype(str).isin(["axis1", "Axis 1", "1"])].copy()
    elif "n_axis_features" in d.columns:
        d = d[pd.to_numeric(d["n_axis_features"], errors="coerce").fillna(1).astype(int).eq(1)].copy()

    endpoint_order = ["EF_lt_40", "NTproBNP_ge_900", "NYHA_ge_3", "LVEDD_dilated"]
    endpoint_labels = {
        "EF_lt_40": "EF <40%",
        "NTproBNP_ge_900": "NT-proBNP ≥900",
        "NYHA_ge_3": "NYHA ≥3",
        "LVEDD_dilated": "LVEDD dilation",
    }
    d = d[d["endpoint"].astype(str).isin(endpoint_order)].copy()
    d["auroc"] = pd.to_numeric(d["auroc"], errors="coerce")
    d = d[np.isfinite(d["auroc"])].copy()
    if len(d) == 0:
        return

    config_order = [
        "full_position_concat",
        "only_A", "only_E", "only_M", "only_P", "only_T",
        "leave_A", "leave_E", "leave_M", "leave_P", "leave_T",
    ]
    config_labels = {
        "full_position_concat": "Full",
        "only_A": "Only A", "only_E": "Only E", "only_M": "Only M", "only_P": "Only P", "only_T": "Only T",
        "leave_A": "Leave A", "leave_E": "Leave E", "leave_M": "Leave M", "leave_P": "Leave P", "leave_T": "Leave T",
    }
    config_order = [c for c in config_order if c in set(d["position_analysis"].astype(str))]
    endpoint_order = [e for e in endpoint_order if e in set(d["endpoint"].astype(str))]
    if not config_order or not endpoint_order:
        return

    colors = {
        "full_position_concat": "#0072B2",
        "only_A": "#BBD3F2",
        "only_E": "#E69F00",
        "only_M": "#F6B26B",
        "only_P": "#009E73",
        "only_T": "#8CD17D",
        "leave_A": "#D62728",
        "leave_E": "#FF9896",
        "leave_M": "#7E57C2",
        "leave_P": "#B39DDB",
        "leave_T": "#8C564B",
    }

    y_base = np.arange(len(endpoint_order), dtype=float)
    offsets = np.linspace(-0.30, 0.30, len(config_order)) if len(config_order) > 1 else np.array([0.0])

    plt.figure(figsize=(12.2, max(5.8, 1.0 * len(endpoint_order) + 1.8)))
    ax = plt.gca()

    all_los, all_his = [], []
    for j, cfg in enumerate(config_order):
        sub = d[d["position_analysis"].astype(str).eq(cfg)].copy()
        xs, ys, xerr_low, xerr_high = [], [], [], []
        for i, ep in enumerate(endpoint_order):
            hit = sub[sub["endpoint"].astype(str).eq(ep)].copy()
            if len(hit) == 0:
                continue
            row = hit.iloc[0]
            x = float(row["auroc"])
            lo = pd.to_numeric(row.get("auroc_ci95_low", np.nan), errors="coerce")
            hi = pd.to_numeric(row.get("auroc_ci95_high", np.nan), errors="coerce")
            lo = float(lo) if pd.notna(lo) else np.nan
            hi = float(hi) if pd.notna(hi) else np.nan
            xs.append(x)
            ys.append(y_base[i] + offsets[j])
            xerr_low.append(max(0.0, x - lo) if np.isfinite(lo) else 0.0)
            xerr_high.append(max(0.0, hi - x) if np.isfinite(hi) else 0.0)
            all_los.append(lo if np.isfinite(lo) else x)
            all_his.append(hi if np.isfinite(hi) else x)
        if len(xs) == 0:
            continue
        ax.errorbar(
            xs, ys,
            xerr=[xerr_low, xerr_high],
            fmt="o", ms=5.8, lw=1.6, elinewidth=1.6, capsize=3.2,
            color=colors.get(cfg, "#333333"),
            markeredgecolor="white", markeredgewidth=0.6,
            label=config_labels.get(cfg, cfg), zorder=3,
        )

    ax.axvline(0.5, color="black", linestyle="--", lw=1.25, alpha=0.85)
    for yi in y_base[:-1] + 0.5:
        ax.axhline(yi, color="0.88", lw=1.0, zorder=0)

    ax.set_yticks(y_base)
    ax.set_yticklabels([endpoint_labels.get(e, e) for e in endpoint_order])
    ax.set_xlabel("AUROC")
    ax.set_ylabel("")

    if all_los and all_his:
        xmin = max(0.45, float(np.nanmin(all_los)) - 0.035)
        xmax = min(0.90, float(np.nanmax(all_his)) + 0.035)
        if xmax - xmin < 0.18:
            mid = (xmax + xmin) / 2
            xmin = max(0.45, mid - 0.09)
            xmax = min(0.90, mid + 0.09)
        ax.set_xlim(xmin, xmax)
    else:
        ax.set_xlim(0.45, 0.85)

    ax.legend(
        frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left",
        borderaxespad=0, fontsize=18, handlelength=1.8,
    )
    savefig(out_path)
# Added manuscript figures: representation retrieval, clinical trends,
# endpoint score distribution, acoustic descriptors, and split site panel
# =============================================================================

ROUTE_PRETTY = {"beats": "BEATs", "panns": "PANNs", "ast": "AST", "ead": "EAD"}
ROUTE_ORDER = ["ead", "panns", "ast", "beats"]
ROUTE_COLORS = {"EAD": "#7F7F7F", "PANNs": "#D55E00", "AST": "#0072B2", "BEATs": "#009E73"}


def resolve_external_csv(path_like: str | None, fallback_name: str) -> Path | None:
    if path_like:
        p = Path(path_like)
        if p.exists():
            return p
    fb = Path("/mnt/data") / fallback_name
    if fb.exists():
        return fb
    if path_like:
        return Path(path_like)
    return None


def mean_ci_normal(vals: np.ndarray) -> Tuple[float, float, float, int]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    m = float(np.mean(vals))
    if n < 2:
        return m, np.nan, np.nan, n
    se = float(np.std(vals, ddof=1) / np.sqrt(n))
    return m, m - 1.96 * se, m + 1.96 * se, n


def binomial_ci_normal(k: int, n: int) -> Tuple[float, float, float]:
    if n <= 0:
        return np.nan, np.nan, np.nan
    p = k / n
    se = np.sqrt(max(p * (1 - p), 0) / n)
    return float(p), max(0.0, float(p - 1.96 * se)), min(1.0, float(p + 1.96 * se))


def zscore_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    sd = x.std(skipna=True)
    if not np.isfinite(sd) or sd < 1e-12:
        return x * np.nan
    return (x - x.mean(skipna=True)) / sd


def plot_retrieval_map_summary(retrieval_df: pd.DataFrame, metric_col: str, out_path: Path) -> None:
    """mAP across window settings: route-level mean ± SD plus individual window-setting points."""
    if len(retrieval_df) == 0 or metric_col not in retrieval_df.columns:
        return
    d = retrieval_df.copy()
    if not {"route_name", "window_setting"}.issubset(d.columns):
        return
    d["route_key"] = d["route_name"].astype(str).str.lower()
    d["route_label"] = d["route_key"].map(ROUTE_PRETTY).fillna(d["route_name"].astype(str))
    d = d[d["route_key"].isin(ROUTE_ORDER)].copy()
    d[metric_col] = pd.to_numeric(d[metric_col], errors="coerce")
    d = d[np.isfinite(d[metric_col])]
    if len(d) == 0:
        return

    labels = [ROUTE_PRETTY[r] for r in ROUTE_ORDER if r in set(d["route_key"])]
    positions = np.arange(len(labels))
    plt.figure(figsize=(7.8, 5.6))
    ax = plt.gca()
    rng = np.random.default_rng(42)

    for i, lab in enumerate(labels):
        color = ROUTE_COLORS.get(lab, "#0072B2")
        vals = d.loc[d["route_label"].eq(lab), metric_col].to_numpy(float)
        mean = np.mean(vals)
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        ax.errorbar(i, mean, yerr=sd, fmt="o", color=color, ecolor=color, elinewidth=2.4,
                    capsize=7, markersize=10, markeredgecolor="black", zorder=3)
        sub = d[d["route_label"].eq(lab)].copy().sort_values("window_setting")
        jitter = rng.uniform(-0.09, 0.09, len(sub))
        ax.scatter(np.full(len(sub), i) + jitter, sub[metric_col], s=70, color=color,
                   alpha=0.60, edgecolor="black", linewidth=0.4)
        for xj, (_, r) in zip(np.full(len(sub), i) + jitter, sub.iterrows()):
            ax.text(xj, r[metric_col] + 0.012, str(r["window_setting"]),
                    ha="center", va="bottom", fontsize=13)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mAP")
    ax.set_ylim(0, max(1.0, float(d[metric_col].max()) + 0.08))
    savefig(out_path)


def plot_figure15_five_view_retrieval(retrieval_df: pd.DataFrame, out_path: Path) -> None:
    plot_retrieval_map_summary(retrieval_df, "five_view_mAP", out_path)


def plot_figure16_single_to_four_retrieval(retrieval_df: pd.DataFrame, out_path: Path) -> None:
    plot_retrieval_map_summary(retrieval_df, "single_to_four_mAP", out_path)


def plot_figure17_continuous_clinical_trend(patient_long: pd.DataFrame, out_path: Path, component: int = 1) -> None:
    d = _selected_continuous_data(patient_long, component)
    if len(d) == 0:
        return
    levels = ["Q1", "Q2", "Q3", "Q4"]
    d = d[d["axis_group"].astype(str).isin(levels)].copy()
    d["axis_group"] = pd.Categorical(d["axis_group"].astype(str), categories=levels, ordered=True)
    d["z_value"] = d.groupby("variable")["value"].transform(zscore_series)
    colors = {v: SELECTED_CONTINUOUS_COLORS.get(v, "#0072B2") for v in SELECTED_CONTINUOUS_VARS}
    plt.figure(figsize=(8.6, 6.2))
    ax = plt.gca()
    x_pos = np.arange(len(levels))
    for var in SELECTED_CONTINUOUS_VARS:
        sub = d[d["variable"].astype(str).eq(var)].copy()
        if len(sub) == 0:
            continue
        means, los, his = [], [], []
        for g in levels:
            vals = sub.loc[sub["axis_group"].astype(str).eq(g), "z_value"].to_numpy(float)
            m, lo, hi, _ = mean_ci_normal(vals)
            means.append(m); los.append(lo); his.append(hi)
        means = np.asarray(means, dtype=float)
        yerr = np.vstack([means - np.asarray(los), np.asarray(his) - means])
        ax.errorbar(x_pos, means, yerr=yerr, marker="o", markersize=8.5, lw=2.4,
                    capsize=5, color=colors.get(var, "#0072B2"), label=PRETTY_LABELS.get(var, var))
    ax.axhline(0, color="black", lw=1.0, alpha=0.75)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(levels)
    ax.set_xlabel("Acoustic CCA axis 1 quartile")
    ax.set_ylabel("Standardized clinical value (z-score)")
    ax.legend(frameon=False, loc="best", fontsize=18)
    savefig(out_path)


def plot_figure18_threshold_clinical_trend(patient_long: pd.DataFrame, out_path: Path, component: int = 1) -> None:
    if len(patient_long) == 0:
        return
    required = {"panel", "adjustment", "component", "axis_group", "variable", "value"}
    if not required.issubset(patient_long.columns):
        return
    targets = [
        ("NYHA", "NYHA ≥3", 3, "#009E73"),
        ("MR_grade", "MR grade ≥2", 2, "#0072B2"),
        ("TR_grade", "TR grade ≥2", 2, "#56B4E9"),
        ("AR_grade", "AR grade ≥2", 2, "#CC79A7"),
    ]
    levels = ["Q1", "Q2", "Q3", "Q4"]
    d = patient_long[
        patient_long["panel"].astype(str).eq("all_clinical")
        & patient_long["adjustment"].astype(str).eq("none")
        & patient_long["component"].astype(int).eq(int(component))
    ].copy()
    plt.figure(figsize=(8.6, 6.2))
    ax = plt.gca()
    x_pos = np.arange(len(levels))
    for var, label, thr, color in targets:
        sub = d[d["variable"].astype(str).eq(var)].copy()
        if len(sub) == 0:
            continue
        ps, los, his = [], [], []
        for g in levels:
            vals = pd.to_numeric(sub.loc[sub["axis_group"].astype(str).eq(g), "value"], errors="coerce").dropna().to_numpy(float)
            k = int(np.sum(vals >= thr))
            p, lo, hi = binomial_ci_normal(k, len(vals))
            ps.append(p); los.append(lo); his.append(hi)
        ps = np.asarray(ps, dtype=float)
        yerr = np.vstack([ps - np.asarray(los), np.asarray(his) - ps])
        ax.errorbar(x_pos, ps, yerr=yerr, marker="o", markersize=8.5, lw=2.4,
                    capsize=5, color=color, label=label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(levels)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Acoustic CCA axis 1 quartile")
    ax.set_ylabel("Proportion")
    ax.legend(frameon=False, loc="best", fontsize=18)
    savefig(out_path)


def plot_figure19_endpoint_score_distribution(axis_ranked: pd.DataFrame, axis_scores: pd.DataFrame, out_path: Path) -> None:
    if len(axis_ranked):
        d = axis_ranked.copy()
    else:
        d = get_all_clinical_axis_scores(axis_scores).copy()
    if len(d) == 0 or "cca_acoustic_axis1" not in d.columns:
        return
    endpoints = [
        ("EF_lt_40", "EF <40%", "#009E73"),
        ("NTproBNP_ge_900", "NT-proBNP ≥900", "#CC79A7"),
        ("LVEDD_dilated", "LVEDD dilated", "#0072B2"),
        ("NYHA_ge_3", "NYHA ≥3", "#56B4E9"),
    ]
    rows = []
    for col, label, color in endpoints:
        if col not in d.columns:
            continue
        sub = d[[col, "cca_acoustic_axis1"]].copy()
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub[sub[col].isin([0, 1]) & np.isfinite(sub["cca_acoustic_axis1"])]
        if len(sub) == 0:
            continue
        sub["endpoint"] = label
        sub["status"] = np.where(sub[col].astype(int).eq(1), "Positive", "Negative")
        rows.append(sub[["endpoint", "status", "cca_acoustic_axis1"]])
    if not rows:
        return
    long = pd.concat(rows, ignore_index=True)
    ordered = [label for _, label, _ in endpoints if label in set(long["endpoint"])]
    plt.figure(figsize=(9.4, 6.2))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.boxplot(data=long, y="endpoint", x="cca_acoustic_axis1", hue="status",
                    order=ordered, hue_order=["Negative", "Positive"],
                    palette={"Negative": "#D0D0D0", "Positive": "#D55E00"},
                    showfliers=False, linewidth=1.25, ax=ax)
        sns.stripplot(data=long, y="endpoint", x="cca_acoustic_axis1", hue="status",
                      order=ordered, hue_order=["Negative", "Positive"],
                      dodge=True, color="black", alpha=0.25, size=3.2, ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], frameon=False, fontsize=18, loc="lower right")
    ax.axvline(0, color="black", lw=1.0, alpha=0.75)
    ax.set_xlabel("Acoustic CCA axis 1 score")
    ax.set_ylabel("")
    savefig(out_path)


def bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    rho, _, n = safe_spearman(x, y)
    if n < 10 or not np.isfinite(rho):
        return rho, np.nan, np.nan
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        r, _, _ = safe_spearman(x[idx], y[idx])
        if np.isfinite(r):
            vals.append(r)
    if len(vals) < 20:
        return rho, np.nan, np.nan
    return rho, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def prepare_descriptor_with_axis(descriptor_df: pd.DataFrame, axis_ranked: pd.DataFrame, axis_scores: pd.DataFrame) -> pd.DataFrame:
    if len(descriptor_df) == 0 or "patient_id" not in descriptor_df.columns:
        return pd.DataFrame()
    if len(axis_ranked) and "patient_id" in axis_ranked.columns and "cca_acoustic_axis1" in axis_ranked.columns:
        a = axis_ranked[["patient_id", "cca_acoustic_axis1"]].copy()
        if "axis1_quartile" in axis_ranked.columns:
            a["axis1_quartile"] = axis_ranked["axis1_quartile"]
    else:
        a = get_all_clinical_axis_scores(axis_scores)
        if len(a) == 0 or "patient_id" not in a.columns:
            return pd.DataFrame()
        a = a[["patient_id", "cca_acoustic_axis1"]].copy()
        a["axis1_quartile"] = pd.qcut(a["cca_acoustic_axis1"].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
    return descriptor_df.merge(a, on="patient_id", how="inner")


def plot_figureS08_acoustic_descriptor_associations(descriptor_df: pd.DataFrame, axis_ranked: pd.DataFrame, axis_scores: pd.DataFrame, out_path: Path, n_boot: int = 1000, seed: int = 42) -> None:
    d = prepare_descriptor_with_axis(descriptor_df, axis_ranked, axis_scores)
    if len(d) == 0:
        return
    desc_cols = [c for c in d.columns if c not in {"patient_id", "cca_acoustic_axis1", "axis1_quartile"} and pd.api.types.is_numeric_dtype(d[c])]
    rows = []
    for i, c in enumerate(desc_cols):
        rho, lo, hi = bootstrap_spearman_ci(d["cca_acoustic_axis1"].to_numpy(float), pd.to_numeric(d[c], errors="coerce").to_numpy(float), n_boot=n_boot, seed=seed + i)
        rows.append({"descriptor": c, "rho": rho, "lo": lo, "hi": hi})
    res = pd.DataFrame(rows).dropna(subset=["rho"])
    if len(res) == 0:
        return
    res = res.sort_values("rho", ascending=True)
    plt.figure(figsize=(10.4, max(5.6, 0.55 * len(res))))
    ax = plt.gca()
    y = np.arange(len(res))
    colors = np.where(res["rho"] >= 0, "#D55E00", "#0072B2")
    for yi, (_, r), color in zip(y, res.iterrows(), colors):
        ax.hlines(yi, 0, r["rho"], color=color, lw=3.0)
        if np.isfinite(r["lo"]) and np.isfinite(r["hi"]):
            ax.plot([r["lo"], r["hi"]], [yi, yi], color="black", lw=1.15)
        ax.scatter([r["rho"]], [yi], s=95, color=color, edgecolor="black", linewidth=0.6, zorder=3)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(res["descriptor"])
    ax.set_xlabel("Spearman correlation with acoustic CCA axis 1")
    savefig(out_path)


def plot_figureS09_acoustic_descriptor_quantile_trend(descriptor_df: pd.DataFrame, axis_ranked: pd.DataFrame, axis_scores: pd.DataFrame, out_path: Path) -> None:
    d = prepare_descriptor_with_axis(descriptor_df, axis_ranked, axis_scores)
    if len(d) == 0:
        return
    desc_cols = [c for c in d.columns if c not in {"patient_id", "cca_acoustic_axis1", "axis1_quartile"} and pd.api.types.is_numeric_dtype(d[c])]
    if len(desc_cols) == 0:
        return
    for c in desc_cols:
        d[c + "__z"] = zscore_series(d[c])
    if "axis1_quartile" not in d.columns or d["axis1_quartile"].isna().all():
        d["axis1_quartile"] = pd.qcut(d["cca_acoustic_axis1"].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
    d["axis1_quartile_clean"] = d["axis1_quartile"].astype(str).str.extract(r"(Q[1-4])", expand=False)
    levels = ["Q1", "Q2", "Q3", "Q4"]
    plt.figure(figsize=(10.2, 6.8))
    ax = plt.gca()
    x_pos = np.arange(len(levels))
    cmap = plt.get_cmap("tab10")
    for idx, c in enumerate(desc_cols):
        means, los, his = [], [], []
        for g in levels:
            vals = d.loc[d["axis1_quartile_clean"].eq(g), c + "__z"].to_numpy(float)
            m, lo, hi, _ = mean_ci_normal(vals)
            means.append(m); los.append(lo); his.append(hi)
        means = np.asarray(means)
        ax.errorbar(x_pos, means, yerr=np.vstack([means - np.asarray(los), np.asarray(his) - means]),
                    marker="o", markersize=6.8, lw=1.9, capsize=4, color=cmap(idx % 10), label=c)
    ax.axhline(0, color="black", lw=1.0, alpha=0.75)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(levels)
    ax.set_xlabel("Acoustic CCA axis 1 quartile")
    ax.set_ylabel("Standardized acoustic descriptor (z-score)")
    ax.legend(frameon=False, fontsize=13, loc="center left", bbox_to_anchor=(1.02, 0.5))
    savefig(out_path)


def plot_figure10_site_robustness_contribution(lopo_df: pd.DataFrame, contribution_df: pd.DataFrame, alignment_summary: pd.DataFrame, out_path: Path) -> None:
    contrib_rows, lopo_rows = [], []
    full_val = np.nan
    if len(contribution_df):
        d = contribution_df[contribution_df.get("component", pd.Series(dtype=float)).astype(float).eq(1)].copy()
        col = _alignment_value_col(d); lo_col, hi_col = _ci_cols(d)
        if col:
            order = {"A": 1, "E": 2, "M": 3, "P": 4, "T": 5}
            for _, r in d.iterrows():
                name = str(r.get("position_analysis", ""))
                if name == "full_position_concat":
                    full_val = r.get(col, np.nan); continue
                if name.startswith("only_"):
                    pos = name.replace("only_", "")
                    contrib_rows.append({"label": f"Only {pos}", "value": r.get(col, np.nan),
                                         "lo": r.get(lo_col, np.nan) if lo_col else np.nan,
                                         "hi": r.get(hi_col, np.nan) if hi_col else np.nan,
                                         "order": order.get(pos, 99)})
    if len(lopo_df):
        d = lopo_df[lopo_df.get("component", pd.Series(dtype=float)).astype(float).eq(1)].copy()
        col = _alignment_value_col(d); lo_col, hi_col = _ci_cols(d)
        if col and "left_out_position" in d.columns:
            order = {"A": 1, "E": 2, "M": 3, "P": 4, "T": 5}
            for _, r in d.iterrows():
                pos = str(r.get("left_out_position", "")).upper()
                lopo_rows.append({"label": f"Leave {pos}", "value": r.get(col, np.nan),
                                  "lo": r.get(lo_col, np.nan) if lo_col else np.nan,
                                  "hi": r.get(hi_col, np.nan) if hi_col else np.nan,
                                  "order": order.get(pos, 99)})
    left = pd.DataFrame(contrib_rows).sort_values("order") if contrib_rows else pd.DataFrame()
    right = pd.DataFrame(lopo_rows).sort_values("order") if lopo_rows else pd.DataFrame()
    if len(left) == 0 and len(right) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), sharex=True)
    for ax, df, color, label in [(axes[0], left, "#0072B2", "Only one site"), (axes[1], right, "#D55E00", "Leave one site")]:
        if len(df) == 0:
            ax.axis("off"); continue
        df = df[np.isfinite(df["value"])].reset_index(drop=True)
        y = np.arange(len(df))
        for yi, (_, r) in zip(y, df.iterrows()):
            if np.isfinite(r["lo"]) and np.isfinite(r["hi"]):
                ax.plot([r["lo"], r["hi"]], [yi, yi], color=color, lw=2.6)
                ax.plot([r["lo"], r["lo"]], [yi - 0.08, yi + 0.08], color=color, lw=1.4)
                ax.plot([r["hi"], r["hi"]], [yi - 0.08, yi + 0.08], color=color, lw=1.4)
            ax.scatter([r["value"]], [yi], color=color, edgecolor="black", linewidth=0.8, s=105, zorder=3)
        if np.isfinite(full_val):
            ax.axvline(full_val, color="black", linestyle="--", lw=1.5, label="Full")
        ax.axvline(0, color="black", lw=1.0, alpha=0.75)
        ax.set_yticks(y); ax.set_yticklabels(df["label"]); ax.invert_yaxis()
        # ax.text(0.02, 0.98, label, transform=ax.transAxes, ha="left", va="top", fontsize=21)
        ax.legend(frameon=False, loc="upper left", fontsize=18)
    axes[0].set_xlabel("OOF Spearman correlation")
    axes[1].set_xlabel("OOF Spearman correlation")
    savefig(out_path)


def plot_endpoint_roc(pred_df: pd.DataFrame, summary_df: pd.DataFrame, endpoint: str, pretty: str, out_path: Path, args) -> None:
    if len(pred_df) == 0:
        return
    d = pred_df[pred_df["endpoint"].astype(str).eq(endpoint)].copy()
    if len(d) == 0:
        log(f"Skip ROC for {endpoint}: no predictions"); return
    if "n_axis_features" not in d.columns:
        d["n_axis_features"] = 1
    d = d[d["n_axis_features"].astype(int).eq(1)].copy()
    if len(d) == 0 or d["y_true"].nunique() < 2:
        log(f"Skip ROC for {endpoint}: no valid Axis 1 predictions"); return
    summary = summary_df.copy() if len(summary_df) else pd.DataFrame()
    if len(summary) and "n_axis_features" not in summary.columns:
        summary["n_axis_features"] = 1
    y = d["y_true"].astype(int).to_numpy()
    p = d["y_prob"].astype(float).to_numpy()
    fpr, tpr, lo, hi = roc_curve_with_ci(y, p, args.n_bootstrap, args.seed + 9001)
    auc = roc_auc_score(y, p)
    row = summary[(summary["endpoint"].astype(str) == endpoint) & (summary["n_axis_features"].astype(int) == 1)] if len(summary) else pd.DataFrame()
    auc_lo = row["auroc_ci95_low"].iloc[0] if len(row) else np.nan
    auc_hi = row["auroc_ci95_high"].iloc[0] if len(row) else np.nan
    color = endpoint_color(endpoint)
    plt.figure(figsize=(6.6, 6.1))
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", lw=1.2, label="_nolegend_")
    if np.isfinite(lo).any():
        ax.fill_between(fpr, lo, hi, color=color, alpha=0.18)
    lab = f"AUROC={auc:.3f}" + (f" ({auc_lo:.3f}–{auc_hi:.3f})" if np.isfinite(auc_lo) and np.isfinite(auc_hi) else "")
    ax.plot(fpr, tpr, lw=3.0, color=color, label=lab)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.text(0.04, 0.96, endpoint_panel_text(endpoint, pretty), transform=ax.transAxes,
            ha="left", va="top", fontsize=21,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.25", alpha=0.92))
    ax.legend(frameon=False, loc="lower right", fontsize=18)
    savefig(out_path)



# =============================================================================
# Final requested overrides: retrieval line plots, ROC legend, endpoint score plot
# =============================================================================

def parse_window_length(window_setting) -> float:
    """Extract window length from labels such as 3_3, 4_4, 5_5."""
    s = str(window_setting)
    first = s.split("_")[0]
    try:
        return float(first)
    except Exception:
        try:
            return float(s)
        except Exception:
            return np.nan


def plot_retrieval_map_summary(retrieval_df: pd.DataFrame, metric_col: str, out_path: Path) -> None:
    """mAP across window length: one point-line curve for each representation.

    x-axis is the window length; each route is shown as a separate line. If repeated
    rows exist for the same route/window length, mean±SD is shown; otherwise points
    are connected without visible SD.
    """
    if len(retrieval_df) == 0 or metric_col not in retrieval_df.columns:
        return
    d = retrieval_df.copy()
    if not {"route_name", "window_setting"}.issubset(d.columns):
        return

    d["route_key"] = d["route_name"].astype(str).str.lower()
    d["route_label"] = d["route_key"].map(ROUTE_PRETTY).fillna(d["route_name"].astype(str))
    d["window_length"] = d["window_setting"].apply(parse_window_length)
    d[metric_col] = pd.to_numeric(d[metric_col], errors="coerce")
    d = d[d["route_key"].isin(ROUTE_ORDER) & np.isfinite(d["window_length"]) & np.isfinite(d[metric_col])].copy()
    if len(d) == 0:
        return

    summary = (
        d.groupby(["route_key", "route_label", "window_length"], as_index=False)
        .agg(mean_map=(metric_col, "mean"), sd_map=(metric_col, "std"), n=(metric_col, "size"))
    )
    summary["sd_map"] = summary["sd_map"].fillna(0.0)

    x_ticks = sorted(summary["window_length"].unique())
    plt.figure(figsize=(8.2, 5.8))
    ax = plt.gca()

    for route in ROUTE_ORDER:
        sub = summary[summary["route_key"].eq(route)].sort_values("window_length")
        if len(sub) == 0:
            continue
        lab = ROUTE_PRETTY.get(route, route)
        color = ROUTE_COLORS.get(lab, "#0072B2")
        yerr = sub["sd_map"].to_numpy(float)
        ax.errorbar(
            sub["window_length"], sub["mean_map"], yerr=yerr,
            marker="o", markersize=9.0, lw=2.8, capsize=5,
            color=color, label=lab
        )

    ax.set_xticks(x_ticks)
    ax.set_xlabel("Window length (s)")
    ax.set_ylabel("mAP")
    vals = summary["mean_map"].to_numpy(float)
    ymin = max(0.0, float(np.nanmin(vals)) - 0.08 * max(1e-6, float(np.nanmax(vals) - np.nanmin(vals))))
    ymax = min(1.02, float(np.nanmax(vals)) + 0.14 * max(1e-6, float(np.nanmax(vals) - np.nanmin(vals))))
    if metric_col == "single_to_four_mAP":
        ymin = max(0.0, float(np.nanmin(vals)) - 0.015)
        ymax = min(0.25, float(np.nanmax(vals)) + 0.025)
    ax.set_ylim(ymin, ymax)
    ax.legend(frameon=False, loc="best", fontsize=18)
    savefig(out_path)


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
    auc_lo = row["auroc_ci95_low"].iloc[0] if len(row) else np.nan
    auc_hi = row["auroc_ci95_high"].iloc[0] if len(row) else np.nan

    color = endpoint_color(endpoint)
    plt.figure(figsize=(6.6, 6.1))
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", lw=1.2, label="_nolegend_")
    if np.isfinite(lo).any():
        ax.fill_between(fpr, lo, hi, color=color, alpha=0.18)
    lab = f"AUROC={auc:.3f}"
    if np.isfinite(auc_lo) and np.isfinite(auc_hi):
        lab += f" ({auc_lo:.3f}–{auc_hi:.3f})"
    ax.plot(fpr, tpr, lw=3.0, color=color, label=lab)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.text(
        0.04, 0.96, endpoint_panel_text(endpoint, pretty),
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=21,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.25", alpha=0.92),
    )
    ax.legend(frameon=False, loc="lower right", fontsize=21)
    savefig(out_path)


def plot_figure18_threshold_clinical_trend(patient_long: pd.DataFrame, out_path: Path, component: int = 1) -> None:
    if len(patient_long) == 0:
        return
    required = {"panel", "adjustment", "component", "axis_group", "variable", "value"}
    if not required.issubset(patient_long.columns):
        return
    targets = [
        ("NYHA", "NYHA ≥3", 3, "#009E73"),
        ("MR_grade", "MR grade ≥2", 2, "#0072B2"),
        ("TR_grade", "TR grade ≥2", 2, "#56B4E9"),
        ("AR_grade", "AR grade ≥2", 2, "#CC79A7"),
    ]
    levels = ["Q1", "Q2", "Q3", "Q4"]
    d = patient_long[
        patient_long["panel"].astype(str).eq("all_clinical")
        & patient_long["adjustment"].astype(str).eq("none")
        & patient_long["component"].astype(int).eq(int(component))
    ].copy()
    plt.figure(figsize=(8.6, 6.2))
    ax = plt.gca()
    x_pos = np.arange(len(levels))
    all_ps = []
    for var, label, thr, color in targets:
        sub = d[d["variable"].astype(str).eq(var)].copy()
        if len(sub) == 0:
            continue
        ps, los, his = [], [], []
        for g in levels:
            vals = pd.to_numeric(sub.loc[sub["axis_group"].astype(str).eq(g), "value"], errors="coerce").dropna().to_numpy(float)
            k = int(np.sum(vals >= thr))
            p, lo, hi = binomial_ci_normal(k, len(vals))
            ps.append(p); los.append(lo); his.append(hi)
        ps = np.asarray(ps, dtype=float)
        all_ps.extend(ps[np.isfinite(ps)].tolist())
        yerr = np.vstack([ps - np.asarray(los), np.asarray(his) - ps])
        ax.errorbar(x_pos, ps, yerr=yerr, marker="o", markersize=8.5, lw=2.4,
                    capsize=5, color=color, label=label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(levels)
    if all_ps:
        ymax = min(1.0, max(all_ps) + 0.12)
        ax.set_ylim(0, max(0.20, ymax))
    else:
        ax.set_ylim(0, 1)
    ax.set_xlabel("Acoustic CCA axis 1 quartile")
    ax.set_ylabel("Proportion")
    ax.legend(frameon=False, loc="best", fontsize=18)
    savefig(out_path)


def plot_figure19_endpoint_score_distribution(axis_ranked: pd.DataFrame, axis_scores: pd.DataFrame, out_path: Path) -> None:
    if len(axis_ranked):
        d = axis_ranked.copy()
    else:
        d = get_all_clinical_axis_scores(axis_scores).copy()
    if len(d) == 0 or "cca_acoustic_axis1" not in d.columns:
        return

    endpoints = [
        ("EF_lt_40", "EF <40%", "#009E73"),
        ("NTproBNP_ge_900", "NT-proBNP ≥900", "#CC79A7"),
        ("LVEDD_dilated", "LVEDD dilated", "#0072B2"),
        ("NYHA_ge_3", "NYHA ≥3", "#56B4E9"),
    ]
    rows = []
    for col, label, color in endpoints:
        if col not in d.columns:
            continue
        sub = d[[col, "cca_acoustic_axis1"]].copy()
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub[sub[col].isin([0, 1]) & np.isfinite(sub["cca_acoustic_axis1"])]
        if len(sub) == 0:
            continue
        sub["endpoint"] = label
        sub["status"] = np.where(sub[col].astype(int).eq(1), "Positive", "Negative")
        rows.append(sub[["endpoint", "status", "cca_acoustic_axis1"]])
    if not rows:
        return

    long = pd.concat(rows, ignore_index=True)
    ordered = [label for _, label, _ in endpoints if label in set(long["endpoint"])]
    neg_color = "#3DDAD7"  # cyan-green
    pos_color = "#FFA94D"  # orange
    palette = {"Negative": neg_color, "Positive": pos_color}

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.boxplot(
            data=long, y="endpoint", x="cca_acoustic_axis1", hue="status",
            order=ordered, hue_order=["Negative", "Positive"],
            palette=palette, showfliers=False, linewidth=1.25, ax=ax
        )
        sns.stripplot(
            data=long, y="endpoint", x="cca_acoustic_axis1", hue="status",
            order=ordered, hue_order=["Negative", "Positive"],
            palette=palette, dodge=True, alpha=0.36, size=4.0, ax=ax
        )
        handles, labels = ax.get_legend_handles_labels()
        # Keep first pair from boxplot handles.
        ax.legend(handles[:2], labels[:2], frameon=False, fontsize=18, loc="upper right")
    ax.axvline(0, color="black", lw=1.0, alpha=0.75)
    vals = long["cca_acoustic_axis1"].to_numpy(float)
    if np.isfinite(vals).any():
        xmin, xmax = np.nanmin(vals), np.nanmax(vals)
        span = max(xmax - xmin, 1e-6)
        ax.set_xlim(xmin - 0.05 * span, xmax + 0.28 * span)
    ax.set_xlabel("Acoustic CCA axis 1 score")
    ax.set_ylabel("")
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
    position_endpoint_summary = read_csv_optional(table_dir / "position_endpoint_validation_summary.csv")

    # Robustness / supplement tables.
    lopo_summary = read_csv_optional(table_dir / "leave_one_position_out_alignment_summary.csv")
    position_contribution = read_csv_optional(table_dir / "position_contribution_alignment_summary.csv")
    repeated_cca_values = read_csv_optional(table_dir / "repeated_cv_cca_alignment_values.csv")
    negative_controls = read_csv_optional(table_dir / "negative_controls.csv")
    negative_control_summary = read_csv_optional(table_dir / "negative_control_summary.csv")
    confounder_summary = read_csv_optional(table_dir / "confounder_adjusted_alignment_summary.csv")
    hyper_endpoint = read_csv_optional(table_dir / "model_hyperparameter_sensitivity_endpoint.csv")
    threshold_summary = read_csv_optional(table_dir / "endpoint_threshold_sensitivity_summary.csv")
    repeated_endpoint_values = read_csv_optional(table_dir / "repeated_cv_endpoint_values.csv")
    retrieval_csv = resolve_external_csv(getattr(args, "retrieval_csv", None), "mc_retrieval_summary_by_window.csv")
    descriptor_csv = resolve_external_csv(getattr(args, "descriptor_csv", None), "patient_acoustic_descriptor_profile.csv")
    retrieval_df = read_csv_optional(retrieval_csv) if retrieval_csv else pd.DataFrame()
    descriptor_df = read_csv_optional(descriptor_csv) if descriptor_csv else pd.DataFrame()

    if len(axis_scores):
        plot_figure01_axis1_scatter_line(
            axis_scores,
            alignment_summary,
            fig_dir / "figure_01_allclinical_axis1_acoustic_vs_clinical_scatter_line.png",
        )

    if len(retrieval_df):
        plot_figure15_five_view_retrieval(
            retrieval_df,
            fig_dir / "figure_15_five_view_retrieval_map_by_representation.png",
        )
        plot_figure16_single_to_four_retrieval(
            retrieval_df,
            fig_dir / "figure_16_single_to_four_retrieval_map_by_representation.png",
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
        plot_figure17_continuous_clinical_trend(
            gradient_patient,
            fig_dir / "figure_17_continuous_clinical_variables_across_axis_quantiles.png",
            component=1,
        )
        plot_figure18_threshold_clinical_trend(
            gradient_patient,
            fig_dir / "figure_18_threshold_clinical_variables_across_axis_quantiles.png",
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

    # Figure 11: ranked acoustic CCA axis 1 score.
    plot_figure11_ranked_axis_score(
        axis_ranked,
        axis_scores,
        fig_dir / "figure_11_allclinical_axis1_ranked_score_plot.png",
    )
    plot_figure19_endpoint_score_distribution(
        axis_ranked,
        axis_scores,
        fig_dir / "figure_19_endpoint_score_distribution_by_acoustic_axis1.png",
    )

    if len(descriptor_df):
        plot_figureS08_acoustic_descriptor_associations(
            descriptor_df,
            axis_ranked,
            axis_scores,
            fig_dir / "figure_S08_acoustic_descriptor_associations_lollipop.png",
            n_boot=getattr(args, "n_bootstrap", 1000),
            seed=getattr(args, "seed", 42),
        )
        plot_figureS09_acoustic_descriptor_quantile_trend(
            descriptor_df,
            axis_ranked,
            axis_scores,
            fig_dir / "figure_S09_acoustic_descriptor_across_axis_quantiles.png",
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
    plot_figureS06_repeated_endpoint_auroc_by_split(
        repeated_endpoint_values,
        fig_dir / "figure_S06_repeated_cv_endpoint_auroc_by_split.png",
    )
    plot_figureS07_pairwise_axis_distance(
        axis_scores,
        fig_dir / "figure_S07_pairwise_acoustic_clinical_cca_axis1_distance.png",
    )
    plot_figureS12_endpoint_auroc_by_site_configuration_axis1(
        position_endpoint_summary,
        fig_dir / "figure_S12_endpoint_auroc_by_site_configuration_axis1_forest.png",
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Regenerate focused all-clinical CCA figures from result tables")
    p.add_argument("--result-dir", type=str, default="Outputs/alignment/CCA/beats/4_5_4_1", help="Output root containing tables/ and figures/. Ignored for table/fig dirs if those are explicitly given.")
    p.add_argument("--table-dir", type=str, default=None, help="Directory containing CSV result tables")
    p.add_argument("--fig-dir", type=str, default=None, help="Directory where figures will be saved")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--retrieval-csv", type=str, default="Outputs/representation/select_main_representation/mc_retrieval_summary_by_window.csv")
    p.add_argument("--descriptor-csv", type=str, default="Outputs/alignment/paper_figures_tables/cache/patient_acoustic_descriptor_profile.csv")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    setup_plotting()
    result_dir = Path(args.result_dir)
    table_dir = Path(args.table_dir) if args.table_dir else result_dir / "tables"
    fig_dir = Path(args.fig_dir) if args.fig_dir else result_dir / "figures"
    log(f"Plotting from table_dir={table_dir}")
    log(f"Saving figures to fig_dir={fig_dir}")
    plot_all_results(table_dir, fig_dir, args)
    log("Done plotting.")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import constants

    min_windows_per_position = constants.MIN_WINDOWN_PER_POSTISIONS
    patient_pass_min_positions = constants.PATIENT_PASS_MIN_POSTISIONS
    window_sec = constants.WINDOW_SEC
    stride_sec = constants.STRIDE_SEC

    route = "beats"

    result_dir = (
        constants.OUTPUT_FOLDER / "alignment" / "CCA" / route /
        f"{min_windows_per_position}_{patient_pass_min_positions}_{window_sec}_{stride_sec}"
    )

    retrieval_csv = (
        constants.OUTPUT_FOLDER / "representation" / "Selection" /
        "mc_retrieval_summary_by_window.csv"
    )

    descriptor_csv = (
            constants.OUTPUT_FOLDER / "alignment" / "acoustic_descriptors" /
            "patient_acoustic_descriptor_profile.csv"
    )

    main_args = [
        "--result-dir", str(result_dir),
        "--table-dir", str(result_dir / "tables"),
        "--fig-dir", str(result_dir / "figures"),
        "--retrieval-csv", str(retrieval_csv),
        "--descriptor-csv", str(descriptor_csv),
        "--seed", "42",
        "--n-bootstrap", "1000",
    ]

    main(main_args)
