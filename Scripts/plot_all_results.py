#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate publication-style tables and figures for
"Clinical alignment of heart sound representations".

Run from the project root, for example:

python plot_all_results.py \
  --prepared-dir Clinical_alignment/outputs/prepared \
  --tables-dir Clinical_alignment/outputs/clinically_anchored_acoustic_phenotyping_clean_v9/tables \
  --retrieval-dir Representation_learning/select_main_representation_with_ead \
  --window-library-dir Data_preprocessing/window_lib_4_1/window_library \
  --out-dir Clinical_alignment/outputs/paper_figures_tables

Main outputs:
  tables/clinical_alignment_tables.xlsx
  figures/Figure2_acoustic_representation_selection.png
  figures/Figure3_cross_validated_cca_alignment.png
  figures/Figure4_clinical_axis_profile.png
  figures/Figure5_endpoint_validation.png
  figures/Figure6_robustness_summary.png
  figures/FigureS1_acoustic_profile.png

The script is deliberately defensive: when a specific robustness/CCA table is
not found, it saves the rest of the figures and writes a clear message in the
missing panel rather than failing midway.
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.gridspec import GridSpec

try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:  # pragma: no cover
    sns = None
    _HAS_SEABORN = False

try:
    from scipy.stats import spearmanr, mannwhitneyu
except Exception:  # pragma: no cover
    spearmanr = None
    mannwhitneyu = None

try:
    from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
except Exception:  # pragma: no cover
    roc_curve = None
    roc_auc_score = None
    average_precision_score = None


# -----------------------------
# Configuration
# -----------------------------

POSITION_ORDER = ["A", "E", "M", "P", "T"]

DISPLAY_NAME = {
    "age_years": "Age, years",
    "sex_male": "Male sex",
    "BMI": "BMI, kg/m²",
    "heart_rate": "Heart rate, bpm",
    "NYHA": "NYHA class",
    "EF_Teich": "EF, %",
    "NTproBNP": "NT-proBNP",
    "hsTnT": "hsTnT",
    "CRP": "CRP",
    "D_dimer": "D-dimer",
    "Hb": "Hemoglobin",
    "LA_mm": "LA, mm",
    "LVEDD_mm": "LVEDD, mm",
    "IVS_mm": "IVS, mm",
    "LVPW_mm": "LVPW, mm",
    "AR_grade": "AR grade",
    "AS_grade": "AS grade",
    "MS_grade": "MS grade",
    "MR_grade": "MR grade",
    "TR_grade": "TR grade",
    "PR_grade": "PR grade",
    "duration_total": "Total recording duration, s",
    "n_windows_total": "Total windows per patient",
}

DOMAIN_MAP = {
    "age_years": "Demographics",
    "sex_male": "Demographics",
    "BMI": "Demographics",
    "heart_rate": "Vital signs",
    "NYHA": "Function / burden",
    "EF_Teich": "Function / burden",
    "NTproBNP": "Function / burden",
    "hsTnT": "Function / burden",
    "CRP": "Function / burden",
    "D_dimer": "Function / burden",
    "Hb": "Function / burden",
    "LA_mm": "Structure",
    "LVEDD_mm": "Structure",
    "IVS_mm": "Structure",
    "LVPW_mm": "Structure",
    "AR_grade": "Valve status",
    "AS_grade": "Valve status",
    "MS_grade": "Valve status",
    "MR_grade": "Valve status",
    "TR_grade": "Valve status",
    "PR_grade": "Valve status",
}

TABLE1_VARIABLES = [
    "age_years", "sex_male", "heart_rate", "NYHA",
    "EF_Teich", "NTproBNP", "hsTnT",
    "LA_mm", "LVEDD_mm", "IVS_mm", "LVPW_mm",
    "MR_grade", "TR_grade", "AR_grade", "AS_grade", "MS_grade", "PR_grade",
    "CRP", "D_dimer", "Hb",
    "duration_A", "duration_E", "duration_M", "duration_P", "duration_T",
    "duration_total", "n_windows_A", "n_windows_E", "n_windows_M", "n_windows_P", "n_windows_T",
    "n_windows_total",
]

PROFILE_CONTINUOUS = ["EF_Teich", "NTproBNP", "LA_mm", "LVEDD_mm"]
PROFILE_BINARY = {
    "NYHA ≥3": ("NYHA", ">=", 3),
    "MR grade ≥2": ("MR_grade", ">=", 2),
    "TR grade ≥2": ("TR_grade", ">=", 2),
    "AR grade ≥2": ("AR_grade", ">=", 2),
}

ENDPOINT_DEFINITIONS = {
    "EF <40%": {
        "variable": "EF_Teich",
        "removed_variable": "EF_Teich",
        "rule": lambda df: pd.to_numeric(df["EF_Teich"], errors="coerce") < 40,
    },
    "NT-proBNP ≥900": {
        "variable": "NTproBNP",
        "removed_variable": "NTproBNP",
        "rule": lambda df: pd.to_numeric(df["NTproBNP"], errors="coerce") >= 900,
    },
    "LVEDD dilation": {
        "variable": "LVEDD_mm",
        "removed_variable": "LVEDD_mm",
        # Fallback threshold for visualization only when no endpoint-prediction
        # table is found. Replace with the exact endpoint labels if available.
        "rule": lambda df: _lvdd_dilation(df),
    },
    "NYHA ≥3": {
        "variable": "NYHA",
        "removed_variable": "NYHA",
        "rule": lambda df: pd.to_numeric(df["NYHA"], errors="coerce") >= 3,
    },
}

DOMAIN_PALETTE = {
    "Demographics": "#7f7f7f",
    "Vital signs": "#8c564b",
    "Function / burden": "#1f77b4",
    "Structure": "#2ca02c",
    "Valve status": "#ff7f0e",
    "Technical": "#9467bd",
    "Other": "#666666",
}

ROUTE_DISPLAY = {
    "ead": "EAD",
    "EAD": "EAD",
    "panns": "PANNs",
    "PANNs": "PANNs",
    "ast": "AST",
    "AST": "AST",
    "beats": "BEATs",
    "BEATs": "BEATs",
}

ROUTE_ORDER = ["EAD", "PANNs", "AST", "BEATs"]


@dataclass
class OutputPaths:
    out_dir: Path
    fig_dir: Path
    table_dir: Path
    cache_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manuscript figures and tables.")
    parser.add_argument("--prepared-dir", type=Path, default=Path("Clinical_alignment/outputs/prepared"))
    parser.add_argument("--tables-dir", type=Path, default=Path("Clinical_alignment/outputs/clinically_anchored_acoustic_phenotyping_clean_v9/tables"))
    parser.add_argument("--retrieval-dir", type=Path, default=Path("Representation_learning/select_main_representation_with_ead"))
    parser.add_argument("--window-library-dir", type=Path, default=Path("Data_preprocessing/window_lib_4_1/window_library"))
    parser.add_argument("--out-dir", type=Path, default=Path("Clinical_alignment/outputs/paper_figures_tables"))
    parser.add_argument("--fs", type=float, default=8000.0, help="Sampling rate of window waveforms.")
    parser.add_argument("--max-windows-per-position", type=int, default=40, help="Max windows sampled per patient-position for acoustic feature profiling.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-acoustic-profile", action="store_true", help="Skip waveform-level acoustic feature extraction.")
    return parser.parse_args()


# -----------------------------
# General utilities
# -----------------------------

def setup_style() -> None:
    if _HAS_SEABORN:
        sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "axes.linewidth": 1.0,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def ensure_outputs(out_dir: Path) -> OutputPaths:
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    cache_dir = out_dir / "cache"
    for p in [out_dir, fig_dir, table_dir, cache_dir]:
        p.mkdir(parents=True, exist_ok=True)
    return OutputPaths(out_dir=out_dir, fig_dir=fig_dir, table_dir=table_dir, cache_dir=cache_dir)


def strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def read_csv_safely(path: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    return strip_bom_columns(df)


def read_table_safely(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() in [".csv", ".txt"]:
            return read_csv_safely(path)
        if path.suffix.lower() == ".tsv":
            return read_csv_safely(path, sep="\t")
        if path.suffix.lower() in [".xlsx", ".xls"]:
            return strip_bom_columns(pd.read_excel(path))
    except Exception as e:
        warnings.warn(f"Could not read {path}: {e}")
    return None


def resolve_file(base_dir: Path, filename: str, fallback_dir: Optional[Path] = None) -> Path:
    p = base_dir / filename
    if p.exists():
        return p
    if fallback_dir is not None and (fallback_dir / filename).exists():
        return fallback_dir / filename
    return p


def load_all_tables(tables_dir: Path) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    if not tables_dir.exists():
        warnings.warn(f"Tables directory not found: {tables_dir}")
        return tables
    for p in sorted(tables_dir.glob("*")):
        if p.suffix.lower() not in [".csv", ".tsv", ".xlsx", ".xls"]:
            continue
        df = read_table_safely(p)
        if df is not None and len(df.columns) > 0:
            tables[p.name] = df
    return tables


def norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")


def has_all(text: str, words: Sequence[str]) -> bool:
    return all(w in text for w in words)


def has_any(text: str, words: Sequence[str]) -> bool:
    return any(w in text for w in words)


def find_column(
    df: pd.DataFrame,
    exact: Sequence[str] = (),
    contains_all: Sequence[str] = (),
    contains_any: Sequence[str] = (),
    excludes: Sequence[str] = (),
) -> Optional[str]:
    norm_map = {c: norm_col(c) for c in df.columns}
    exact_norm = {norm_col(x) for x in exact}
    for c, n in norm_map.items():
        if n in exact_norm:
            return c
    candidates = []
    for c, n in norm_map.items():
        if contains_all and not has_all(n, contains_all):
            continue
        if contains_any and not has_any(n, contains_any):
            continue
        if excludes and has_any(n, excludes):
            continue
        candidates.append(c)
    return candidates[0] if candidates else None


def as_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def fmt_num(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


def fmt_ci(value: float, low: Optional[float] = None, high: Optional[float] = None, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ({low:.{digits}f}, {high:.{digits}f})"


def median_iqr(s: pd.Series, digits: int = 1) -> str:
    x = as_numeric(s).dropna()
    if x.empty:
        return "NA"
    q1, med, q3 = np.nanpercentile(x, [25, 50, 75])
    return f"{med:.{digits}f} ({q1:.{digits}f}, {q3:.{digits}f})"


def mean_sd(s: pd.Series, digits: int = 1) -> str:
    x = as_numeric(s).dropna()
    if x.empty:
        return "NA"
    return f"{x.mean():.{digits}f} ± {x.std(ddof=1):.{digits}f}"


def n_percent(mask: pd.Series, denominator: Optional[int] = None, digits: int = 1) -> str:
    m = mask.dropna().astype(bool)
    if denominator is None:
        denominator = len(m)
    if denominator == 0:
        return "NA"
    n = int(m.sum())
    return f"{n} ({100 * n / denominator:.{digits}f}%)"


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return np.nan, np.nan
    if spearmanr is not None:
        r, p = spearmanr(x[valid], y[valid])
        return float(r), float(p)
    r = x[valid].rank().corr(y[valid].rank())
    return float(r), np.nan


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    temp = np.empty_like(q)
    temp[order] = q
    out[valid] = temp
    return out


def bootstrap_ci_proportion(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denom
    return max(0, centre - half), min(1, centre + half)


def add_panel_label(ax, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top", ha="left")


def clean_axes(ax, grid: bool = False) -> None:
    ax.grid(grid, alpha=0.25)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)


def empty_panel(ax, title: str, message: str) -> None:
    ax.set_title(title, loc="left", fontweight="bold")
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=11, color="#555555")
    ax.set_xticks([])
    ax.set_yticks([])
    clean_axes(ax)


def save_figure(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[done] Saved figure: {path}")
    plt.close(fig)


# -----------------------------
# Table generation
# -----------------------------

def make_table1(clinical_tech: pd.DataFrame) -> pd.DataFrame:
    df = clinical_tech.copy()
    rows = []
    n_total = len(df)

    # cohort-level records first
    rows.append({
        "Domain": "Cohort",
        "Variable": "Patients included",
        "Summary": f"{n_total}",
        "Missing": "0",
        "Coverage": "100.0%",
        "Notes": "Patients with paired clinical data and technical recording metadata.",
    })
    if all(f"duration_{p}" in df.columns for p in POSITION_ORDER):
        complete_five = df[[f"duration_{p}" for p in POSITION_ORDER]].notna().all(axis=1)
        rows.append({
            "Domain": "Recording coverage",
            "Variable": "Complete five-site recording coverage",
            "Summary": n_percent(complete_five, len(df)),
            "Missing": str(int(complete_five.isna().sum())),
            "Coverage": f"{100 * complete_five.notna().mean():.1f}%",
            "Notes": "Availability of A/E/M/P/T recordings after quality control.",
        })

    for col in TABLE1_VARIABLES:
        if col not in df.columns:
            continue
        domain = DOMAIN_MAP.get(col, "Recording quality" if col.startswith(("duration", "n_windows")) else "Other")
        display = DISPLAY_NAME.get(col, col)
        s = df[col]
        missing = int(s.isna().sum())
        coverage = f"{100 * s.notna().mean():.1f}%"
        note = ""
        if col == "sex_male":
            summary = n_percent(as_numeric(s) == 1, int(s.notna().sum()))
            note = "Number and percentage of male patients among non-missing records."
        elif col.endswith("_grade"):
            x = as_numeric(s)
            summary = median_iqr(x, 0)
            ge2 = n_percent(x >= 2, int(x.notna().sum()))
            note = f"Ordinal valve grade; moderate-or-greater (grade ≥2): {ge2}."
        elif col == "NYHA":
            x = as_numeric(s)
            summary = median_iqr(x, 0)
            ge3 = n_percent(x >= 3, int(x.notna().sum()))
            note = f"Ordinal class; NYHA ≥3: {ge3}."
        elif col == "NTproBNP":
            summary = median_iqr(s, 1)
            note = "Reported as raw value; log-transformed for modeling when specified in the registry."
        elif col.startswith("n_windows"):
            summary = median_iqr(s, 0)
            note = "Number of fixed-length analysis windows."
        elif col.startswith("duration"):
            summary = median_iqr(s, 1)
            note = "Recording duration in seconds."
        else:
            summary = median_iqr(s, 1)
        rows.append({
            "Domain": domain,
            "Variable": display,
            "Summary": summary,
            "Missing": str(missing),
            "Coverage": coverage,
            "Notes": note,
        })
    return pd.DataFrame(rows)


def make_clinical_matrix_table(
    registry: Optional[pd.DataFrame],
    missingness: Optional[pd.DataFrame],
    clinical: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    if registry is not None and "clean_name" in registry.columns:
        reg = registry.copy()
    else:
        reg = pd.DataFrame({"clean_name": [c for c in clinical.columns if c != "patient_id"]})

    miss = missingness.copy() if missingness is not None else pd.DataFrame()
    miss_map = {}
    if not miss.empty and "clean_name" in miss.columns:
        for _, r in miss.iterrows():
            miss_map[str(r["clean_name"])] = r

    endpoint_removed = {v["removed_variable"] for v in ENDPOINT_DEFINITIONS.values()}

    for _, r in reg.iterrows():
        clean = str(r.get("clean_name", ""))
        if not clean or clean == "nan" or clean not in clinical.columns:
            continue
        m = miss_map.get(clean, None)
        n_rows = len(clinical)
        n_missing = int(clinical[clean].isna().sum())
        coverage = 1 - n_missing / max(n_rows, 1)
        if m is not None:
            n_missing = int(rget(m, "n_missing", n_missing))
            coverage = float(rget(m, "coverage", coverage))

        use_for_distance = parse_bool(r.get("use_for_distance", False))
        use_for_neighbor = parse_bool(r.get("use_for_neighbor", False))
        use_for_retrieval = parse_bool(r.get("use_for_retrieval", False))
        use_for_adjustment = parse_bool(r.get("use_for_adjustment", False))

        rows.append({
            "Clinical domain": str(r.get("group", DOMAIN_MAP.get(clean, "Other"))),
            "Variable": DISPLAY_NAME.get(clean, clean),
            "Clean name": clean,
            "Type": str(r.get("var_type", infer_var_type(clean, clinical[clean]))),
            "Transformation": str(r.get("transform", "none")),
            "Encoding": infer_encoding(clean, r, clinical[clean]),
            "Missing, n (%)": f"{n_missing} ({100 * n_missing / max(n_rows, 1):.1f}%)",
            "Coverage": f"{100 * coverage:.1f}%",
            "Included in CCA / clinical matrix": bool(use_for_distance or use_for_neighbor or (clean in clinical.columns and clean not in ["patient_id"])),
            "Used for retrieval/endpoint summaries": bool(use_for_retrieval),
            "Adjustment covariate": bool(use_for_adjustment),
            "Removed in leave-endpoint-out validation": clean in endpoint_removed,
            "Notes": str(r.get("notes", "")),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        order = ["demographic", "function", "burden", "structure", "valve", "symptom", "technical", "other"]
        out["_domain_order"] = out["Clinical domain"].astype(str).str.lower().apply(
            lambda x: next((i for i, k in enumerate(order) if k in x), 99)
        )
        out = out.sort_values(["_domain_order", "Clinical domain", "Variable"]).drop(columns="_domain_order")
    return out


def rget(row: pd.Series, key: str, default=None):
    try:
        val = row.get(key, default)
    except Exception:
        return default
    if pd.isna(val):
        return default
    return val


def parse_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    return str(x).strip().lower() in ["true", "1", "yes", "y"]


def infer_var_type(name: str, s: pd.Series) -> str:
    if name == "sex_male" or name.startswith("symptom_"):
        return "binary"
    if name.endswith("_grade") or name == "NYHA":
        return "ordinal"
    if pd.api.types.is_numeric_dtype(s):
        return "continuous"
    return "categorical"


def infer_encoding(name: str, registry_row: pd.Series, s: pd.Series) -> str:
    vtype = str(registry_row.get("var_type", infer_var_type(name, s))).lower()
    transform = str(registry_row.get("transform", "none"))
    if vtype == "continuous":
        return f"standardized continuous; transform={transform}"
    if vtype == "ordinal":
        return "integer-ordered ordinal variable"
    if vtype == "binary":
        return "0/1 indicator"
    return "categorical or manually encoded variable"


# -----------------------------
# Retrieval table and Figure 2
# -----------------------------

def load_retrieval_data(retrieval_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    across_path = retrieval_dir / "mc_retrieval_summary_across_windows.csv"
    by_path = retrieval_dir / "mc_retrieval_summary_by_window.csv"
    across = read_table_safely(across_path) if across_path.exists() else pd.DataFrame()
    bywin = read_table_safely(by_path) if by_path.exists() else pd.DataFrame()
    if across.empty:
        candidates = list(retrieval_dir.glob("*retrieval*across*.csv")) + list(retrieval_dir.glob("*summary*.csv"))
        for c in candidates:
            df = read_table_safely(c)
            if df is not None and not df.empty:
                across = df
                break
    if bywin.empty:
        candidates = list(retrieval_dir.glob("*retrieval*by*window*.csv")) + list(retrieval_dir.glob("*window*.csv"))
        for c in candidates:
            df = read_table_safely(c)
            if df is not None and not df.empty:
                bywin = df
                break
    return standardize_retrieval(across), standardize_retrieval(bywin)


def standardize_retrieval(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    route_col = find_column(df, exact=["route", "route_name", "model", "representation"])
    win_col = find_column(df, exact=["window_setting", "window", "window_length", "setting", "window_sec"])
    task_col = find_column(df, exact=["task", "retrieval_task", "protocol"])
    rank_col = find_column(df, contains_any=["rank1", "rank_1", "rank1_accuracy", "accuracy"], excludes=["top"])
    map_col = find_column(df, contains_any=["map", "mean_average_precision", "m_ap"])

    out = pd.DataFrame(index=df.index)
    out["route"] = df[route_col].map(lambda x: ROUTE_DISPLAY.get(str(x), str(x))) if route_col else "Unknown"
    if win_col:
        out["window_setting"] = df[win_col].map(normalize_window_label)
    else:
        out["window_setting"] = "All"
    if task_col:
        out["task"] = df[task_col].astype(str).map(normalize_task_label)
    else:
        # If no explicit task column, infer from metric column names later.
        out["task"] = "Overall"
    out["rank1"] = as_numeric(df[rank_col]) if rank_col else np.nan
    out["mAP"] = as_numeric(df[map_col]) if map_col else np.nan

    # Some files may store each task in separate metric columns.
    wide_rows = []
    metric_names = {norm_col(c): c for c in df.columns}
    for c in df.columns:
        lc = norm_col(c)
        metric = None
        if "rank" in lc and "five" in lc:
            metric = "rank1"
            task = "Five-view"
        elif "map" in lc and "five" in lc:
            metric = "mAP"
            task = "Five-view"
        elif "rank" in lc and ("single" in lc or "four" in lc or "cross" in lc):
            metric = "rank1"
            task = "Single-to-four"
        elif "map" in lc and ("single" in lc or "four" in lc or "cross" in lc):
            metric = "mAP"
            task = "Single-to-four"
        if metric is None:
            continue
        for i, val in enumerate(df[c]):
            row = {
                "route": out.loc[i, "route"] if i in out.index else "Unknown",
                "window_setting": out.loc[i, "window_setting"] if i in out.index else "All",
                "task": task,
                metric: pd.to_numeric(val, errors="coerce"),
            }
            wide_rows.append(row)
    if wide_rows:
        wide = pd.DataFrame(wide_rows)
        wide = wide.groupby(["route", "window_setting", "task"], as_index=False).first()
        for metric in ["rank1", "mAP"]:
            if metric not in wide.columns:
                wide[metric] = np.nan
        return wide

    return out.dropna(how="all", subset=["rank1", "mAP"])


def normalize_window_label(x) -> str:
    if pd.isna(x):
        return "All"
    s = str(x)
    nums = re.findall(r"\d+", s)
    if nums:
        return f"{nums[0]} s"
    return s


def normalize_task_label(x) -> str:
    s = str(x).lower()
    if "five" in s or "5" in s or "view" in s:
        return "Five-view"
    if "single" in s or "four" in s or "cross" in s or "1to4" in s:
        return "Single-to-four"
    return str(x)


def draw_route_pipeline(ax) -> None:
    ax.set_title("A  Shared window library and representation routes", loc="left", fontweight="bold")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    colors = ["#e8f2fb", "#eef7ec", "#fff4e6", "#f1ecf7"]
    labels = ["4-s heart-sound\nwindows", "EAD", "PANNs", "AST", "BEATs"]
    x_positions = [0.8, 3.2, 5.0, 6.8, 8.6]
    for i, (x, lab) in enumerate(zip(x_positions, labels)):
        w = 1.35 if i else 1.65
        color = "#f5f5f5" if i == 0 else colors[(i - 1) % len(colors)]
        box = FancyBboxPatch((x - w / 2, 3.4), w, 1.0, boxstyle="round,pad=0.03,rounding_size=0.08",
                             facecolor=color, edgecolor="#333333", linewidth=1)
        ax.add_patch(box)
        ax.text(x, 3.9, lab, ha="center", va="center", fontsize=10)
    for x in x_positions[1:]:
        ax.add_patch(FancyArrowPatch((1.65, 3.9), (x - 0.75, 3.9), arrowstyle="-|>", mutation_scale=9,
                                     linewidth=0.8, color="#555555", alpha=0.7))
    for j, p in enumerate(POSITION_ORDER):
        circ = Circle((2.1 + j * 0.55, 1.75), 0.18, facecolor="#ffffff", edgecolor="#444444")
        ax.add_patch(circ)
        ax.text(2.1 + j * 0.55, 1.75, p, ha="center", va="center", fontsize=9)
    ax.text(3.2, 2.35, "Position-level aggregation", ha="center", fontsize=10)
    ax.add_patch(FancyArrowPatch((3.2, 3.25), (3.2, 2.05), arrowstyle="-|>", mutation_scale=10, color="#555555"))
    box = FancyBboxPatch((4.7, 1.2), 2.1, 1.1, boxstyle="round,pad=0.04,rounding_size=0.08",
                         facecolor="#f7f7f7", edgecolor="#333333")
    ax.add_patch(box)
    ax.text(5.75, 1.75, "Patient-level\nA/E/M/P/T profile", ha="center", va="center", fontsize=10)
    ax.add_patch(FancyArrowPatch((3.65, 1.75), (4.65, 1.75), arrowstyle="-|>", mutation_scale=10, color="#555555"))


def draw_five_view_schematic(ax) -> None:
    ax.set_title("B  Five-view retrieval", loc="left", fontweight="bold")
    draw_retrieval_schematic(ax, query_positions=POSITION_ORDER, gallery_positions=POSITION_ORDER,
                             query_label="Query: one window from each site",
                             gallery_label="Gallery: independent five-site profile")


def draw_single_four_schematic(ax) -> None:
    ax.set_title("C  Single-to-four retrieval", loc="left", fontweight="bold")
    draw_retrieval_schematic(ax, query_positions=["A"], gallery_positions=["E", "M", "P", "T"],
                             query_label="Query: one site", gallery_label="Gallery: remaining four sites")


def draw_retrieval_schematic(ax, query_positions: List[str], gallery_positions: List[str], query_label: str, gallery_label: str) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for i, p in enumerate(query_positions):
        x = 1.0 + i * 0.55
        ax.add_patch(Circle((x, 2.5), 0.18, facecolor="#dbe9f6", edgecolor="#333333"))
        ax.text(x, 2.5, p, ha="center", va="center", fontsize=8)
    ax.text(1.4, 3.05, query_label, ha="left", fontsize=9)
    ax.add_patch(FancyArrowPatch((3.4, 2.5), (5.0, 2.5), arrowstyle="-|>", mutation_scale=12, color="#555555"))
    for i, p in enumerate(gallery_positions):
        x = 5.8 + i * 0.55
        ax.add_patch(Circle((x, 2.5), 0.18, facecolor="#e7f4df", edgecolor="#333333"))
        ax.text(x, 2.5, p, ha="center", va="center", fontsize=8)
    ax.text(5.1, 3.05, gallery_label, ha="left", fontsize=9)
    ax.text(4.2, 1.25, "Rank same-patient profile\namong all patients", ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fafafa", edgecolor="#cccccc"))


def plot_retrieval_metric(ax, bywin: pd.DataFrame, metric: str, title: str) -> None:
    ax.set_title(title, loc="left", fontweight="bold")
    if bywin.empty or metric not in bywin.columns or bywin[metric].notna().sum() == 0:
        empty_panel(ax, title, "Retrieval summary table not found\nor metric columns could not be parsed.")
        return
    data = bywin.copy()
    data = data[data["route"].isin(ROUTE_ORDER)].copy()
    if data.empty:
        data = bywin.copy()
    data["route"] = pd.Categorical(data["route"], categories=ROUTE_ORDER, ordered=True)
    win_order = sorted(data["window_setting"].dropna().unique(), key=lambda x: int(re.findall(r"\d+", str(x))[0]) if re.findall(r"\d+", str(x)) else 999)
    task_order = [t for t in ["Five-view", "Single-to-four", "Overall"] if t in set(data["task"])]
    palette = sns.color_palette("Set2", n_colors=max(len(ROUTE_ORDER), 4)) if _HAS_SEABORN else None
    for ti, task in enumerate(task_order):
        sub = data[data["task"] == task]
        for ri, route in enumerate(ROUTE_ORDER):
            ss = sub[sub["route"] == route]
            if ss.empty:
                continue
            xs = [win_order.index(w) + (ti - (len(task_order)-1)/2)*0.08 for w in ss["window_setting"]]
            color = palette[ri] if palette else None
            ax.plot(xs, ss[metric], marker="o", linewidth=1.5, label=f"{route} ({task})", color=color, alpha=0.9)
    ax.set_xticks(range(len(win_order)))
    ax.set_xticklabels(win_order)
    ax.set_ylabel(metric)
    ax.set_xlabel("Window length")
    ax.set_ylim(bottom=0)
    clean_axes(ax)
    if len(ax.get_legend_handles_labels()[0]) > 0:
        ax.legend(frameon=False, ncol=1, fontsize=8, loc="best")


def make_figure2(bywin: pd.DataFrame, out: OutputPaths, dpi: int) -> None:
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1.0, 1.2], hspace=0.45, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])
    draw_route_pipeline(ax_a)
    draw_five_view_schematic(ax_b)
    draw_single_four_schematic(ax_c)
    plot_retrieval_metric(ax_d, bywin, "rank1", "D  Rank-1 accuracy")
    plot_retrieval_metric(ax_e, bywin, "mAP", "E  Mean average precision")
    save_figure(fig, out.fig_dir / "Figure2_acoustic_representation_selection.png", dpi)


# -----------------------------
# CCA result discovery and Figure 3
# -----------------------------

def find_axis_scores(tables: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:
    best_name = None
    best_df = pd.DataFrame()
    best_score = -1
    for name, df in tables.items():
        cols_norm = [norm_col(c) for c in df.columns]
        if "patient_id" not in cols_norm:
            continue
        joined = " ".join(cols_norm)
        score = 0
        if "acoustic" in joined:
            score += 2
        if "clinical" in joined:
            score += 2
        if "axis" in joined or "cca" in joined:
            score += 2
        if "score" in joined:
            score += 1
        if any(k in norm_col(name) for k in ["axis", "score", "oof", "cca"]):
            score += 2
        if score > best_score:
            best_name, best_df, best_score = name, df, score
    if best_df.empty:
        return pd.DataFrame(), None, None
    acoustic_col, clinical_col = infer_axis_columns(best_df, axis=1)
    if acoustic_col is None or clinical_col is None:
        # try first two numeric columns after patient_id
        numeric_cols = [c for c in best_df.columns if c != "patient_id" and pd.api.types.is_numeric_dtype(pd.to_numeric(best_df[c], errors="coerce"))]
        if len(numeric_cols) >= 2:
            acoustic_col, clinical_col = numeric_cols[0], numeric_cols[1]
    print(f"[info] Axis-score table: {best_name}; acoustic={acoustic_col}; clinical={clinical_col}")
    return best_df.copy(), acoustic_col, clinical_col


def infer_axis_columns(df: pd.DataFrame, axis: int = 1) -> Tuple[Optional[str], Optional[str]]:
    cols = list(df.columns)
    axis_patterns = [f"axis_{axis}", f"axis{axis}", f"component_{axis}", f"comp{axis}", f"cca{axis}", f"cca_{axis}"]
    acoustic = []
    clinical = []
    for c in cols:
        n = norm_col(c)
        axis_match = any(p in n for p in axis_patterns) or (axis == 1 and any(p in n for p in ["u1", "x1", "score1", "cv1"]))
        if not axis_match and "score" not in n:
            continue
        if "acoustic" in n or n.startswith("x_") or "xscore" in n or "x_axis" in n or "u" == n[:1]:
            acoustic.append(c)
        if "clinical" in n or n.startswith("y_") or "yscore" in n or "y_axis" in n or "v" == n[:1]:
            clinical.append(c)
    if acoustic and clinical:
        return acoustic[0], clinical[0]
    # common naming alternatives
    acoustic = [c for c in cols if any(k in norm_col(c) for k in ["acoustic_cca_axis_1", "acoustic_axis_1", "x_cca_1", "u_1"])]
    clinical = [c for c in cols if any(k in norm_col(c) for k in ["clinical_cca_axis_1", "clinical_axis_1", "y_cca_1", "v_1"])]
    return (acoustic[0] if acoustic else None, clinical[0] if clinical else None)


def find_rho_table(
    tables: Dict[str, pd.DataFrame],
    name_keywords: Sequence[str],
    required_cols_any: Sequence[str] = ("rho", "spearman", "corr", "correlation"),
) -> pd.DataFrame:
    candidates = []
    for name, df in tables.items():
        lname = norm_col(name)
        if not any(k in lname for k in name_keywords):
            continue
        joined = " ".join(norm_col(c) for c in df.columns)
        if any(k in joined for k in required_cols_any):
            candidates.append((name, df))
    if candidates:
        print(f"[info] Using table for {name_keywords}: {candidates[0][0]}")
        return candidates[0][1].copy()
    return pd.DataFrame()


def find_corr_col(df: pd.DataFrame) -> Optional[str]:
    return find_column(df, contains_any=["rho", "spearman", "corr", "correlation"], excludes=["p_value", "pval", "p_"])


def compute_axis_comparison(axis_scores: pd.DataFrame) -> pd.DataFrame:
    if axis_scores.empty:
        return pd.DataFrame()
    rows = []
    for axis in range(1, 8):
        ac, cl = infer_axis_columns(axis_scores, axis=axis)
        if ac is None or cl is None:
            continue
        rho, p = spearman_corr(axis_scores[ac], axis_scores[cl])
        rows.append({"Axis": f"Axis {axis}", "rho": rho, "p_value": p, "acoustic_col": ac, "clinical_col": cl})
    return pd.DataFrame(rows)


def get_observed_rho(axis_scores: pd.DataFrame, acoustic_col: Optional[str], clinical_col: Optional[str]) -> float:
    if axis_scores.empty or acoustic_col is None or clinical_col is None:
        return np.nan
    rho, _ = spearman_corr(axis_scores[acoustic_col], axis_scores[clinical_col])
    return rho


def plot_axis_scatter(ax, axis_scores: pd.DataFrame, acoustic_col: Optional[str], clinical_col: Optional[str]) -> None:
    ax.set_title("A  Out-of-fold acoustic–clinical axis alignment", loc="left", fontweight="bold")
    if axis_scores.empty or acoustic_col is None or clinical_col is None:
        empty_panel(ax, "A  Out-of-fold acoustic–clinical axis alignment", "Axis-score table not found.")
        return
    x = as_numeric(axis_scores[acoustic_col])
    y = as_numeric(axis_scores[clinical_col])
    valid = x.notna() & y.notna()
    rho, p = spearman_corr(x, y)
    ax.scatter(x[valid], y[valid], s=18, alpha=0.35, edgecolor="none", color="#4C72B0")
    # Linear trend
    if valid.sum() > 3:
        coef = np.polyfit(x[valid], y[valid], 1)
        xs = np.linspace(x[valid].min(), x[valid].max(), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="#333333", linewidth=1.8)
        # Binned mean trajectory
        bins = pd.qcut(x[valid], q=10, duplicates="drop")
        bm = pd.DataFrame({"x": x[valid], "y": y[valid], "bin": bins}).groupby("bin", observed=False).agg(x=("x", "mean"), y=("y", "mean"))
        ax.plot(bm["x"], bm["y"], marker="o", color="#DD8452", linewidth=1.8, markersize=5)
    ax.set_xlabel("Acoustic CCA axis 1 score")
    ax.set_ylabel("Clinical CCA axis 1 score")
    ax.text(0.03, 0.97, f"Spearman ρ = {rho:.2f}\nP = {p:.2g}" if np.isfinite(p) else f"Spearman ρ = {rho:.2f}",
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.9), fontsize=10)
    clean_axes(ax)


def plot_null_distribution(ax, null_df: pd.DataFrame, observed_rho: float) -> None:
    ax.set_title("B  Null distribution", loc="left", fontweight="bold")
    rho_col = find_corr_col(null_df) if not null_df.empty else None
    if null_df.empty or rho_col is None:
        empty_panel(ax, "B  Null distribution", "Permutation/random-embedding table not found.")
        return
    df = null_df.copy()
    type_col = find_column(df, exact=["type", "control", "control_type", "analysis", "setting"])
    df["rho"] = as_numeric(df[rho_col])
    if type_col:
        if _HAS_SEABORN:
            sns.histplot(data=df, x="rho", hue=type_col, ax=ax, element="step", stat="density", common_norm=False, alpha=0.35)
        else:
            for label, sub in df.groupby(type_col):
                ax.hist(sub["rho"].dropna(), bins=20, alpha=0.35, density=True, label=str(label))
            ax.legend(frameon=False)
    else:
        ax.hist(df["rho"].dropna(), bins=25, alpha=0.6, density=True, color="#9ecae1")
    if np.isfinite(observed_rho):
        ax.axvline(observed_rho, color="#C44E52", linewidth=2, label="Observed")
    ax.set_xlabel("Out-of-fold Spearman ρ")
    ax.set_ylabel("Density")
    clean_axes(ax)


def plot_repeated_splits(ax, repeated_df: pd.DataFrame, observed_rho: float) -> None:
    ax.set_title("C  Repeated-split stability", loc="left", fontweight="bold")
    rho_col = find_corr_col(repeated_df) if not repeated_df.empty else None
    if repeated_df.empty or rho_col is None:
        empty_panel(ax, "C  Repeated-split stability", "Repeated-split table not found.")
        return
    rho = as_numeric(repeated_df[rho_col]).dropna()
    if _HAS_SEABORN:
        sns.stripplot(x=rho, ax=ax, size=4, alpha=0.55, color="#4C72B0")
        sns.boxplot(x=rho, ax=ax, width=0.25, fliersize=0, color="white", linewidth=1)
    else:
        ax.scatter(rho, np.zeros_like(rho), s=18, alpha=0.5)
    if np.isfinite(observed_rho):
        ax.axvline(observed_rho, color="#C44E52", linewidth=2)
    ax.set_yticks([])
    ax.set_xlabel("Out-of-fold Spearman ρ")
    clean_axes(ax)


def plot_axis_comparison(ax, axis_comp: pd.DataFrame) -> None:
    ax.set_title("D  CCA-axis comparison", loc="left", fontweight="bold")
    if axis_comp.empty or "rho" not in axis_comp.columns:
        empty_panel(ax, "D  CCA-axis comparison", "Multiple axis-score columns not found.")
        return
    df = axis_comp.dropna(subset=["rho"]).copy()
    if df.empty:
        empty_panel(ax, "D  CCA-axis comparison", "Axis correlations could not be computed.")
        return
    df["Axis"] = pd.Categorical(df["Axis"], categories=list(df["Axis"]), ordered=True)
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.plot(range(len(df)), df["rho"], marker="o", color="#4C72B0", linewidth=1.8)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["Axis"], rotation=30, ha="right")
    ax.set_ylabel("Spearman ρ")
    ax.set_xlabel("CCA axis")
    clean_axes(ax)


def make_figure3(tables: Dict[str, pd.DataFrame], out: OutputPaths, dpi: int) -> Tuple[pd.DataFrame, Optional[str], Optional[str], float]:
    axis_scores, acoustic_col, clinical_col = find_axis_scores(tables)
    observed_rho = get_observed_rho(axis_scores, acoustic_col, clinical_col)
    null_df = find_rho_table(tables, ["null", "permutation", "negative", "random"])
    repeated_df = find_rho_table(tables, ["repeat", "split", "bootstrap", "stability"])
    axis_comp = find_rho_table(tables, ["axis", "component", "cca"])
    if axis_comp.empty or "rho" not in [norm_col(c) for c in axis_comp.columns]:
        axis_comp = compute_axis_comparison(axis_scores)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    plot_axis_scatter(axes[0, 0], axis_scores, acoustic_col, clinical_col)
    plot_null_distribution(axes[0, 1], null_df, observed_rho)
    plot_repeated_splits(axes[1, 0], repeated_df, observed_rho)
    plot_axis_comparison(axes[1, 1], standardize_axis_comp(axis_comp))
    plt.tight_layout()
    save_figure(fig, out.fig_dir / "Figure3_cross_validated_cca_alignment.png", dpi)
    return axis_scores, acoustic_col, clinical_col, observed_rho


def standardize_axis_comp(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    rho_col = find_corr_col(out)
    axis_col = find_column(out, exact=["axis", "component", "cca_axis", "cca_component"])
    if rho_col and rho_col != "rho":
        out["rho"] = as_numeric(out[rho_col])
    if axis_col and axis_col != "Axis":
        out["Axis"] = out[axis_col].map(lambda x: f"Axis {int(x)}" if str(x).replace(".", "", 1).isdigit() else str(x))
    if "Axis" not in out.columns:
        out["Axis"] = [f"Axis {i+1}" for i in range(len(out))]
    return out


# -----------------------------
# Clinical profile and Figure 4
# -----------------------------

def get_axis_patient_table(axis_scores: pd.DataFrame, acoustic_col: Optional[str]) -> pd.DataFrame:
    if axis_scores.empty or acoustic_col is None or "patient_id" not in axis_scores.columns:
        return pd.DataFrame()
    out = axis_scores[["patient_id", acoustic_col]].copy()
    out = out.rename(columns={acoustic_col: "acoustic_axis1"})
    out["acoustic_axis1"] = as_numeric(out["acoustic_axis1"])
    return out


def compute_variable_associations(clinical: pd.DataFrame, axis_pt: pd.DataFrame, registry: Optional[pd.DataFrame]) -> pd.DataFrame:
    if axis_pt.empty:
        return pd.DataFrame()
    df = clinical.merge(axis_pt, on="patient_id", how="inner")
    rows = []
    variables = [c for c in clinical.columns if c != "patient_id" and not c.startswith(("n_windows", "duration"))]
    for c in variables:
        if c not in df.columns:
            continue
        x = as_numeric(df[c])
        if x.notna().sum() < 30 or x.nunique(dropna=True) < 2:
            continue
        rho, p = spearman_corr(df["acoustic_axis1"], x)
        rows.append({
            "variable": c,
            "display": DISPLAY_NAME.get(c, c),
            "domain": get_domain_for_variable(c, registry),
            "rho": rho,
            "p_value": p,
            "n": int((df["acoustic_axis1"].notna() & x.notna()).sum()),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["FDR"] = bh_fdr(out["p_value"])
        out = out.sort_values("rho")
    return out


def get_domain_for_variable(var: str, registry: Optional[pd.DataFrame]) -> str:
    if registry is not None and "clean_name" in registry.columns and "group" in registry.columns:
        hit = registry.loc[registry["clean_name"].astype(str) == var]
        if not hit.empty:
            g = str(hit.iloc[0]["group"])
            if "demographic" in g.lower():
                return "Demographics"
            if "structure" in g.lower():
                return "Structure"
            if "valve" in g.lower():
                return "Valve status"
            if "burden" in g.lower() or "function" in g.lower():
                return "Function / burden"
            if "technical" in g.lower():
                return "Technical"
            return g
    return DOMAIN_MAP.get(var, "Other")


def plot_variable_lollipop(ax, assoc: pd.DataFrame, top_n: int = 18) -> None:
    ax.set_title("A  Variable-wise clinical profile", loc="left", fontweight="bold")
    if assoc.empty:
        empty_panel(ax, "A  Variable-wise clinical profile", "Axis scores or clinical variables not available.")
        return
    df = assoc.copy().dropna(subset=["rho"])
    if len(df) > top_n:
        df = pd.concat([df.nsmallest(top_n // 2, "rho"), df.nlargest(top_n - top_n // 2, "rho")])
    df = df.sort_values("rho")
    y = np.arange(len(df))
    colors = [DOMAIN_PALETTE.get(d, DOMAIN_PALETTE["Other"]) for d in df["domain"]]
    ax.axvline(0, color="#777777", linewidth=0.8)
    for i, (rho, color) in enumerate(zip(df["rho"], colors)):
        ax.plot([0, rho], [i, i], color=color, linewidth=2, alpha=0.7)
        ax.scatter(rho, i, color=color, s=40, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["display"])
    ax.set_xlabel("Spearman ρ with acoustic CCA axis 1")
    ax.set_ylabel("")
    handles = []
    for dom in df["domain"].dropna().unique():
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=DOMAIN_PALETTE.get(dom, "#666"), markersize=7, label=dom))
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=8)
    clean_axes(ax)


def add_axis_quartile(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "acoustic_axis1" not in df.columns:
        return df
    df["axis_quartile"] = pd.qcut(df["acoustic_axis1"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    return df


def plot_continuous_quartiles(ax, df: pd.DataFrame) -> None:
    ax.set_title("B  Continuous clinical variables across axis quartiles", loc="left", fontweight="bold")
    if df.empty or "axis_quartile" not in df.columns:
        empty_panel(ax, "B  Continuous clinical variables across axis quartiles", "Axis quartiles could not be computed.")
        return
    plot_rows = []
    for var in PROFILE_CONTINUOUS:
        if var not in df.columns:
            continue
        y = as_numeric(df[var])
        if var == "NTproBNP":
            y = np.log1p(y)
            label = "log1p(NT-proBNP)"
        else:
            label = DISPLAY_NAME.get(var, var)
        temp = pd.DataFrame({"quartile": df["axis_quartile"], "value": y, "variable": label}).dropna()
        if temp.empty:
            continue
        summ = temp.groupby(["variable", "quartile"], observed=False).agg(
            mean=("value", "mean"),
            se=("value", lambda z: z.std(ddof=1) / math.sqrt(len(z)) if len(z) > 1 else np.nan),
        ).reset_index()
        plot_rows.append(summ)
    if not plot_rows:
        empty_panel(ax, "B  Continuous clinical variables across axis quartiles", "Selected variables not available.")
        return
    summ = pd.concat(plot_rows, ignore_index=True)
    variables = list(summ["variable"].unique())
    palette = sns.color_palette("Set2", n_colors=len(variables)) if _HAS_SEABORN else [None]*len(variables)
    for i, var in enumerate(variables):
        ss = summ[summ["variable"] == var]
        xs = np.arange(len(ss))
        ax.errorbar(xs, ss["mean"], yerr=1.96 * ss["se"], marker="o", linewidth=1.8, capsize=3, label=var, color=palette[i])
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_xlabel("Acoustic-axis quartile")
    ax.set_ylabel("Mean value (95% CI)")
    ax.legend(frameon=False, fontsize=8, loc="best")
    clean_axes(ax)


def plot_binary_quartiles(ax, df: pd.DataFrame) -> None:
    ax.set_title("C  Threshold-based clinical profiles", loc="left", fontweight="bold")
    if df.empty or "axis_quartile" not in df.columns:
        empty_panel(ax, "C  Threshold-based clinical profiles", "Axis quartiles could not be computed.")
        return
    rows = []
    for label, (var, op, thr) in PROFILE_BINARY.items():
        if var not in df.columns:
            continue
        x = as_numeric(df[var])
        positive = x >= thr if op == ">=" else x > thr
        for q, sub_idx in df.groupby("axis_quartile", observed=False).groups.items():
            idx = list(sub_idx)
            mask = positive.loc[idx]
            valid = mask.notna()
            n = int(valid.sum())
            k = int(mask[valid].sum()) if n else 0
            lo, hi = bootstrap_ci_proportion(k, n)
            rows.append({"profile": label, "quartile": str(q), "prop": k / n if n else np.nan, "lo": lo, "hi": hi, "n": n})
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        empty_panel(ax, "C  Threshold-based clinical profiles", "Selected threshold variables not available.")
        return
    profiles = list(plot_df["profile"].unique())
    palette = sns.color_palette("Set2", n_colors=len(profiles)) if _HAS_SEABORN else [None]*len(profiles)
    q_order = ["Q1", "Q2", "Q3", "Q4"]
    for i, prof in enumerate(profiles):
        ss = plot_df[plot_df["profile"] == prof]
        xs = np.array([q_order.index(q) for q in ss["quartile"]])
        y = ss["prop"].to_numpy()
        yerr = np.vstack([y - ss["lo"].to_numpy(), ss["hi"].to_numpy() - y])
        ax.errorbar(xs, y, yerr=yerr, marker="o", linewidth=1.8, capsize=3, label=prof, color=palette[i])
    ax.set_xticks(range(4))
    ax.set_xticklabels(q_order)
    ax.set_ylim(0, min(1, max(0.25, plot_df["hi"].max() * 1.2)))
    ax.set_xlabel("Acoustic-axis quartile")
    ax.set_ylabel("Proportion (95% CI)")
    ax.legend(frameon=False, fontsize=8, loc="best")
    clean_axes(ax)


def make_figure4(clinical: pd.DataFrame, axis_scores: pd.DataFrame, acoustic_col: Optional[str], registry: Optional[pd.DataFrame], out: OutputPaths, dpi: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    axis_pt = get_axis_patient_table(axis_scores, acoustic_col)
    merged = clinical.merge(axis_pt, on="patient_id", how="inner") if not axis_pt.empty else pd.DataFrame()
    merged = add_axis_quartile(merged) if not merged.empty else merged
    assoc = compute_variable_associations(clinical, axis_pt, registry)

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.2, 1.0], height_ratios=[1, 1], hspace=0.35, wspace=0.38)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    plot_variable_lollipop(ax_a, assoc, top_n=22)
    plot_continuous_quartiles(ax_b, merged)
    plot_binary_quartiles(ax_c, merged)
    save_figure(fig, out.fig_dir / "Figure4_clinical_axis_profile.png", dpi)
    return assoc, merged


# -----------------------------
# Acoustic profile supplement
# -----------------------------

def _safe_float_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    x = x.astype(np.float32, copy=False)
    # Convert int16-like audio to roughly [-1, 1] if needed.
    max_abs = np.nanmax(np.abs(x)) if x.size else 1.0
    if max_abs > 10:
        x = x / 32768.0
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def compute_window_features(windows: np.ndarray, fs: float) -> pd.DataFrame:
    w = _safe_float_array(windows)
    rows = []
    for x in w:
        if x.size < 8:
            continue
        x = x - np.mean(x)
        rms = float(np.sqrt(np.mean(x ** 2)))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(x).astype(int))))) if x.size > 1 else np.nan
        spec = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
        total = float(np.sum(spec) + 1e-12)
        centroid = float(np.sum(freqs * spec) / total)
        p = spec / total
        entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p) + 1e-12))
        def band(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            return float(np.sum(spec[mask]) / total) if mask.any() else np.nan
        low = band(20, 100)
        mid = band(100, 250)
        high = band(250, 600)
        murmur = band(150, 600)
        periodicity = autocorr_periodicity(x, fs)
        rows.append({
            "RMS energy": rms,
            "Zero-crossing rate": zcr,
            "Spectral centroid": centroid,
            "Low-frequency energy ratio": low,
            "Mid-frequency energy ratio": mid,
            "High-frequency energy ratio": high,
            "Murmur-band energy ratio": murmur,
            "Spectral entropy": entropy,
            "S1/S2 periodicity proxy": periodicity,
        })
    return pd.DataFrame(rows)


def autocorr_periodicity(x: np.ndarray, fs: float, lag_min_sec: float = 0.30, lag_max_sec: float = 2.00) -> float:
    if x.size < int(lag_min_sec * fs) + 2:
        return np.nan
    x = x - np.mean(x)
    denom = np.sum(x ** 2) + 1e-12
    ac = np.correlate(x, x, mode="full")[x.size - 1:]
    ac = ac / denom
    lo = int(lag_min_sec * fs)
    hi = min(int(lag_max_sec * fs), len(ac))
    if hi <= lo:
        return np.nan
    return float(np.nanmax(ac[lo:hi]))


def compute_patient_acoustic_features(
    window_library_dir: Path,
    patient_ids: Sequence[str],
    fs: float,
    max_windows_per_position: int,
    cache_path: Path,
) -> pd.DataFrame:
    if cache_path.exists():
        try:
            return read_csv_safely(cache_path)
        except Exception:
            pass
    rows = []
    for i, pid in enumerate(patient_ids):
        pdir = window_library_dir / str(pid)
        if not pdir.exists():
            continue
        pos_features = []
        for pos in POSITION_ORDER:
            npy = pdir / f"{pos}_windows.npy"
            if not npy.exists():
                continue
            try:
                arr = np.load(npy, mmap_mode="r")
                if arr.ndim == 1:
                    arr = arr[None, :]
                if arr.shape[0] > max_windows_per_position:
                    idx = np.linspace(0, arr.shape[0] - 1, max_windows_per_position).round().astype(int)
                    arr = np.asarray(arr[idx])
                else:
                    arr = np.asarray(arr)
                feats = compute_window_features(arr, fs)
                if not feats.empty:
                    feats["position"] = pos
                    pos_features.append(feats)
            except Exception as e:
                warnings.warn(f"Could not process {npy}: {e}")
        if not pos_features:
            continue
        all_feat = pd.concat(pos_features, ignore_index=True)
        agg = all_feat.drop(columns=["position"], errors="ignore").mean(numeric_only=True).to_dict()
        agg["patient_id"] = pid
        rows.append(agg)
        if (i + 1) % 100 == 0:
            print(f"[info] Acoustic features processed for {i+1}/{len(patient_ids)} patients")
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(cache_path, index=False, encoding="utf-8-sig")
        print(f"[done] Saved acoustic-feature cache: {cache_path}")
    return out


def make_acoustic_profile_figure(
    window_library_dir: Path,
    axis_scores: pd.DataFrame,
    acoustic_col: Optional[str],
    out: OutputPaths,
    fs: float,
    max_windows_per_position: int,
    dpi: int,
) -> pd.DataFrame:
    axis_pt = get_axis_patient_table(axis_scores, acoustic_col)
    if axis_pt.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        empty_panel(ax, "Acoustic profile of the aligned axis", "Axis-score table not found.")
        save_figure(fig, out.fig_dir / "FigureS1_acoustic_profile.png", dpi)
        return pd.DataFrame()
    feats = compute_patient_acoustic_features(
        window_library_dir=window_library_dir,
        patient_ids=axis_pt["patient_id"].astype(str).tolist(),
        fs=fs,
        max_windows_per_position=max_windows_per_position,
        cache_path=out.cache_dir / "patient_acoustic_descriptor_profile.csv",
    )
    if feats.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        empty_panel(ax, "Acoustic profile of the aligned axis", "Window library not found or no .npy windows could be processed.")
        save_figure(fig, out.fig_dir / "FigureS1_acoustic_profile.png", dpi)
        return feats
    merged = axis_pt.merge(feats, on="patient_id", how="inner")
    merged = add_axis_quartile(merged)
    feature_cols = [c for c in feats.columns if c != "patient_id"]
    rows = []
    for feat in feature_cols:
        rho, p = spearman_corr(merged["acoustic_axis1"], merged[feat])
        rows.append({"feature": feat, "rho": rho, "p_value": p})
    assoc = pd.DataFrame(rows)
    assoc["FDR"] = bh_fdr(assoc["p_value"])

    fig = plt.figure(figsize=(13, 8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.9, 1.2], wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    # lollipop
    a = assoc.sort_values("rho")
    y = np.arange(len(a))
    ax1.axvline(0, color="#777777", linewidth=0.8)
    for i, (_, r) in enumerate(a.iterrows()):
        col = "#4C72B0" if r["rho"] >= 0 else "#DD8452"
        ax1.plot([0, r["rho"]], [i, i], color=col, linewidth=2)
        ax1.scatter(r["rho"], i, color=col, s=42)
    ax1.set_yticks(y)
    ax1.set_yticklabels(a["feature"])
    ax1.set_xlabel("Spearman ρ with acoustic CCA axis 1")
    ax1.set_title("A  Acoustic-descriptor associations", loc="left", fontweight="bold")
    clean_axes(ax1)

    # quartile means for selected top features
    top_feats = assoc.reindex(assoc["rho"].abs().sort_values(ascending=False).index).head(5)["feature"].tolist()
    plot_rows = []
    for feat in top_feats:
        temp = merged[["axis_quartile", feat]].dropna().rename(columns={feat: "value"})
        if temp.empty:
            continue
        summ = temp.groupby("axis_quartile", observed=False).agg(
            mean=("value", "mean"),
            se=("value", lambda z: z.std(ddof=1) / math.sqrt(len(z)) if len(z) > 1 else np.nan),
        ).reset_index()
        summ["feature"] = feat
        plot_rows.append(summ)
    if plot_rows:
        pdq = pd.concat(plot_rows, ignore_index=True)
        palette = sns.color_palette("Set2", n_colors=len(top_feats)) if _HAS_SEABORN else [None] * len(top_feats)
        for i, feat in enumerate(top_feats):
            ss = pdq[pdq["feature"] == feat]
            xs = np.arange(len(ss))
            ax2.errorbar(xs, ss["mean"], yerr=1.96 * ss["se"], marker="o", linewidth=1.8, capsize=3, label=feat, color=palette[i])
        ax2.set_xticks(range(4))
        ax2.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        ax2.legend(frameon=False, fontsize=8, loc="best")
        ax2.set_xlabel("Acoustic-axis quartile")
        ax2.set_ylabel("Mean descriptor value (95% CI)")
        ax2.set_title("B  Acoustic descriptors across axis quartiles", loc="left", fontweight="bold")
        clean_axes(ax2)
    else:
        empty_panel(ax2, "B  Acoustic descriptors across axis quartiles", "No descriptor summaries available.")
    save_figure(fig, out.fig_dir / "FigureS1_acoustic_profile.png", dpi)
    return assoc


# -----------------------------
# Endpoint validation and Figure 5
# -----------------------------

def find_endpoint_table(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    candidates = []
    for name, df in tables.items():
        lname = norm_col(name)
        joined = " ".join(norm_col(c) for c in df.columns)
        if ("endpoint" in lname or "auroc" in lname or "roc" in lname or "validation" in lname) and ("auroc" in joined or "auc" in joined):
            candidates.append((name, df))
    if candidates:
        print(f"[info] Endpoint table: {candidates[0][0]}")
        return standardize_endpoint_table(candidates[0][1])
    return pd.DataFrame()


def standardize_endpoint_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = pd.DataFrame()
    endpoint_col = find_column(df, exact=["endpoint", "target", "outcome", "label"])
    auroc_col = find_column(df, exact=["auroc", "auc", "roc_auc"], contains_any=["auroc", "auc"])
    low_col = find_column(df, contains_all=["ci", "low"], contains_any=["lower", "low", "lo", "lcl"])
    high_col = find_column(df, contains_all=["ci", "high"], contains_any=["upper", "high", "hi", "ucl"])
    auprc_col = find_column(df, contains_any=["auprc", "average_precision", "ap"])
    sens_col = find_column(df, contains_any=["sensitivity", "recall", "tpr"])
    spec_col = find_column(df, contains_any=["specificity", "tnr"])
    n_pos_col = find_column(df, exact=["n_positive", "n_pos", "positive_n", "events"])
    n_neg_col = find_column(df, exact=["n_negative", "n_neg", "negative_n", "nonevents"])
    prev_col = find_column(df, exact=["prevalence", "prev"])
    removed_col = find_column(df, contains_any=["removed", "excluded", "left_out", "source_variable"])

    if endpoint_col is None:
        out["endpoint"] = [f"Endpoint {i+1}" for i in range(len(df))]
    else:
        out["endpoint"] = df[endpoint_col].astype(str)
    if auroc_col is not None:
        out["AUROC"] = as_numeric(df[auroc_col])
    else:
        out["AUROC"] = np.nan
    out["CI_low"] = as_numeric(df[low_col]) if low_col else np.nan
    out["CI_high"] = as_numeric(df[high_col]) if high_col else np.nan
    out["AUPRC"] = as_numeric(df[auprc_col]) if auprc_col else np.nan
    out["Sensitivity"] = as_numeric(df[sens_col]) if sens_col else np.nan
    out["Specificity"] = as_numeric(df[spec_col]) if spec_col else np.nan
    out["n_positive"] = as_numeric(df[n_pos_col]) if n_pos_col else np.nan
    out["n_negative"] = as_numeric(df[n_neg_col]) if n_neg_col else np.nan
    out["Prevalence"] = as_numeric(df[prev_col]) if prev_col else np.nan
    out["Removed variable"] = df[removed_col].astype(str) if removed_col else out["endpoint"].map(lambda e: ENDPOINT_DEFINITIONS.get(e, {}).get("removed_variable", ""))
    return out


def find_endpoint_predictions(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    candidates = []
    for name, df in tables.items():
        lname = norm_col(name)
        joined = " ".join(norm_col(c) for c in df.columns)
        if ("endpoint" in lname or "prediction" in lname or "roc" in lname) and any(k in joined for k in ["y_true", "label", "outcome", "score", "prob", "axis"]):
            candidates.append((name, df))
    for name, df in candidates:
        ep = find_column(df, exact=["endpoint", "target", "outcome_name"])
        y = find_column(df, exact=["y_true", "label", "target", "outcome", "truth"])
        score = find_column(df, contains_any=["score", "prob", "prediction", "axis"])
        if ep and y and score:
            print(f"[info] Endpoint prediction table: {name}")
            out = df.copy().rename(columns={ep: "endpoint", y: "y_true", score: "score"})
            out["y_true"] = as_numeric(out["y_true"])
            out["score"] = as_numeric(out["score"])
            return out
    return pd.DataFrame()


def derive_endpoint_statuses(clinical_axis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if clinical_axis.empty or "acoustic_axis1" not in clinical_axis.columns:
        return pd.DataFrame()
    for endpoint, spec in ENDPOINT_DEFINITIONS.items():
        var = spec["variable"]
        if var not in clinical_axis.columns:
            continue
        try:
            y = spec["rule"](clinical_axis)
        except Exception:
            continue
        temp = pd.DataFrame({
            "patient_id": clinical_axis["patient_id"],
            "endpoint": endpoint,
            "y_true": y.astype(float),
            "score": clinical_axis["acoustic_axis1"],
        }).dropna(subset=["y_true", "score"])
        rows.append(temp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _lvdd_dilation(df: pd.DataFrame) -> pd.Series:
    lv = as_numeric(df["LVEDD_mm"])
    if "sex_male" in df.columns:
        male = as_numeric(df["sex_male"]) == 1
        return (male & (lv > 58)) | (~male & (lv > 52))
    return lv > 55


def endpoint_summary_from_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    rows = []
    for endpoint, sub in pred.groupby("endpoint"):
        y = as_numeric(sub["y_true"])
        score = as_numeric(sub["score"])
        valid = y.notna() & score.notna()
        yv = y[valid].astype(int)
        sv = score[valid]
        if yv.nunique() < 2:
            continue
        auroc = roc_auc_score(yv, sv) if roc_auc_score else np.nan
        auprc = average_precision_score(yv, sv) if average_precision_score else np.nan
        rows.append({
            "endpoint": endpoint,
            "AUROC": auroc,
            "CI_low": np.nan,
            "CI_high": np.nan,
            "AUPRC": auprc,
            "Sensitivity": np.nan,
            "Specificity": np.nan,
            "n_positive": int(yv.sum()),
            "n_negative": int((1 - yv).sum()),
            "Prevalence": float(yv.mean()),
            "Removed variable": ENDPOINT_DEFINITIONS.get(str(endpoint), {}).get("removed_variable", ""),
        })
    return pd.DataFrame(rows)


def draw_endpoint_schematic(ax) -> None:
    ax.set_title("A  Leave-endpoint-out validation", loc="left", fontweight="bold")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        ("Clinical matrix", 0.7, 2.4, 2.0),
        ("Remove endpoint-\ndefining variable", 3.2, 2.4, 2.3),
        ("Fit CCA on\ntraining patients", 6.0, 2.4, 1.9),
        ("Evaluate endpoint\nin held-out patients", 8.2, 2.4, 2.0),
    ]
    for label, x, y, w in boxes:
        patch = FancyBboxPatch((x, y-0.5), w, 1.0, boxstyle="round,pad=0.04,rounding_size=0.08",
                               facecolor="#f7f7f7", edgecolor="#333333")
        ax.add_patch(patch)
        ax.text(x + w/2, y, label, ha="center", va="center", fontsize=9)
    for x1, x2 in [(2.7, 3.1), (5.5, 5.9), (7.9, 8.1)]:
        ax.add_patch(FancyArrowPatch((x1, 2.4), (x2, 2.4), arrowstyle="-|>", mutation_scale=12, color="#555555"))
    ax.text(5.0, 1.05, "Endpoint examples: EF <40%, NT-proBNP ≥900, LVEDD dilation, NYHA ≥3",
            ha="center", va="center", fontsize=9, color="#555555")


def plot_endpoint_forest(ax, endpoint_table: pd.DataFrame) -> None:
    ax.set_title("B  Endpoint AUROC", loc="left", fontweight="bold")
    if endpoint_table.empty or "AUROC" not in endpoint_table.columns:
        empty_panel(ax, "B  Endpoint AUROC", "Endpoint AUROC table not found.")
        return
    df = endpoint_table.dropna(subset=["AUROC"]).copy()
    if df.empty:
        empty_panel(ax, "B  Endpoint AUROC", "AUROC values not available.")
        return
    df = df.sort_values("AUROC")
    y = np.arange(len(df))
    low = df["CI_low"] if "CI_low" in df.columns else np.nan
    high = df["CI_high"] if "CI_high" in df.columns else np.nan
    for i, (_, r) in enumerate(df.iterrows()):
        lo = r.get("CI_low", np.nan)
        hi = r.get("CI_high", np.nan)
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [i, i], color="#4C72B0", linewidth=2)
        ax.scatter(r["AUROC"], i, s=55, color="#4C72B0", zorder=3)
    ax.axvline(0.5, color="#888888", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["endpoint"])
    ax.set_xlabel("AUROC (95% CI)")
    ax.set_xlim(0.45, min(1.0, max(0.85, df["AUROC"].max() + 0.1)))
    clean_axes(ax)


def plot_endpoint_rocs(ax, pred: pd.DataFrame) -> None:
    ax.set_title("C  ROC curves", loc="left", fontweight="bold")
    if pred.empty or roc_curve is None:
        empty_panel(ax, "C  ROC curves", "Endpoint-level prediction table not found.")
        return
    palette = sns.color_palette("Set2", n_colors=pred["endpoint"].nunique()) if _HAS_SEABORN else [None] * pred["endpoint"].nunique()
    for i, (endpoint, sub) in enumerate(pred.groupby("endpoint")):
        y = as_numeric(sub["y_true"])
        score = as_numeric(sub["score"])
        valid = y.notna() & score.notna()
        if valid.sum() < 5 or y[valid].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(y[valid].astype(int), score[valid])
        auc = roc_auc_score(y[valid].astype(int), score[valid]) if roc_auc_score else np.nan
        ax.plot(fpr, tpr, linewidth=1.8, label=f"{endpoint} ({auc:.2f})", color=palette[i])
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    clean_axes(ax)


def plot_endpoint_score_distribution(ax, pred: pd.DataFrame) -> None:
    ax.set_title("D  Endpoint prevalence and score distributions", loc="left", fontweight="bold")
    if pred.empty:
        empty_panel(ax, "D  Endpoint prevalence and score distributions", "Endpoint status table not available.")
        return
    df = pred.dropna(subset=["y_true", "score"]).copy()
    if df.empty:
        empty_panel(ax, "D  Endpoint prevalence and score distributions", "Endpoint status table not available.")
        return
    df["status"] = df["y_true"].map(lambda x: "Positive" if int(x) == 1 else "Negative")
    if _HAS_SEABORN:
        sns.boxplot(data=df, y="endpoint", x="score", hue="status", ax=ax, fliersize=0, linewidth=1, palette="Set2")
        sns.stripplot(data=df, y="endpoint", x="score", hue="status", dodge=True, ax=ax, size=2, alpha=0.25, palette="Set2", legend=False)
    else:
        for j, (endpoint, sub) in enumerate(df.groupby("endpoint")):
            for status, off in [("Negative", -0.12), ("Positive", 0.12)]:
                ss = sub[sub["status"] == status]["score"]
                ax.scatter(ss, np.full(len(ss), j + off), s=8, alpha=0.25)
    ax.set_xlabel("Acoustic-axis score")
    ax.set_ylabel("")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:2], labels[:2], frameon=False, fontsize=8, loc="best")
    clean_axes(ax)


def make_figure5(
    tables: Dict[str, pd.DataFrame],
    clinical_axis: pd.DataFrame,
    out: OutputPaths,
    dpi: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    endpoint_table = find_endpoint_table(tables)
    pred = find_endpoint_predictions(tables)
    if pred.empty:
        pred = derive_endpoint_statuses(clinical_axis)
    if endpoint_table.empty and not pred.empty:
        endpoint_table = endpoint_summary_from_predictions(pred)
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    draw_endpoint_schematic(ax_a)
    plot_endpoint_forest(ax_b, endpoint_table)
    plot_endpoint_rocs(ax_c, pred)
    plot_endpoint_score_distribution(ax_d, pred)
    save_figure(fig, out.fig_dir / "Figure5_endpoint_validation.png", dpi)
    return endpoint_table, pred


# -----------------------------
# Robustness and Figure 6
# -----------------------------

def standardize_covariate_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    setting_col = find_column(out, exact=["setting", "adjustment", "covariate", "model", "residualization", "analysis"])
    rho_col = find_corr_col(out)
    low_col = find_column(out, contains_all=["ci", "low"], contains_any=["lower", "low", "lo", "lcl"])
    high_col = find_column(out, contains_all=["ci", "high"], contains_any=["upper", "high", "hi", "ucl"])
    ret = pd.DataFrame()
    ret["setting"] = out[setting_col].astype(str) if setting_col else [f"Setting {i+1}" for i in range(len(out))]
    ret["rho"] = as_numeric(out[rho_col]) if rho_col else np.nan
    ret["low"] = as_numeric(out[low_col]) if low_col else np.nan
    ret["high"] = as_numeric(out[high_col]) if high_col else np.nan
    return ret


def standardize_site_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    setting_col = find_column(out, exact=["setting", "site", "position", "representation", "analysis", "model"])
    rho_col = find_corr_col(out)
    ret = pd.DataFrame()
    ret["setting"] = out[setting_col].astype(str) if setting_col else [f"Setting {i+1}" for i in range(len(out))]
    ret["rho"] = as_numeric(out[rho_col]) if rho_col else np.nan
    return ret.dropna(subset=["rho"])


def plot_covariate_forest(ax, cov_df: pd.DataFrame, observed_rho: float) -> None:
    ax.set_title("C  Covariate residualization", loc="left", fontweight="bold")
    df = standardize_covariate_table(cov_df)
    if df.empty or df["rho"].notna().sum() == 0:
        if np.isfinite(observed_rho):
            df = pd.DataFrame({"setting": ["Unadjusted"], "rho": [observed_rho], "low": [np.nan], "high": [np.nan]})
        else:
            empty_panel(ax, "C  Covariate residualization", "Covariate residualization table not found.")
            return
    df = df.dropna(subset=["rho"]).copy()
    y = np.arange(len(df))
    for i, (_, r) in enumerate(df.iterrows()):
        if np.isfinite(r.get("low", np.nan)) and np.isfinite(r.get("high", np.nan)):
            ax.plot([r["low"], r["high"]], [i, i], color="#4C72B0", linewidth=2)
        ax.scatter(r["rho"], i, s=50, color="#4C72B0")
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["setting"])
    ax.set_xlabel("Spearman ρ")
    clean_axes(ax)


def plot_site_sensitivity(ax, site_df: pd.DataFrame) -> None:
    ax.set_title("D  Auscultation-site sensitivity", loc="left", fontweight="bold")
    df = standardize_site_table(site_df)
    if df.empty:
        empty_panel(ax, "D  Auscultation-site sensitivity", "Site sensitivity table not found.")
        return
    df = df.sort_values("rho", ascending=True)
    colors = ["#4C72B0" if "full" in str(s).lower() or "five" in str(s).lower() else "#55A868" if "leave" in str(s).lower() else "#DD8452" for s in df["setting"]]
    ax.barh(np.arange(len(df)), df["rho"], color=colors, alpha=0.85)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["setting"])
    ax.set_xlabel("Out-of-fold Spearman ρ")
    clean_axes(ax)


def make_figure6(tables: Dict[str, pd.DataFrame], observed_rho: float, out: OutputPaths, dpi: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repeated_df = find_rho_table(tables, ["repeat", "split", "bootstrap", "stability"])
    null_df = find_rho_table(tables, ["null", "permutation", "negative", "random"])
    cov_df = find_rho_table(tables, ["residual", "covariate", "adjust", "age", "sex", "heart"])
    site_df = find_rho_table(tables, ["site", "position", "auscult", "leave_one", "single"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    plot_repeated_splits(axes[0, 0], repeated_df, observed_rho)
    axes[0, 0].set_title("A  Repeated-split stability", loc="left", fontweight="bold")
    plot_null_distribution(axes[0, 1], null_df, observed_rho)
    axes[0, 1].set_title("B  Negative controls", loc="left", fontweight="bold")
    plot_covariate_forest(axes[1, 0], cov_df, observed_rho)
    plot_site_sensitivity(axes[1, 1], site_df)
    plt.tight_layout()
    save_figure(fig, out.fig_dir / "Figure6_robustness_summary.png", dpi)
    return repeated_df, null_df, cov_df, site_df


# -----------------------------
# Excel output
# -----------------------------

def make_endpoint_table_for_excel(endpoint_table: pd.DataFrame) -> pd.DataFrame:
    if endpoint_table.empty:
        return endpoint_table
    df = endpoint_table.copy()
    df["AUROC (95% CI)"] = [fmt_ci(v, l, h, 3) for v, l, h in zip(df["AUROC"], df.get("CI_low", np.nan), df.get("CI_high", np.nan))]
    if "AUPRC" in df.columns:
        df["AUPRC"] = df["AUPRC"].map(lambda x: fmt_num(x, 3) if pd.notna(x) else "NA")
    if "Prevalence" in df.columns:
        df["Prevalence"] = df["Prevalence"].map(lambda x: f"{100*x:.1f}%" if pd.notna(x) and x <= 1 else fmt_num(x, 1))
    keep = [c for c in ["endpoint", "Removed variable", "n_positive", "n_negative", "Prevalence", "AUROC (95% CI)", "AUPRC", "Sensitivity", "Specificity"] if c in df.columns]
    return df[keep]


def write_excel_tables(
    path: Path,
    table1: pd.DataFrame,
    clinical_matrix: pd.DataFrame,
    retrieval_across: pd.DataFrame,
    retrieval_bywin: pd.DataFrame,
    variable_assoc: pd.DataFrame,
    endpoint_table: pd.DataFrame,
    robustness_tables: Dict[str, pd.DataFrame],
    acoustic_assoc: pd.DataFrame,
) -> None:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            table1.to_excel(writer, sheet_name="Table1_cohort_recording", index=False)
            clinical_matrix.to_excel(writer, sheet_name="TableS1_clinical_matrix", index=False)
            if not retrieval_across.empty:
                retrieval_across.to_excel(writer, sheet_name="TableS2_retrieval_across", index=False)
            if not retrieval_bywin.empty:
                retrieval_bywin.to_excel(writer, sheet_name="TableS3_retrieval_by_window", index=False)
            if not variable_assoc.empty:
                va = variable_assoc.copy()
                va["rho (p, FDR)"] = va.apply(lambda r: f"{r['rho']:.3f} (p={r['p_value']:.2g}, FDR={r['FDR']:.2g})" if pd.notna(r.get('rho')) else "NA", axis=1)
                va.to_excel(writer, sheet_name="TableS4_variable_assoc", index=False)
            if not endpoint_table.empty:
                make_endpoint_table_for_excel(endpoint_table).to_excel(writer, sheet_name="Table2_endpoint_validation", index=False)
                endpoint_table.to_excel(writer, sheet_name="TableS5_endpoint_raw", index=False)
            for name, df in robustness_tables.items():
                if df is not None and not df.empty:
                    sheet = re.sub(r"[^A-Za-z0-9_]+", "_", name)[:31]
                    df.to_excel(writer, sheet_name=sheet, index=False)
            if acoustic_assoc is not None and not acoustic_assoc.empty:
                acoustic_assoc.to_excel(writer, sheet_name="TableS_acoustic_profile", index=False)
        print(f"[done] Saved Excel workbook: {path}")
    except ImportError as e:
        raise RuntimeError("Writing .xlsx requires openpyxl. Install it with: pip install openpyxl") from e


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()
    setup_style()
    out = ensure_outputs(args.out_dir)

    fallback = Path("/mnt/data") if Path("/mnt/data").exists() else None

    prepared_dir = args.prepared_dir
    clinical_path = resolve_file(prepared_dir, "aligned_clinical_clean.csv", fallback)
    clinical_tech_path = resolve_file(prepared_dir, "aligned_clinical_plus_technical.csv", fallback)
    missing_path = resolve_file(prepared_dir, "missingness_summary.csv", fallback)
    registry_path = resolve_file(prepared_dir, "registry_snapshot.csv", fallback)

    clinical = read_csv_safely(clinical_path) if clinical_path.exists() else pd.DataFrame()
    clinical_tech = read_csv_safely(clinical_tech_path) if clinical_tech_path.exists() else clinical.copy()
    missingness = read_csv_safely(missing_path) if missing_path.exists() else None
    registry = read_csv_safely(registry_path) if registry_path.exists() else None

    if clinical.empty:
        raise FileNotFoundError(f"Cannot find aligned_clinical_clean.csv in {prepared_dir} or /mnt/data")
    if clinical_tech.empty:
        clinical_tech = clinical.copy()

    tables = load_all_tables(args.tables_dir)

    # Table 1 and clinical matrix table
    table1 = make_table1(clinical_tech)
    clinical_matrix = make_clinical_matrix_table(registry, missingness, clinical)

    # Figure 2 and retrieval tables
    retrieval_across, retrieval_bywin = load_retrieval_data(args.retrieval_dir)
    make_figure2(retrieval_bywin if not retrieval_bywin.empty else retrieval_across, out, args.dpi)

    # Figure 3: CCA alignment
    axis_scores, acoustic_col, clinical_col, observed_rho = make_figure3(tables, out, args.dpi)

    # Figure 4: clinical profile
    variable_assoc, clinical_axis = make_figure4(clinical, axis_scores, acoustic_col, registry, out, args.dpi)

    # Figure S1: acoustic meaning / acoustic profile
    acoustic_assoc = pd.DataFrame()
    if not args.no_acoustic_profile:
        acoustic_assoc = make_acoustic_profile_figure(
            args.window_library_dir,
            axis_scores,
            acoustic_col,
            out,
            fs=args.fs,
            max_windows_per_position=args.max_windows_per_position,
            dpi=args.dpi,
        )

    # Figure 5: endpoint validation
    endpoint_table, endpoint_pred = make_figure5(tables, clinical_axis, out, args.dpi)

    # Figure 6: robustness summary
    repeated_df, null_df, cov_df, site_df = make_figure6(tables, observed_rho, out, args.dpi)

    robustness_tables = {
        "TableS6_repeated_splits": repeated_df,
        "TableS7_negative_controls": null_df,
        "TableS8_covariate_residual": cov_df,
        "TableS9_site_sensitivity": site_df,
    }

    write_excel_tables(
        out.table_dir / "clinical_alignment_tables.xlsx",
        table1=table1,
        clinical_matrix=clinical_matrix,
        retrieval_across=retrieval_across,
        retrieval_bywin=retrieval_bywin,
        variable_assoc=variable_assoc,
        endpoint_table=endpoint_table,
        robustness_tables=robustness_tables,
        acoustic_assoc=acoustic_assoc,
    )

    # Also save a simple manifest.
    manifest = pd.DataFrame({
        "output": [
            "tables/clinical_alignment_tables.xlsx",
            "figures/Figure2_acoustic_representation_selection.png",
            "figures/Figure3_cross_validated_cca_alignment.png",
            "figures/Figure4_clinical_axis_profile.png",
            "figures/Figure5_endpoint_validation.png",
            "figures/Figure6_robustness_summary.png",
            "figures/FigureS1_acoustic_profile.png",
        ]
    })
    manifest.to_csv(out.out_dir / "output_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"[done] Saved manifest: {out.out_dir / 'output_manifest.csv'}")


if __name__ == "__main__":
    main()
