#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clinically anchored acoustic phenotyping from multi-site heart-sound embeddings.

V3 script for the manuscript question:
    Do multi-site heart-sound representations contain patient-level acoustic
    phenotype structures that align with multidimensional clinical status?

Design principles
-----------------
1) Single-variable acoustic readout is used as a complementary signal overview
   and is NOT framed as a failed alternative to multivariable alignment.
2) Main alignment uses CCA only. CCA is fitted inside each training fold.
   Test patients are projected by the fold-specific CCA model. The main metric
   is out-of-fold latent-score correlation between acoustic and clinical axes.
3) Clinical panels are pre-specified and analyzed separately:
   - functional_impairment_hf_burden: EF, NT-proBNP, NYHA
   - structural_remodeling: LA, LVEDD, IVS, LVPW
   - valvular_regurgitation: MR, TR, AR, PR
   - valvular_stenosis: AS, MS
   - all_clinical: the union of all pre-specified domain variables
   Heart rate is not used as an anchoring target; it is handled as a covariate.
4) Reports permutation p values, bootstrap CIs, clinical redundancy, fold-level
   sign stability, bootstrap loading stability, clinical gradients, endpoint
   validation, minimal negative controls, and basic confounder/position controls.

Example
-------
python run_clinically_anchored_acoustic_phenotyping_v4.py --embedding-dir Representation_learning/embeddings_4_1/beats --clinical-csv Clinical_alignment/outputs/prepared/aligned_clinical_clean.csv --out-dir Clinical_alignment/outputs/clinically_anchored_acoustic_phenotyping_v4 --n-splits 5 --n-components 2 --n-pca 50 --n-permutations 100 --n-bootstrap 1000 --n-loading-bootstrap 200 --n-random-controls 20 --seed 42
"""

from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from dataclasses import dataclass
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
except Exception:  # pragma: no cover
    HAS_SEABORN = False

from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# General utilities
# =============================================================================

def now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def df_to_markdown_safe(df: pd.DataFrame, index: bool = False) -> str:
    """Safe markdown conversion. Fixes the undefined function bug in the old script."""
    if df is None or len(df) == 0:
        return "\n(empty)\n"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    try:
        return "\n" + d.to_markdown(index=index) + "\n"
    except Exception:
        return "\n" + d.to_string(index=index) + "\n"


def normalize_patient_id(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    s = re.sub(r"\s+", "", s)
    return s


def clean_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", str(s))[:180]


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 5 or np.nanstd(x[mask]) < 1e-12 or np.nanstd(y[mask]) < 1e-12:
        return np.nan, np.nan, n
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p), n


def safe_pearson(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 5 or np.nanstd(x[mask]) < 1e-12 or np.nanstd(y[mask]) < 1e-12:
        return np.nan, np.nan, n
    r, p = stats.pearsonr(x[mask], y[mask])
    return float(r), float(p), n


def fdr_bh(pvals: Sequence[float]) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    q = np.full_like(pvals, np.nan, dtype=float)
    finite = np.isfinite(pvals)
    p = pvals[finite]
    if len(p) == 0:
        return q
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty_like(p)
    out[order] = adj
    q[finite] = out
    return q


def bootstrap_ci_spearman(x, y, n_boot: int, seed: int) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 10 or n_boot <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.nanstd(x[idx]) < 1e-12 or np.nanstd(y[idx]) < 1e-12:
            continue
        rho = stats.spearmanr(x[idx], y[idx]).correlation
        if np.isfinite(rho):
            vals.append(float(rho))
    if len(vals) < 20:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_ci_auc(y_true, y_score, n_boot: int, seed: int) -> Tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    n = len(y_true)
    if n < 20 or len(np.unique(y_true)) < 2 or n_boot <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(float(roc_auc_score(y_true[idx], y_score[idx])))
    if len(vals) < 20:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# =============================================================================
# Loading data
# =============================================================================

def infer_patient_id_col(df: pd.DataFrame, explicit: Optional[str] = None, context: str = "table") -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Explicit patient ID column '{explicit}' not found in {context}.")
        return explicit
    candidates = [
        "patient_id", "patient", "pid", "ID", "id", "PatientID", "subject_id", "subject", "case_id",
        "住院号", "病人ID", "患者ID", "患者编号", "编号", "病例号", "就诊号",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    if df.shape[1] == 1:
        return df.columns[0]
    for c in df.columns:
        lc = str(c).lower()
        if ("patient" in lc or "subject" in lc or lc in {"pid", "id"}) and "embedding" not in lc:
            return c
    raise ValueError(f"Cannot infer patient ID column in {context}. Columns={list(df.columns)}")


def find_first_existing(base: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        p = base / name
        if p.exists():
            return p
    return None


def load_patient_embeddings(args) -> Tuple[np.ndarray, pd.DataFrame, Path, Path]:
    emb_dir = Path(args.embedding_dir)
    if args.patient_embedding_npy:
        emb_path = Path(args.patient_embedding_npy)
    else:
        emb_path = find_first_existing(
            emb_dir,
            [
                "patient_embeddings.npy", "patient_embedding.npy", "patient_level_embeddings.npy",
                "patient_embeds.npy", "patient_emb.npy", "embeddings_patient.npy",
            ],
        )
        if emb_path is None:
            matches = sorted(emb_dir.glob("*patient*embed*.npy"))
            emb_path = matches[0] if matches else None
    if emb_path is None or not emb_path.exists():
        raise FileNotFoundError("Cannot find patient-level embedding .npy. Pass --patient-embedding-npy.")

    if args.patient_meta_csv:
        meta_path = Path(args.patient_meta_csv)
    else:
        meta_path = find_first_existing(
            emb_dir,
            [
                "patient_meta.csv", "patient_metadata.csv", "patient_embeddings_meta.csv",
                "patient_order.csv", "patient_ids.csv", "patients.csv",
            ],
        )
        if meta_path is None:
            matches = sorted(emb_dir.glob("*patient*meta*.csv")) + sorted(emb_dir.glob("*patient*order*.csv"))
            meta_path = matches[0] if matches else None
    if meta_path is None or not meta_path.exists():
        raise FileNotFoundError("Cannot find patient-level meta/order CSV. Pass --patient-meta-csv.")

    X = np.load(emb_path)
    meta = pd.read_csv(meta_path)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got shape={X.shape}")
    if len(meta) != X.shape[0]:
        raise ValueError(f"Meta rows ({len(meta)}) do not match embedding rows ({X.shape[0]})")
    pid_col = infer_patient_id_col(meta, explicit=args.embedding_patient_id_col, context="embedding meta")
    meta = meta.copy()
    meta["patient_id"] = meta[pid_col].map(normalize_patient_id)
    meta["embedding_row"] = np.arange(len(meta))
    if meta["patient_id"].duplicated().any():
        dup = meta.loc[meta["patient_id"].duplicated(), "patient_id"].head().tolist()
        raise ValueError(f"Duplicate patient_id in embedding meta, e.g. {dup}")
    log(f"Loaded patient embeddings: {emb_path}, shape={X.shape}")
    log(f"Loaded patient meta: {meta_path}, patient ID column='{pid_col}'")
    return X.astype(np.float32), meta, emb_path, meta_path


def read_clinical_table(args) -> pd.DataFrame:
    if args.clinical_csv:
        path = Path(args.clinical_csv)
        if not path.exists():
            raise FileNotFoundError(f"Clinical CSV not found: {path}")
        df = pd.read_csv(path)
    else:
        path = Path(args.clinical_xlsx)
        if not path.exists():
            raise FileNotFoundError(f"Clinical Excel not found: {path}")
        df = pd.read_excel(path, sheet_name=args.clinical_sheet)
    pid_col = infer_patient_id_col(df, explicit=args.patient_id_col, context="clinical table")
    df = df.copy()
    df["patient_id"] = df[pid_col].map(normalize_patient_id)
    before = len(df)
    df = df.dropna(subset=["patient_id"]).drop_duplicates("patient_id", keep="first")
    log(f"Loaded clinical table: {before} rows -> {len(df)} unique patient IDs, ID column='{pid_col}'")
    return df


def coerce_to_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    mapping = {
        "i": 1, "ii": 2, "iii": 3, "iv": 4,
        "Ⅰ": 1, "Ⅱ": 2, "Ⅲ": 3, "Ⅳ": 4,
        "一": 1, "二": 2, "三": 3, "四": 4,
        "无": 0, "没有": 0, "否": 0, "阴性": 0, "正常": 0,
        "是": 1, "阳性": 1, "有": 1,
        "轻": 1, "轻度": 1, "mild": 1,
        "中": 2, "中度": 2, "moderate": 2,
        "重": 3, "重度": 3, "severe": 3,
        "少量": 1, "大量": 3,
        "男": 1, "male": 1, "m": 1,
        "女": 0, "female": 0, "f": 0,
    }
    out = []
    for v in s:
        if pd.isna(v):
            out.append(np.nan)
            continue
        txt = str(v).strip()
        low = txt.lower().replace(" ", "").replace(",", "")
        if low in mapping:
            out.append(mapping[low])
            continue
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", low)
        if m:
            try:
                out.append(float(m.group(0)))
            except Exception:
                out.append(np.nan)
        else:
            out.append(np.nan)
    return pd.Series(out, index=s.index, dtype=float)


def build_numeric_clinical_table(df: pd.DataFrame) -> pd.DataFrame:
    num = pd.DataFrame(index=df.index)
    for c in df.columns:
        if c == "patient_id":
            continue
        coerced = coerce_to_numeric(df[c])
        if coerced.notna().sum() >= 5:
            num[c] = coerced
    num.insert(0, "patient_id", df["patient_id"].values)
    return num


def align_embeddings_and_clinical(X: np.ndarray, meta: pd.DataFrame, clinical: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    merged = meta[["patient_id", "embedding_row"]].merge(clinical, on="patient_id", how="inner")
    if len(merged) == 0:
        raise ValueError("No overlapping patient IDs between embeddings and clinical table.")
    row_idx = merged["embedding_row"].to_numpy(dtype=int)
    X_aligned = X[row_idx]
    clinical_aligned = merged.drop(columns=["embedding_row"]).reset_index(drop=True)
    patient_df = merged[["patient_id", "embedding_row"]].reset_index(drop=True)
    log(f"Aligned acoustic embeddings with clinical table: n={len(patient_df)} patients")
    return X_aligned, clinical_aligned, patient_df


# =============================================================================
# Clinical concept detection and panel construction
# =============================================================================

@dataclass
class ClinicalConcept:
    concept: str
    domain: str
    patterns: List[str]
    orientation: int  # +1: higher means heavier burden; -1: higher means better function; 0: neutral
    priority: int


CONCEPTS: List[ClinicalConcept] = [
    ClinicalConcept("EF", "function_burden", [r"^ef$", r"lvef", r"ejection", r"射血"], -1, 1),
    ClinicalConcept("NTproBNP", "function_burden", [r"nt[-_ ]?pro[-_ ]?bnp", r"ntprobnp", r"nt.*bnp", r"脑钠", r"^bnp$"], +1, 1),
    ClinicalConcept("NYHA", "function_burden", [r"nyha"], +1, 1),
    ClinicalConcept("HR", "covariate", [r"^hr$", r"heart.*rate", r"心率"], +1, 2),

    ClinicalConcept("LA", "structure", [r"^la$", r"^lad$", r"left.*atri", r"左房", r"左心房"], +1, 1),
    ClinicalConcept("LVEDD", "structure", [r"lvedd", r"lvdd", r"left.*vent.*diast", r"左室.*舒张", r"左心室.*舒张"], +1, 1),
    ClinicalConcept("LVESD", "structure", [r"lvesd", r"left.*vent.*syst", r"左室.*收缩", r"左心室.*收缩"], +1, 2),
    ClinicalConcept("IVS", "structure", [r"ivs", r"室间隔"], +1, 3),
    ClinicalConcept("LVPW", "structure", [r"lvpw", r"后壁"], +1, 3),
    ClinicalConcept("RA", "structure", [r"^ra$", r"right.*atri", r"右房", r"右心房"], +1, 3),
    ClinicalConcept("RV", "structure", [r"^rv$", r"right.*vent", r"右室", r"右心室"], +1, 3),

    # Valve patterns are intentionally broad because cleaned clinical tables often use
    # names such as MR_grade, tricuspid_regurgitation, AR_severity, 二尖瓣返流, etc.
    ClinicalConcept("MR", "valve", [r"^mr($|[_\- ]|grade|severity)", r"mr[_\- ]?(grade|severity)", r"mitral.*(regurg|insuff)", r"二尖瓣.*(反流|返流|关闭不全)"], +1, 1),
    ClinicalConcept("TR", "valve", [r"^tr($|[_\- ]|grade|severity)", r"tr[_\- ]?(grade|severity)", r"tricuspid.*(regurg|insuff)", r"三尖瓣.*(反流|返流|关闭不全)"], +1, 1),
    ClinicalConcept("AR", "valve", [r"^ar($|[_\- ]|grade|severity)", r"ar[_\- ]?(grade|severity)", r"aortic.*(regurg|insuff)", r"主动脉.*(反流|返流|关闭不全)"], +1, 2),
    ClinicalConcept("PR", "valve", [r"^pr($|[_\- ]|grade|severity)", r"pr[_\- ]?(grade|severity)", r"pulmon.*(regurg|insuff)", r"肺动脉.*(反流|返流|关闭不全)"], +1, 3),
    ClinicalConcept("AS", "valve", [r"^as($|[_\- ]|grade|severity)", r"as[_\- ]?(grade|severity)", r"aortic.*sten", r"主动脉.*狭窄"], +1, 2),
    ClinicalConcept("MS", "valve", [r"^ms($|[_\- ]|grade|severity)", r"ms[_\- ]?(grade|severity)", r"mitral.*sten", r"二尖瓣.*狭窄"], +1, 2),
]

DEMOGRAPHIC_PATTERNS = {
    "age": [r"^age($|[_\- ]|year|yrs)", r"age", r"年龄"],
    "sex": [r"^sex($|[_\- ])", r"sex_male", r"gender", r"性别", r"男=1", r"男"],
}

# Exact column names observed in Clinical_alignment/outputs/prepared/aligned_clinical_clean.csv.
# These are tried before regex matching to avoid missing or mis-identifying important variables.
EXACT_CONCEPT_COLUMNS = {
    "EF": ["EF_Teich"],
    "NTproBNP": ["NTproBNP"],
    "NYHA": ["NYHA"],
    "HR": ["heart_rate"],
    "LA": ["LA_mm"],
    "LVEDD": ["LVEDD_mm"],
    "IVS": ["IVS_mm"],
    "LVPW": ["LVPW_mm"],
    "AR": ["AR_grade"],
    "AS": ["AS_grade"],
    "MS": ["MS_grade"],
    "MR": ["MR_grade"],
    "TR": ["TR_grade"],
    "PR": ["PR_grade"],
}

EXACT_COVARIATE_COLUMNS = {
    "age": "age_years",
    "sex": "sex_male",
    "heart_rate": "heart_rate",
}

EXACT_ENDPOINT_COLUMNS = {
    "EF_lt_40": "EF_Teich",
    "NTproBNP_ge_300": "NTproBNP",
    "NYHA_ge_3": "NYHA",
    "LA_ge_40": "LA_mm",
    "LVEDD_dilated": "LVEDD_mm",
}

DOMAIN_PANEL_CONCEPTS = {
    "functional_impairment_hf_burden": ["EF", "NTproBNP", "NYHA"],
    "structural_remodeling": ["LA", "LVEDD", "IVS", "LVPW"],
    "valvular_regurgitation": ["MR", "TR", "AR", "PR"],
    "valvular_stenosis": ["AS", "MS"],
}

DOMAIN_PLOT_PANELS = list(DOMAIN_PANEL_CONCEPTS.keys())


def select_first_matching_column(columns: Sequence[str], patterns: Sequence[str]) -> Optional[str]:
    for p in patterns:
        regex = re.compile(p, flags=re.I)
        matches = [c for c in columns if regex.search(str(c))]
        if matches:
            return sorted(matches, key=lambda x: (len(str(x)), str(x)))[0]
    return None


def _concept_thresholds(cc: ClinicalConcept, args) -> Tuple[float, int, int]:
    """Domain-specific variable inclusion thresholds.

    Valve columns are often sparse ordinal/binary variables in real-world echo tables,
    so they use a more permissive missingness and sample-size threshold than continuous
    function/structure variables. This helps distinguish "not detected" from "detected
    but too sparse" and prevents the valve panel from disappearing silently.
    """
    if cc.domain == "valve":
        return args.valve_max_missing, args.valve_min_n, args.valve_min_unique
    return args.max_missing, args.min_target_n, 2


def _filter_reason(miss: float, n_non: int, n_unique: int, max_missing: float, min_n: int, min_unique: int) -> str:
    reasons = []
    if miss > max_missing:
        reasons.append(f"missing>{max_missing:g}")
    if n_non < min_n:
        reasons.append(f"n<{min_n}")
    if n_unique < min_unique:
        reasons.append(f"unique<{min_unique}")
    return ";".join(reasons) if reasons else "ok"


def detect_clinical_concepts(num_df: pd.DataFrame, args) -> pd.DataFrame:
    """Detect clinically relevant columns and record why variables were kept/filtered.

    Important change from V2: for each concept, all matching columns are evaluated and
    the best usable column is selected. V2 selected the first regex match; if that first
    match was sparse or constant, the concept could be filtered even when another usable
    column existed.
    """
    cols = [c for c in num_df.columns if c != "patient_id"]
    used_cols = set()
    selected_rows = []
    candidate_rows = []

    for cc in CONCEPTS:
        matches = []
        for col in cols:
            if col in used_cols:
                continue
            if any(re.search(p, str(col), flags=re.I) for p in cc.patterns):
                matches.append(col)

        if not matches:
            selected_rows.append({
                "concept": cc.concept, "domain": cc.domain, "column": "", "status": "not_found",
                "filter_reason": "not_found", "missing_fraction": np.nan, "n_nonmissing": 0, "n_unique": 0,
                "orientation": cc.orientation, "priority": cc.priority,
                "max_missing_used": np.nan, "min_n_used": np.nan, "min_unique_used": np.nan,
            })
            continue

        max_missing, min_n, min_unique = _concept_thresholds(cc, args)
        cand = []
        for col in matches:
            miss = float(num_df[col].isna().mean())
            n_non = int(num_df[col].notna().sum())
            n_unique = int(num_df[col].nunique(dropna=True))
            reason = _filter_reason(miss, n_non, n_unique, max_missing, min_n, min_unique)
            status = "ok" if reason == "ok" else "filtered"
            row = {
                "concept": cc.concept, "domain": cc.domain, "column": col, "status": status,
                "filter_reason": reason, "missing_fraction": miss, "n_nonmissing": n_non,
                "n_unique": n_unique, "orientation": cc.orientation, "priority": cc.priority,
                "max_missing_used": max_missing, "min_n_used": min_n, "min_unique_used": min_unique,
            }
            cand.append(row)
            candidate_rows.append({**row, "is_selected_for_concept": False})

        cand_sorted = sorted(
            cand,
            key=lambda r: (
                0 if r["status"] == "ok" else 1,
                r["missing_fraction"] if np.isfinite(r["missing_fraction"]) else 1.0,
                -r["n_unique"],
                len(str(r["column"])),
                str(r["column"]),
            ),
        )
        best = cand_sorted[0]
        if best["status"] == "ok":
            used_cols.add(best["column"])
        selected_rows.append(best)
        for cr in candidate_rows:
            if cr["concept"] == cc.concept and cr["column"] == best["column"]:
                cr["is_selected_for_concept"] = True

    selected = pd.DataFrame(selected_rows)
    candidates = pd.DataFrame(candidate_rows)
    args._clinical_concept_candidates = candidates
    return selected
def find_covariate_columns(num_df: pd.DataFrame) -> Dict[str, str]:
    cols = [c for c in num_df.columns if c != "patient_id"]
    out = {}
    for name, pats in DEMOGRAPHIC_PATTERNS.items():
        col = select_first_matching_column(cols, pats)
        if col is not None:
            out[name] = col
    hr = select_first_matching_column(cols, [r"^hr$", r"heart.*rate", r"心率"])
    if hr is not None:
        out["heart_rate"] = hr
    return out


def clinical_burden_direction(col_or_concept: str, concept_df: Optional[pd.DataFrame] = None) -> int:
    if concept_df is not None and "column" in concept_df.columns:
        m = concept_df[concept_df["column"].astype(str) == str(col_or_concept)]
        if len(m):
            return int(m["orientation"].iloc[0])
    s = str(col_or_concept).lower()
    if re.search(r"(^ef$|lvef|ejection|射血)", s):
        return -1
    if re.search(r"nt.*bnp|bnp|脑钠|nyha|heart.*rate|^hr$|心率|la|lad|lvedd|lvesd|lvdd|左房|左室|iv|lvpw|mr|tr|ar|pr|as|ms|反流|狭窄", s):
        return +1
    return 0


def panel_min_nonmissing(panel_vars: List[str], args) -> int:
    if len(panel_vars) <= 2:
        return 2
    return min(args.min_nonmissing_clinical_vars, len(panel_vars))


def get_ok_vars_by_domain(concept_df: pd.DataFrame, domain: str) -> List[str]:
    d = concept_df[(concept_df["domain"] == domain) & (concept_df["status"] == "ok")].copy()
    if len(d) == 0:
        return []
    d = d.sort_values(["priority", "concept"])
    return d["column"].tolist()



def concept_columns(concept_df: pd.DataFrame, concepts: Sequence[str]) -> List[str]:
    """Return selected usable clinical columns for a list of concept names."""
    out: List[str] = []
    ok = concept_df[concept_df["status"] == "ok"].copy()
    for concept in concepts:
        m = ok[ok["concept"] == concept]
        if len(m):
            col = str(m["column"].iloc[0])
            if col and col not in out:
                out.append(col)
    return out


def build_clinical_panels(concept_df: pd.DataFrame, single_df: pd.DataFrame, args) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """Build a priori clinical-domain panels.

    V4 intentionally removes the manually selected "core" panel. The main analyses
    are domain-specific. An all_clinical panel is still analyzed as a supplementary
    global anchoring space, but it is not used for main-domain figures.
    """
    panels: Dict[str, List[str]] = {}
    rows = []

    for panel_name, concepts in DOMAIN_PANEL_CONCEPTS.items():
        cols = concept_columns(concept_df, concepts)
        if len(cols) >= 2:
            panels[panel_name] = cols
        else:
            log(f"Panel '{panel_name}' not built: found {len(cols)} usable variables ({cols})")

    all_vars: List[str] = []
    for panel_name in DOMAIN_PANEL_CONCEPTS.keys():
        for v in panels.get(panel_name, []):
            if v not in all_vars:
                all_vars.append(v)
    if len(all_vars) >= 2:
        panels["all_clinical"] = all_vars

    for panel, vars_ in panels.items():
        for rank, v in enumerate(vars_, start=1):
            info = concept_df[concept_df["column"] == v]
            rows.append({
                "panel": panel,
                "rank_in_panel": rank,
                "variable": v,
                "concept": info["concept"].iloc[0] if len(info) else "",
                "domain": info["domain"].iloc[0] if len(info) else "",
                "orientation": int(info["orientation"].iloc[0]) if len(info) else clinical_burden_direction(v),
                "is_plotted_domain_panel": panel in DOMAIN_PLOT_PANELS,
            })
    return panels, pd.DataFrame(rows)

def filter_for_panel(X: np.ndarray, clinical: pd.DataFrame, patient_ids: Sequence[str], panel_vars: List[str], args) -> Tuple[np.ndarray, pd.DataFrame, List[str], np.ndarray]:
    min_non = panel_min_nonmissing(panel_vars, args)
    mask = clinical[panel_vars].notna().sum(axis=1).to_numpy() >= min_non
    if int(mask.sum()) < args.min_panel_n:
        raise ValueError(f"Panel retained only {int(mask.sum())} patients, below --min-panel-n={args.min_panel_n}")
    return X[mask], clinical.loc[mask].reset_index(drop=True), [pid for pid, keep in zip(patient_ids, mask) if keep], mask


# =============================================================================
# Modeling helpers: leakage-safe CCA
# =============================================================================

@dataclass
class CCAFoldResult:
    x_scores_train: np.ndarray
    y_scores_train: np.ndarray
    x_scores_test: np.ndarray
    y_scores_test: np.ndarray
    used_clinical_vars: List[str]
    n_pca: int
    signs: List[int]


def choose_n_pca(n_train: int, n_features: int, n_targets: int, requested: int) -> int:
    return int(max(1, min(requested, n_features, n_train - 2)))


def residualize_train_test(A_train: np.ndarray, A_test: np.ndarray, C_train: Optional[np.ndarray], C_test: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if C_train is None or C_test is None or C_train.shape[1] == 0:
        return A_train, A_test
    Ctr = np.asarray(C_train, dtype=float)
    Cte = np.asarray(C_test, dtype=float)
    Dtr = np.column_stack([np.ones(len(Ctr)), Ctr])
    Dte = np.column_stack([np.ones(len(Cte)), Cte])
    beta, *_ = np.linalg.lstsq(Dtr, A_train, rcond=None)
    return A_train - Dtr @ beta, A_test - Dte @ beta


def prepare_covariates(cov_train_raw: Optional[pd.DataFrame], cov_test_raw: Optional[pd.DataFrame]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if cov_train_raw is None or cov_test_raw is None or cov_train_raw.shape[1] == 0:
        return None, None
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    Ctr = imp.fit_transform(cov_train_raw)
    Cte = imp.transform(cov_test_raw)
    Ctr = sc.fit_transform(Ctr)
    Cte = sc.transform(Cte)
    keep = np.nanstd(Ctr, axis=0) > 1e-12
    if keep.sum() == 0:
        return None, None
    return Ctr[:, keep], Cte[:, keep]


def orient_by_clinical_burden(
    xtr: np.ndarray,
    xte: np.ndarray,
    ytr: np.ndarray,
    yte: np.ndarray,
    clinical_train_raw: pd.DataFrame,
    clinical_vars: List[str],
    concept_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    xtr = xtr.copy(); xte = xte.copy(); ytr = ytr.copy(); yte = yte.copy()
    signs = []
    for k in range(xtr.shape[1]):
        evidence = []
        for c in clinical_vars:
            d = clinical_burden_direction(c, concept_df)
            if d == 0:
                continue
            rho, _, n = safe_spearman(xtr[:, k], clinical_train_raw[c].values)
            if n >= 20 and np.isfinite(rho):
                evidence.append(d * rho)
        score = float(np.nanmean(evidence)) if evidence else 0.0
        sign = +1 if score >= 0 else -1
        signs.append(sign)
        if sign < 0:
            xtr[:, k] *= -1; xte[:, k] *= -1; ytr[:, k] *= -1; yte[:, k] *= -1
    return xtr, xte, ytr, yte, signs


def fit_cca_one_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train_raw: pd.DataFrame,
    Y_test_raw: pd.DataFrame,
    clinical_vars: List[str],
    args,
    concept_df: pd.DataFrame,
    cov_train_raw: Optional[pd.DataFrame] = None,
    cov_test_raw: Optional[pd.DataFrame] = None,
) -> CCAFoldResult:
    # Clinical imputation and constant-column filtering are fitted only on training data.
    Ytr_df = Y_train_raw[clinical_vars].copy()
    Yte_df = Y_test_raw[clinical_vars].copy()
    y_imp = SimpleImputer(strategy="median")
    Ytr = y_imp.fit_transform(Ytr_df)
    Yte = y_imp.transform(Yte_df)
    keep = np.nanstd(Ytr, axis=0) > 1e-12
    used_vars = [c for c, k in zip(clinical_vars, keep) if k]
    Ytr = Ytr[:, keep]
    Yte = Yte[:, keep]
    if Ytr.shape[1] < 2:
        raise ValueError("Too few non-constant clinical variables for CCA in this fold.")

    Ctr, Cte = prepare_covariates(cov_train_raw, cov_test_raw)
    Xtr_res, Xte_res = residualize_train_test(np.asarray(X_train, float), np.asarray(X_test, float), Ctr, Cte)
    Ytr_res, Yte_res = residualize_train_test(Ytr, Yte, Ctr, Cte)

    x_scaler = StandardScaler()
    Xtr_s = x_scaler.fit_transform(Xtr_res)
    Xte_s = x_scaler.transform(Xte_res)

    n_pca = choose_n_pca(len(X_train), X_train.shape[1], Ytr_res.shape[1], args.n_pca)
    pca = PCA(n_components=n_pca, random_state=args.seed)
    Xtr_r = pca.fit_transform(Xtr_s)
    Xte_r = pca.transform(Xte_s)

    y_scaler = StandardScaler()
    Ytr_s = y_scaler.fit_transform(Ytr_res)
    Yte_s = y_scaler.transform(Yte_res)

    n_comp = int(min(args.n_components, Xtr_r.shape[1], Ytr_s.shape[1], len(X_train) - 2))
    if n_comp < 1:
        raise ValueError("n_components became <1 in this CCA fold.")

    cca = CCA(n_components=n_comp, max_iter=args.cca_max_iter, scale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cca.fit(Xtr_r, Ytr_s)
        xtr, ytr = cca.transform(Xtr_r, Ytr_s)
        xte, yte = cca.transform(Xte_r, Yte_s)

    xtr = np.atleast_2d(xtr); xte = np.atleast_2d(xte); ytr = np.atleast_2d(ytr); yte = np.atleast_2d(yte)
    if xtr.shape[0] != len(X_train): xtr = xtr.T
    if xte.shape[0] != len(X_test): xte = xte.T
    if ytr.shape[0] != len(X_train): ytr = ytr.T
    if yte.shape[0] != len(X_test): yte = yte.T

    xtr, xte, ytr, yte, signs = orient_by_clinical_burden(
        xtr, xte, ytr, yte, Y_train_raw[used_vars], used_vars, concept_df
    )
    return CCAFoldResult(xtr, ytr, xte, yte, used_vars, n_pca, signs)


def run_oof_cca_panel(
    X: np.ndarray,
    clinical: pd.DataFrame,
    panel_vars: List[str],
    patient_ids: Sequence[str],
    panel_name: str,
    args,
    concept_df: pd.DataFrame,
    covariate_cols: Optional[List[str]] = None,
    adjustment: str = "none",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(clinical)
    kf = KFold(n_splits=min(args.n_splits, n), shuffle=True, random_state=args.seed)
    n_comp = args.n_components
    x_scores = np.full((n, n_comp), np.nan)
    y_scores = np.full((n, n_comp), np.nan)
    folds = np.full(n, -1, dtype=int)
    fold_rows = []

    if verbose:
        log(f"Running OOF CCA: panel={panel_name}, adjustment={adjustment}, n={n}, vars={len(panel_vars)}")

    for fold, (tr, te) in enumerate(kf.split(X), start=1):
        if verbose:
            log(f"  panel={panel_name} fold {fold}/{kf.get_n_splits()}: train={len(tr)}, test={len(te)}")
        if covariate_cols:
            cov_tr = clinical.iloc[tr][covariate_cols]
            cov_te = clinical.iloc[te][covariate_cols]
        else:
            cov_tr = cov_te = None
        res = fit_cca_one_fold(
            X_train=X[tr], X_test=X[te],
            Y_train_raw=clinical.iloc[tr], Y_test_raw=clinical.iloc[te],
            clinical_vars=panel_vars, args=args, concept_df=concept_df,
            cov_train_raw=cov_tr, cov_test_raw=cov_te,
        )
        m = res.x_scores_test.shape[1]
        x_scores[te, :m] = res.x_scores_test[:, :m]
        y_scores[te, :m] = res.y_scores_test[:, :m]
        folds[te] = fold
        for comp in range(m):
            rho, p, nn = safe_spearman(res.x_scores_test[:, comp], res.y_scores_test[:, comp])
            r, rp, _ = safe_pearson(res.x_scores_test[:, comp], res.y_scores_test[:, comp])
            fold_rows.append({
                "panel": panel_name, "adjustment": adjustment, "fold": fold, "component": comp + 1,
                "n_test": nn, "test_spearman_x_y_score": rho, "test_spearman_p": p,
                "test_pearson_x_y_score": r, "test_pearson_p": rp,
                "n_pca": res.n_pca, "used_clinical_vars": ";".join(res.used_clinical_vars),
                "axis_orientation_sign": res.signs[comp] if comp < len(res.signs) else np.nan,
            })

    score_df = pd.DataFrame({"patient_id": list(patient_ids), "fold": folds, "panel": panel_name, "adjustment": adjustment})
    for comp in range(n_comp):
        score_df[f"cca_acoustic_axis{comp+1}"] = x_scores[:, comp]
        score_df[f"cca_clinical_axis{comp+1}"] = y_scores[:, comp]
    return score_df, pd.DataFrame(fold_rows)


def summarize_alignment(score_df: pd.DataFrame, panel_vars: List[str], clinical: pd.DataFrame, args, seed_offset: int = 0) -> pd.DataFrame:
    rows = []
    for comp in range(1, args.n_components + 1):
        xcol = f"cca_acoustic_axis{comp}"
        ycol = f"cca_clinical_axis{comp}"
        if xcol not in score_df.columns or ycol not in score_df.columns:
            continue
        rho, p, n = safe_spearman(score_df[xcol], score_df[ycol])
        r, rp, _ = safe_pearson(score_df[xcol], score_df[ycol])
        ci_l, ci_u = bootstrap_ci_spearman(score_df[xcol], score_df[ycol], args.n_bootstrap, args.seed + 100 + seed_offset + comp)

        # Clinical redundancy: mean squared Pearson correlation between acoustic axis and each clinical variable.
        red_vals = []
        max_abs = np.nan
        for v in panel_vars:
            rr, _, nn = safe_pearson(score_df[xcol], clinical[v].values)
            if nn >= 10 and np.isfinite(rr):
                red_vals.append(rr ** 2)
        if red_vals:
            max_abs = float(np.sqrt(np.nanmax(red_vals)))
        rows.append({
            "panel": score_df["panel"].iloc[0],
            "adjustment": score_df["adjustment"].iloc[0],
            "component": comp,
            "n": n,
            "spearman_acoustic_vs_clinical_axis": rho,
            "spearman_p": p,
            "spearman_ci95_low": ci_l,
            "spearman_ci95_high": ci_u,
            "pearson_acoustic_vs_clinical_axis": r,
            "pearson_p": rp,
            "clinical_redundancy_mean_r2": float(np.nanmean(red_vals)) if red_vals else np.nan,
            "clinical_redundancy_max_abs_r": max_abs,
        })

    # Cumulative redundancy of acoustic axes 1+2 with clinical variables.
    if args.n_components >= 2 and {"cca_acoustic_axis1", "cca_acoustic_axis2"}.issubset(score_df.columns):
        Z = score_df[["cca_acoustic_axis1", "cca_acoustic_axis2"]].to_numpy(dtype=float)
        cum_vals = []
        for v in panel_vars:
            y = clinical[v].to_numpy(dtype=float)
            mask = np.isfinite(y) & np.isfinite(Z).all(axis=1)
            if mask.sum() >= 20 and np.nanstd(y[mask]) > 1e-12:
                try:
                    lr = LinearRegression().fit(Z[mask], y[mask])
                    yhat = lr.predict(Z[mask])
                    cum_vals.append(r2_score(y[mask], yhat))
                except Exception:
                    pass
        if rows:
            rows[0]["clinical_redundancy_axis1_2_mean_r2"] = float(np.nanmean(cum_vals)) if cum_vals else np.nan
    out = pd.DataFrame(rows)
    return out


# =============================================================================
# Single-variable readout
# =============================================================================

def single_variable_readout(X: np.ndarray, clinical: pd.DataFrame, candidate_vars: List[str], args) -> pd.DataFrame:
    log("Experiment: single-variable acoustic readout as complementary signal overview")
    rows = []
    alphas = np.logspace(-3, 3, 13)
    for i, c in enumerate(candidate_vars, start=1):
        y_all = clinical[c].to_numpy(dtype=float)
        mask = np.isfinite(y_all)
        n = int(mask.sum())
        if n < args.min_target_n or np.nanstd(y_all[mask]) < 1e-12:
            rows.append({"variable": c, "n": n, "status": "skipped_too_few_or_constant"})
            continue
        Xc = X[mask]
        yc = y_all[mask]
        n_splits = min(args.n_splits, n)
        preds = np.full(n, np.nan)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        log(f"  [{i}/{len(candidate_vars)}] Ridge readout for {c}: n={n}")
        for tr, te in kf.split(Xc):
            n_pca = choose_n_pca(len(tr), Xc.shape[1], 1, args.n_pca)
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_pca, random_state=args.seed)),
                ("ridge", RidgeCV(alphas=alphas)),
            ])
            pipe.fit(Xc[tr], yc[tr])
            preds[te] = pipe.predict(Xc[te])
        rho, p, nn = safe_spearman(preds, yc)
        r, rp, _ = safe_pearson(preds, yc)
        ci_l, ci_u = bootstrap_ci_spearman(preds, yc, args.n_bootstrap, args.seed + i)
        try:
            r2 = r2_score(yc[np.isfinite(preds)], preds[np.isfinite(preds)])
        except Exception:
            r2 = np.nan
        try:
            mae = mean_absolute_error(yc[np.isfinite(preds)], preds[np.isfinite(preds)])
        except Exception:
            mae = np.nan
        rows.append({
            "variable": c, "n": nn, "status": "ok",
            "spearman_pred_true": rho, "spearman_p": p,
            "spearman_ci95_low": ci_l, "spearman_ci95_high": ci_u,
            "pearson_pred_true": r, "pearson_p": rp, "r2": r2, "mae": mae,
        })
    out = pd.DataFrame(rows)
    if "spearman_p" in out.columns:
        out["spearman_fdr"] = fdr_bh(out["spearman_p"].values)
    return out


# =============================================================================
# Axis interpretation, fold stability, bootstrap stability, gradients
# =============================================================================

def axis_clinical_associations(score_df: pd.DataFrame, clinical: pd.DataFrame, panel_vars: List[str], args, concept_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    fold_rows = []
    for comp in range(1, args.n_components + 1):
        xcol = f"cca_acoustic_axis{comp}"
        if xcol not in score_df.columns:
            continue
        for j, v in enumerate(panel_vars):
            rho, p, n = safe_spearman(score_df[xcol], clinical[v].values)
            ci_l, ci_u = bootstrap_ci_spearman(score_df[xcol], clinical[v].values, args.n_bootstrap, args.seed + 1000 + comp * 100 + j)
            rows.append({
                "panel": score_df["panel"].iloc[0], "adjustment": score_df["adjustment"].iloc[0],
                "component": comp, "variable": v, "n": n,
                "spearman_axis_variable": rho, "spearman_p": p,
                "spearman_ci95_low": ci_l, "spearman_ci95_high": ci_u,
                "burden_orientation": clinical_burden_direction(v, concept_df),
                "oriented_spearman": rho * clinical_burden_direction(v, concept_df) if np.isfinite(rho) else np.nan,
            })
            for fold in sorted(pd.unique(score_df["fold"])):
                m = score_df["fold"].to_numpy() == fold
                rr, pp, nn = safe_spearman(score_df.loc[m, xcol].values, clinical.loc[m, v].values)
                fold_rows.append({
                    "panel": score_df["panel"].iloc[0], "adjustment": score_df["adjustment"].iloc[0],
                    "component": comp, "variable": v, "fold": int(fold), "n": nn,
                    "fold_spearman_axis_variable": rr, "fold_spearman_p": pp,
                    "expected_burden_orientation": clinical_burden_direction(v, concept_df),
                    "fold_oriented_spearman": rr * clinical_burden_direction(v, concept_df) if np.isfinite(rr) else np.nan,
                })
    assoc = pd.DataFrame(rows)
    if len(assoc):
        assoc["spearman_fdr_within_panel_component"] = np.nan
        for (_, _, comp), idx in assoc.groupby(["panel", "adjustment", "component"]).groups.items():
            assoc.loc[idx, "spearman_fdr_within_panel_component"] = fdr_bh(assoc.loc[idx, "spearman_p"].values)
    fold_assoc = pd.DataFrame(fold_rows)

    stability_rows = []
    if len(fold_assoc):
        for (panel, adjustment, comp, var), sub in fold_assoc.groupby(["panel", "adjustment", "component", "variable"]):
            vals = sub["fold_spearman_axis_variable"].dropna().to_numpy(dtype=float)
            oriented = sub["fold_oriented_spearman"].dropna().to_numpy(dtype=float)
            stability_rows.append({
                "panel": panel, "adjustment": adjustment, "component": comp, "variable": var,
                "n_folds_with_valid_rho": len(vals),
                "mean_fold_spearman": float(np.nanmean(vals)) if len(vals) else np.nan,
                "sd_fold_spearman": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else np.nan,
                "same_sign_rate_raw": float(np.mean(np.sign(vals) == np.sign(np.nanmean(vals)))) if len(vals) else np.nan,
                "expected_direction_positive_rate": float(np.mean(oriented > 0)) if len(oriented) else np.nan,
            })
    return assoc, fold_assoc, pd.DataFrame(stability_rows)


def fit_cca_full_scores(X: np.ndarray, clinical: pd.DataFrame, panel_vars: List[str], args, concept_df: pd.DataFrame) -> np.ndarray:
    Y = SimpleImputer(strategy="median").fit_transform(clinical[panel_vars])
    keep = np.nanstd(Y, axis=0) > 1e-12
    panel_vars = [v for v, k in zip(panel_vars, keep) if k]
    Y = Y[:, keep]
    Xs = StandardScaler().fit_transform(X)
    n_pca = choose_n_pca(len(X), X.shape[1], Y.shape[1], args.n_pca)
    Xr = PCA(n_components=n_pca, random_state=args.seed).fit_transform(Xs)
    Ys = StandardScaler().fit_transform(Y)
    n_comp = int(min(args.n_components, Xr.shape[1], Ys.shape[1], len(X) - 2))
    cca = CCA(n_components=n_comp, max_iter=args.cca_max_iter, scale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cca.fit(Xr, Ys)
        xs, ys = cca.transform(Xr, Ys)
    xs = np.atleast_2d(xs)
    if xs.shape[0] != len(X):
        xs = xs.T
    # Orient on the fitting sample for stability summaries.
    xs, _, ys, _, _ = orient_by_clinical_burden(xs, xs.copy(), ys, ys.copy(), clinical[panel_vars], panel_vars, concept_df)
    return xs


def bootstrap_loading_stability(X: np.ndarray, clinical: pd.DataFrame, panel_vars: List[str], panel_name: str, args, concept_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if args.n_loading_bootstrap <= 0:
        return pd.DataFrame(), pd.DataFrame()
    log(f"Bootstrap loading stability: panel={panel_name}, B={args.n_loading_bootstrap}")
    rng = np.random.default_rng(args.seed + 404)
    rows = []
    n = len(clinical)
    for b in range(args.n_loading_bootstrap):
        if (b + 1) % max(1, args.progress_every) == 0 or b == 0:
            log(f"  loading bootstrap {b+1}/{args.n_loading_bootstrap} for panel={panel_name}")
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]
        Cb = clinical.iloc[idx].reset_index(drop=True)
        try:
            xs = fit_cca_full_scores(Xb, Cb, panel_vars, args, concept_df)
        except Exception:
            continue
        for comp in range(min(args.n_components, xs.shape[1])):
            abs_order = []
            for v in panel_vars:
                rho, p, nn = safe_spearman(xs[:, comp], Cb[v].values)
                rows.append({
                    "panel": panel_name, "bootstrap": b + 1, "component": comp + 1,
                    "variable": v, "n": nn, "spearman_axis_variable": rho,
                    "oriented_spearman": rho * clinical_burden_direction(v, concept_df) if np.isfinite(rho) else np.nan,
                })
                abs_order.append((v, abs(rho) if np.isfinite(rho) else -np.inf))
            ranked = sorted(abs_order, key=lambda z: z[1], reverse=True)
            rank_map = {v: r + 1 for r, (v, _) in enumerate(ranked)}
            # Store ranks by updating existing rows for this bootstrap/component.
            for r in rows[-len(panel_vars):]:
                r["abs_loading_rank"] = rank_map.get(r["variable"], np.nan)
    boot = pd.DataFrame(rows)
    if len(boot) == 0:
        return boot, pd.DataFrame()
    summary_rows = []
    for (panel, comp, var), sub in boot.groupby(["panel", "component", "variable"]):
        vals = sub["spearman_axis_variable"].dropna().to_numpy(dtype=float)
        oriented = sub["oriented_spearman"].dropna().to_numpy(dtype=float)
        ranks = sub["abs_loading_rank"].dropna().to_numpy(dtype=float)
        summary_rows.append({
            "panel": panel, "component": comp, "variable": var,
            "n_boot_valid": len(vals),
            "mean_spearman": float(np.mean(vals)) if len(vals) else np.nan,
            "ci95_low": float(np.percentile(vals, 2.5)) if len(vals) else np.nan,
            "ci95_high": float(np.percentile(vals, 97.5)) if len(vals) else np.nan,
            "positive_rate_raw": float(np.mean(vals > 0)) if len(vals) else np.nan,
            "expected_direction_positive_rate": float(np.mean(oriented > 0)) if len(oriented) else np.nan,
            "median_abs_loading_rank": float(np.median(ranks)) if len(ranks) else np.nan,
        })
    return boot, pd.DataFrame(summary_rows)



def variable_kind(variable: str, concept_df: Optional[pd.DataFrame] = None) -> str:
    """Classify clinical variable display type for gradient plots."""
    concept = ""
    domain = ""
    if concept_df is not None and len(concept_df):
        m = concept_df[concept_df["column"].astype(str) == str(variable)]
        if len(m):
            concept = str(m["concept"].iloc[0])
            domain = str(m["domain"].iloc[0])
    if concept == "NYHA":
        return "nyha_ordinal"
    if domain == "valve" or concept in {"MR", "TR", "AR", "PR", "AS", "MS"}:
        return "valve_ordinal"
    return "continuous"


def gradient_display_metric(variable: str, concept_df: Optional[pd.DataFrame] = None) -> Tuple[str, str, str]:
    """Return (value_column, label, unit_note) used for clinical-gradient plotting.

    Continuous variables are shown by median. Ordinal variables are shown by a
    clinically interpretable proportion instead of median, because medians are
    often zero in sparse valve-grade variables.
    """
    kind = variable_kind(variable, concept_df)
    if kind == "nyha_ordinal":
        return "prop_ge_3", "Proportion with NYHA ≥3", "proportion"
    if kind == "valve_ordinal":
        return "prop_ge_1", "Proportion with grade ≥1", "proportion"
    return "median", "Median", "original unit"


def clinical_gradient_by_axis(score_df: pd.DataFrame, clinical: pd.DataFrame, panel_vars: List[str], args, concept_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    rows = []
    for comp in range(1, args.n_components + 1):
        xcol = f"cca_acoustic_axis{comp}"
        if xcol not in score_df.columns:
            continue
        scores = score_df[xcol].to_numpy(dtype=float)
        valid = np.isfinite(scores)
        if valid.sum() < 20:
            continue
        ranks = pd.Series(scores[valid]).rank(method="first")
        labels = [f"Q{i+1}" for i in range(args.n_axis_groups)]
        groups = pd.qcut(ranks, q=args.n_axis_groups, labels=labels)
        tmp = clinical.loc[valid, ["patient_id"] + panel_vars].copy()
        tmp["axis_score"] = scores[valid]
        tmp["axis_group"] = groups.to_numpy()
        for v in panel_vars:
            rho, p, n = safe_spearman(tmp["axis_score"].values, tmp[v].values)
            kind = variable_kind(v, concept_df)
            display_col, display_label, display_unit = gradient_display_metric(v, concept_df)
            for g, sub in tmp.groupby("axis_group", observed=False):
                vv = sub[v].astype(float)
                clean = vv.dropna()
                n_non = int(clean.shape[0])
                if n_non:
                    prop_ge_1 = float((clean >= 1).mean())
                    prop_ge_2 = float((clean >= 2).mean())
                    prop_ge_3 = float((clean >= 3).mean())
                    median = float(np.nanmedian(clean))
                    mean = float(np.nanmean(clean))
                    q25 = float(np.nanpercentile(clean, 25))
                    q75 = float(np.nanpercentile(clean, 75))
                else:
                    prop_ge_1 = prop_ge_2 = prop_ge_3 = median = mean = q25 = q75 = np.nan
                display_value = {
                    "median": median,
                    "prop_ge_1": prop_ge_1,
                    "prop_ge_2": prop_ge_2,
                    "prop_ge_3": prop_ge_3,
                }.get(display_col, median)
                rows.append({
                    "panel": score_df["panel"].iloc[0], "adjustment": score_df["adjustment"].iloc[0],
                    "component": comp, "variable": v, "axis_group": str(g),
                    "n": n_non,
                    "variable_kind": kind,
                    "median": median,
                    "mean": mean,
                    "q25": q25,
                    "q75": q75,
                    "prop_ge_1": prop_ge_1,
                    "prop_ge_2": prop_ge_2,
                    "prop_ge_3": prop_ge_3,
                    "display_metric": display_col,
                    "display_metric_label": display_label,
                    "display_unit": display_unit,
                    "display_value": display_value,
                    "axis_variable_spearman": rho,
                    "axis_variable_p": p,
                    "axis_variable_n": n,
                })
    return pd.DataFrame(rows)

# =============================================================================
# Permutation

# =============================================================================
# Permutation / negative controls
# =============================================================================

def permutation_test_panel(X: np.ndarray, clinical: pd.DataFrame, panel_vars: List[str], patient_ids: Sequence[str], panel_name: str, observed_rho: float, args, concept_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    rng = np.random.default_rng(args.seed + 909)
    rows = []
    if args.n_permutations <= 0:
        return pd.DataFrame(), {"panel": panel_name, "component": 1, "permutation_p_abs": np.nan}
    log(f"Permutation test: panel={panel_name}, B={args.n_permutations}")
    for b in range(args.n_permutations):
        if (b + 1) % max(1, args.progress_every) == 0 or b == 0:
            log(f"  permutation {b+1}/{args.n_permutations} for panel={panel_name}")
        idx = rng.permutation(len(clinical))
        cperm = clinical.iloc[idx].reset_index(drop=True)
        try:
            sdf, _ = run_oof_cca_panel(X, cperm, panel_vars, patient_ids, panel_name, args, concept_df, verbose=False)
            rho, p, n = safe_spearman(sdf["cca_acoustic_axis1"], sdf["cca_clinical_axis1"])
        except Exception:
            rho, p, n = np.nan, np.nan, 0
        rows.append({"panel": panel_name, "control_type": "patient_label_permutation", "iteration": b + 1, "component": 1, "spearman": rho, "p": p, "n": n})
    vals = pd.Series([r["spearman"] for r in rows]).dropna().to_numpy(dtype=float)
    perm_p = (1 + np.sum(np.abs(vals) >= abs(observed_rho))) / (1 + len(vals)) if len(vals) and np.isfinite(observed_rho) else np.nan
    summary = {
        "panel": panel_name, "component": 1, "control_type": "patient_label_permutation",
        "observed_spearman": observed_rho, "control_n": len(vals),
        "control_mean": float(np.mean(vals)) if len(vals) else np.nan,
        "control_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
        "control_p2_5": float(np.percentile(vals, 2.5)) if len(vals) else np.nan,
        "control_p97_5": float(np.percentile(vals, 97.5)) if len(vals) else np.nan,
        "empirical_p_abs_ge_observed": perm_p,
    }
    return pd.DataFrame(rows), summary


def random_embedding_control_core(X: np.ndarray, clinical: pd.DataFrame, panel_vars: List[str], patient_ids: Sequence[str], panel_name: str, observed_rho: float, args, concept_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    rng = np.random.default_rng(args.seed + 1009)
    rows = []
    if args.n_random_controls <= 0:
        return pd.DataFrame(), {}
    log(f"Random-embedding control: panel={panel_name}, B={args.n_random_controls}")
    for b in range(args.n_random_controls):
        log(f"  random embedding {b+1}/{args.n_random_controls}")
        Xrand = rng.normal(size=X.shape).astype(np.float32)
        try:
            sdf, _ = run_oof_cca_panel(Xrand, clinical, panel_vars, patient_ids, panel_name, args, concept_df, verbose=False)
            rho, p, n = safe_spearman(sdf["cca_acoustic_axis1"], sdf["cca_clinical_axis1"])
        except Exception:
            rho, p, n = np.nan, np.nan, 0
        rows.append({"panel": panel_name, "control_type": "random_embedding", "iteration": b + 1, "component": 1, "spearman": rho, "p": p, "n": n})
    vals = pd.Series([r["spearman"] for r in rows]).dropna().to_numpy(dtype=float)
    p_emp = (1 + np.sum(np.abs(vals) >= abs(observed_rho))) / (1 + len(vals)) if len(vals) and np.isfinite(observed_rho) else np.nan
    summary = {
        "panel": panel_name, "component": 1, "control_type": "random_embedding",
        "observed_spearman": observed_rho, "control_n": len(vals),
        "control_mean": float(np.mean(vals)) if len(vals) else np.nan,
        "control_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
        "control_p2_5": float(np.percentile(vals, 2.5)) if len(vals) else np.nan,
        "control_p97_5": float(np.percentile(vals, 97.5)) if len(vals) else np.nan,
        "empirical_p_abs_ge_observed": p_emp,
    }
    return pd.DataFrame(rows), summary


# =============================================================================
# Endpoint validation
# =============================================================================


def find_endpoint_columns(clinical: pd.DataFrame) -> Dict[str, str]:
    cols = [c for c in clinical.columns if c != "patient_id"]
    out = {}
    # Prefer exact cleaned-table endpoint columns.
    for endpoint, col in EXACT_ENDPOINT_COLUMNS.items():
        if col in cols:
            out[endpoint] = col
    # Fall back to regex matching for compatibility with other clinical tables.
    if "EF_lt_40" not in out:
        ef = select_first_matching_column(cols, [r"^ef$", r"lvef", r"ejection", r"射血"])
        if ef: out["EF_lt_40"] = ef
    if "NTproBNP_ge_300" not in out:
        nt = select_first_matching_column(cols, [r"nt[-_ ]?pro[-_ ]?bnp", r"ntprobnp", r"nt.*bnp", r"脑钠", r"^bnp$"])
        if nt: out["NTproBNP_ge_300"] = nt
    if "NYHA_ge_3" not in out:
        nyha = select_first_matching_column(cols, [r"nyha"])
        if nyha: out["NYHA_ge_3"] = nyha
    if "LA_ge_40" not in out:
        la = select_first_matching_column(cols, [r"^la$", r"^lad$", r"la_mm", r"left.*atri", r"左房"])
        if la: out["LA_ge_40"] = la
    if "LVEDD_dilated" not in out:
        lvedd = select_first_matching_column(cols, [r"lvedd", r"lvdd", r"left.*vent.*diast", r"左室.*舒张"])
        if lvedd: out["LVEDD_dilated"] = lvedd
    return out


def make_endpoint_series(name: str, col: str, clinical: pd.DataFrame) -> Tuple[pd.Series, float, str]:
    v = clinical[col].astype(float)
    y = pd.Series(np.nan, index=clinical.index, dtype=float)
    if name == "EF_lt_40":
        threshold = 0.40 if np.nanmax(v.values) <= 1.5 else 40.0
        y.loc[v.notna()] = (v.loc[v.notna()] < threshold).astype(float)
        return y, threshold, f"{col} < {threshold:g}"
    if name == "NTproBNP_ge_300":
        maxv = np.nanmax(v.values)
        lc = str(col).lower()
        if "log10" in lc or ("log" in lc and maxv <= 5.0):
            threshold = float(np.log10(300.0))
        elif "log" in lc or "ln" in lc or maxv <= 20.0:
            # Cleaned clinical tables often store NT-proBNP as ln(1+x); e.g., ln(301)=5.707.
            threshold = float(np.log1p(300.0))
        else:
            threshold = 300.0
        y.loc[v.notna()] = (v.loc[v.notna()] >= threshold).astype(float)
        return y, threshold, f"{col} >= {threshold:g} (corresponds to NT-proBNP >=300 if log-transformed)"
    if name == "NYHA_ge_3":
        threshold = 3.0
        y.loc[v.notna()] = (v.loc[v.notna()] >= threshold).astype(float)
        return y, threshold, f"{col} >= 3"
    if name == "LA_ge_40":
        threshold = 40.0
        y.loc[v.notna()] = (v.loc[v.notna()] >= threshold).astype(float)
        return y, threshold, f"{col} >= 40 mm (exploratory LA enlargement endpoint using available LA diameter)"
    if name == "LVEDD_dilated":
        # ASE/EACVI-style sex-specific upper reference limits for LV end-diastolic dimension
        # are approximately 58 mm for men and 52 mm for women. If sex is unavailable,
        # use a conservative single threshold of 55 mm and mark it in the rule.
        sex_col = EXACT_COVARIATE_COLUMNS.get("sex", "sex_male")
        if sex_col in clinical.columns:
            sex = clinical[sex_col].astype(float)
            valid = v.notna() & sex.notna()
            thr = pd.Series(np.where(sex >= 0.5, 58.0, 52.0), index=clinical.index)
            y.loc[valid] = (v.loc[valid] > thr.loc[valid]).astype(float)
            return y, np.nan, f"{col} >58 mm for male or >52 mm for female (sex-specific exploratory LV dilation endpoint)"
        threshold = 55.0
        y.loc[v.notna()] = (v.loc[v.notna()] >= threshold).astype(float)
        return y, threshold, f"{col} >=55 mm (exploratory LV dilation endpoint; sex unavailable)"
    raise ValueError(name)


def endpoint_exclude_vars(endpoint_name: str, endpoint_col: str, panel_vars: List[str]) -> List[str]:
    pats = []
    if endpoint_name.startswith("EF"):
        pats = [r"^ef$", r"lvef", r"ejection", r"射血"]
    elif endpoint_name.startswith("NTproBNP"):
        pats = [r"nt.*bnp", r"ntprobnp", r"bnp", r"脑钠"]
    elif endpoint_name.startswith("NYHA"):
        pats = [r"nyha"]
    elif endpoint_name.startswith("LA"):
        pats = [r"^la$", r"^lad$", r"la_mm", r"left.*atri", r"左房"]
    elif endpoint_name.startswith("LVEDD"):
        pats = [r"lvedd", r"lvdd", r"left.*vent.*diast", r"左室.*舒张"]
    out = []
    for v in panel_vars:
        if v == endpoint_col or any(re.search(p, str(v), flags=re.I) for p in pats):
            continue
        out.append(v)
    return out

def endpoint_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(prob)
    y = y_true[mask].astype(int)
    p = prob[mask]
    out = {"n": int(len(y)), "n_positive": int(y.sum()) if len(y) else 0, "positive_rate": float(y.mean()) if len(y) else np.nan}
    if len(y) == 0 or len(np.unique(y)) < 2:
        out.update({"auroc": np.nan, "accuracy": np.nan, "balanced_accuracy": np.nan, "sensitivity": np.nan, "specificity": np.nan})
        return out
    pred = (p >= 0.5).astype(int)
    out["auroc"] = float(roc_auc_score(y, p))
    out["accuracy"] = float(accuracy_score(y, pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else np.nan
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else np.nan
    return out



def run_endpoint_validation_all_clinical(X: np.ndarray, clinical: pd.DataFrame, patient_ids: Sequence[str], all_clinical_vars: List[str], args, concept_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate clinical readability using all_clinical CCA axes 1+2.

    For each endpoint, the endpoint's own source variable is removed from the
    anchoring panel before CCA is fitted inside each training fold.
    """
    log("Endpoint validation using all_clinical CCA acoustic axes 1+2 and a simple logistic classifier")
    endpoints = find_endpoint_columns(clinical)
    summary_rows = []
    pred_rows = []
    info_rows = []
    for endpoint_name, endpoint_col in endpoints.items():
        y_series, threshold, rule = make_endpoint_series(endpoint_name, endpoint_col, clinical)
        valid = y_series.notna() & clinical[endpoint_col].notna()
        y_all = y_series[valid].astype(int).to_numpy()
        info = {
            "endpoint": endpoint_name, "source_column": endpoint_col, "rule": rule, "threshold": threshold,
            "n": int(valid.sum()), "n_positive": int(y_all.sum()) if len(y_all) else 0,
            "positive_rate": float(y_all.mean()) if len(y_all) else np.nan,
            "status": "candidate",
        }
        min_class = int(min(y_all.sum(), len(y_all) - y_all.sum())) if len(y_all) else 0
        if len(y_all) < args.min_endpoint_n or len(np.unique(y_all)) < 2 or min_class < args.min_endpoint_class_n:
            info["status"] = "skipped_too_few_for_oof_auc"
            info["min_class_count"] = min_class
            info["min_endpoint_class_n_used"] = args.min_endpoint_class_n
            info_rows.append(info)
            log(f"  Skip {endpoint_name}: {info['status']}, n={len(y_all)}, pos={int(y_all.sum()) if len(y_all) else 0}, min_class={min_class}")
            continue
        info["status"] = "ok"
        info["min_class_count"] = min_class
        info["min_endpoint_class_n_used"] = args.min_endpoint_class_n
        if min_class < max(20, args.n_splits * 2):
            info["warning"] = "strong_class_imbalance_or_small_minority_class; AUROC is reported, threshold metrics should be interpreted cautiously"
        else:
            info["warning"] = ""
        info_rows.append(info)

        X_e = X[valid.values]
        clinical_e = clinical.loc[valid].reset_index(drop=True)
        patient_ids_e = [pid for pid, keep in zip(patient_ids, valid.values) if keep]

        if args.leave_endpoint_out:
            panel_vars = endpoint_exclude_vars(endpoint_name, endpoint_col, all_clinical_vars)
            panel_type = "all_clinical_leave_endpoint_out"
            if len(panel_vars) < 2:
                panel_vars = all_clinical_vars
                panel_type = "all_clinical_full_fallback"
        else:
            panel_vars = all_clinical_vars
            panel_type = "all_clinical_full"

        pos = int(y_all.sum()); neg = int(len(y_all) - pos)
        n_splits = min(args.n_splits, pos, neg, len(y_all))
        if n_splits < 3:
            continue
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        prob = np.full(len(y_all), np.nan)
        folds = np.full(len(y_all), -1)
        log(f"  Endpoint {endpoint_name}: n={len(y_all)}, pos={pos}, panel={panel_type}, axis features=1+2")
        for fold, (tr, te) in enumerate(skf.split(X_e, y_all), start=1):
            res = fit_cca_one_fold(
                X_train=X_e[tr], X_test=X_e[te],
                Y_train_raw=clinical_e.iloc[tr], Y_test_raw=clinical_e.iloc[te],
                clinical_vars=panel_vars, args=args, concept_df=concept_df,
            )
            k = min(2, res.x_scores_train.shape[1])
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
            ])
            clf.fit(res.x_scores_train[:, :k], y_all[tr])
            prob[te] = clf.predict_proba(res.x_scores_test[:, :k])[:, 1]
            folds[te] = fold
        m = endpoint_metrics(y_all, prob)
        auc_l, auc_u = bootstrap_ci_auc(y_all, prob, args.n_bootstrap, args.seed + 8000)
        summary_rows.append({
            "endpoint": endpoint_name, "source_column": endpoint_col, "rule": rule,
            "panel": "all_clinical", "panel_type": panel_type, "n_axis_features": 2,
            "clinical_panel_vars": ";".join(panel_vars), **m,
            "auroc_ci95_low": auc_l, "auroc_ci95_high": auc_u,
        })
        for pid, y, p, f in zip(patient_ids_e, y_all, prob, folds):
            pred_rows.append({
                "patient_id": pid, "endpoint": endpoint_name, "panel_type": panel_type,
                "n_axis_features": 2, "fold": int(f), "y_true": int(y), "y_prob": float(p) if np.isfinite(p) else np.nan,
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(pred_rows), pd.DataFrame(info_rows)

# =============================================================================
# Confounder

# =============================================================================
# Confounder and position controls
# =============================================================================

def remove_cols_by_patterns(vars_: List[str], patterns: Sequence[str]) -> List[str]:
    out = []
    for v in vars_:
        if any(re.search(p, str(v), flags=re.I) for p in patterns):
            continue
        out.append(v)
    return out



def run_confounder_controls(X: np.ndarray, clinical: pd.DataFrame, patient_ids: Sequence[str], all_clinical_vars: List[str], cov_cols: Dict[str, str], args, concept_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log("Confounder controls for the all_clinical panel")
    rows = []
    assoc_rows = []
    analyses = []
    age_sex = [cov_cols[k] for k in ["age", "sex"] if k in cov_cols]
    if len(age_sex) > 0:
        analyses.append(("age_sex_residualized", all_clinical_vars, age_sex))
    age_sex_hr = [cov_cols[k] for k in ["age", "sex", "heart_rate"] if k in cov_cols]
    if len(age_sex_hr) > 0:
        analyses.append(("age_sex_heart_rate_residualized", all_clinical_vars, age_sex_hr))
    for label, pvars, cvars in analyses:
        try:
            Xp, Cp, pids_p, _ = filter_for_panel(X, clinical, patient_ids, pvars, args)
            sdf, fdf = run_oof_cca_panel(Xp, Cp, pvars, pids_p, "all_clinical", args, concept_df, covariate_cols=cvars, adjustment=label, verbose=True)
            summ = summarize_alignment(sdf, pvars, Cp, args, seed_offset=2000)
            summ["covariates_used"] = ";".join(cvars)
            rows.append(summ)
            assoc, _, _ = axis_clinical_associations(sdf, Cp, pvars, args, concept_df)
            assoc["covariates_used"] = ";".join(cvars)
            assoc_rows.append(assoc)
        except Exception as e:
            log(f"  Confounder control skipped/failed: {label}: {e}")
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
            pd.concat(assoc_rows, ignore_index=True) if assoc_rows else pd.DataFrame())

def load_position_embeddings_for_lopo(args, target_patient_ids: Sequence[str]) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    emb_dir = Path(args.embedding_dir)
    if args.position_embedding_npy:
        pos_emb_path = Path(args.position_embedding_npy)
    else:
        pos_emb_path = find_first_existing(emb_dir, ["position_embeddings.npy", "position_embedding.npy", "position_embeds.npy"])
        if pos_emb_path is None:
            matches = sorted(emb_dir.glob("*position*embed*.npy"))
            pos_emb_path = matches[0] if matches else None
    if args.position_meta_csv:
        pos_meta_path = Path(args.position_meta_csv)
    else:
        pos_meta_path = find_first_existing(emb_dir, ["position_meta.csv", "position_metadata.csv", "position_embeddings_meta.csv"])
        if pos_meta_path is None:
            matches = sorted(emb_dir.glob("*position*meta*.csv"))
            pos_meta_path = matches[0] if matches else None
    if pos_emb_path is None or pos_meta_path is None or not pos_emb_path.exists() or not pos_meta_path.exists():
        log("Position embeddings/meta not found; leave-one-position-out control will be skipped.")
        return None, None
    Xpos = np.load(pos_emb_path).astype(np.float32)
    meta = pd.read_csv(pos_meta_path)
    if len(meta) != Xpos.shape[0]:
        log("Position meta rows do not match position embedding rows; skipping leave-one-position-out control.")
        return None, None
    pid_col = infer_patient_id_col(meta, explicit=args.position_patient_id_col, context="position meta")
    pos_col = args.position_col if args.position_col and args.position_col in meta.columns else None
    if pos_col is None:
        for c in ["position", "pos", "auscultation_position", "site", "location", "部位"]:
            if c in meta.columns:
                pos_col = c
                break
    if pos_col is None:
        log(f"Cannot infer position column from position meta columns={list(meta.columns)}; skipping LOPO.")
        return None, None
    meta = meta.copy()
    meta["patient_id"] = meta[pid_col].map(normalize_patient_id)
    meta["position"] = meta[pos_col].astype(str).str.upper().str.strip()
    meta["row"] = np.arange(len(meta))
    log(f"Loaded position embeddings: {pos_emb_path}, shape={Xpos.shape}; meta={pos_meta_path}; position column={pos_col}")
    return Xpos, meta


def build_lopo_embedding(Xpos: np.ndarray, pos_meta: pd.DataFrame, leave_pos: str, patient_ids: Sequence[str], positions: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    rows = []
    pids = []
    for pid in patient_ids:
        parts = []
        ok = True
        for pos in positions:
            if pos == leave_pos:
                continue
            m = pos_meta[(pos_meta["patient_id"] == pid) & (pos_meta["position"] == pos)]
            if len(m) == 0:
                ok = False
                break
            # If duplicates exist, average them for safety.
            emb = Xpos[m["row"].to_numpy(dtype=int)].mean(axis=0)
            parts.append(emb)
        if ok and parts:
            rows.append(np.concatenate(parts, axis=0))
            pids.append(pid)
    if len(rows) == 0:
        return np.empty((0, 0)), []
    return np.vstack(rows).astype(np.float32), pids



def run_leave_one_position_control(X: np.ndarray, clinical: pd.DataFrame, patient_ids: Sequence[str], all_clinical_vars: List[str], args, concept_df: pd.DataFrame) -> pd.DataFrame:
    if not args.run_leave_one_position_out:
        return pd.DataFrame()
    Xpos, pos_meta = load_position_embeddings_for_lopo(args, patient_ids)
    if Xpos is None or pos_meta is None:
        return pd.DataFrame()
    positions = [p.strip().upper() for p in args.positions.split(",") if p.strip()]
    clinical_by_pid = clinical.set_index("patient_id", drop=False)
    rows = []
    for leave in positions:
        Xl, pids_l = build_lopo_embedding(Xpos, pos_meta, leave, patient_ids, positions)
        if len(pids_l) < args.min_panel_n:
            log(f"  LOPO skip leave={leave}: only {len(pids_l)} patients")
            continue
        Cl = clinical_by_pid.loc[pids_l].reset_index(drop=True)
        try:
            Xf, Cf, pids_f, _ = filter_for_panel(Xl, Cl, pids_l, all_clinical_vars, args)
            sdf, _ = run_oof_cca_panel(Xf, Cf, all_clinical_vars, pids_f, f"all_clinical_leave_{leave}", args, concept_df, verbose=True)
            summ = summarize_alignment(sdf, all_clinical_vars, Cf, args, seed_offset=3000)
            summ["left_out_position"] = leave
            summ["n_positions_used"] = len(positions) - 1
            rows.append(summ)
        except Exception as e:
            log(f"  LOPO failed leave={leave}: {e}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# =============================================================================
# Plotting

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



def variable_domain_for_plot(variable: str, concept_df: Optional[pd.DataFrame] = None) -> str:
    if concept_df is not None and len(concept_df):
        m = concept_df[concept_df["column"].astype(str) == str(variable)]
        if len(m):
            concept = str(m["concept"].iloc[0])
            domain = str(m["domain"].iloc[0])
            if domain == "function_burden":
                return "Functional/HF burden"
            if domain == "structure":
                return "Structural remodeling"
            if concept in {"MR", "TR", "AR", "PR"}:
                return "Valvular regurgitation"
            if concept in {"AS", "MS"}:
                return "Valvular stenosis"
            if domain == "valve":
                return "Valvular"
    return "Other"


def plot_single_variable_readout(df: pd.DataFrame, out_path: Path, concept_df: Optional[pd.DataFrame] = None, top_n: int = 20) -> None:
    if len(df) == 0 or "spearman_pred_true" not in df.columns:
        return
    d = df[df.get("status", "ok") == "ok"].copy()
    d = d[np.isfinite(d["spearman_pred_true"])]
    if len(d) == 0:
        return
    d["abs_rho"] = d["spearman_pred_true"].abs()
    d["domain_label"] = d["variable"].map(lambda v: variable_domain_for_plot(v, concept_df))
    d = d.sort_values("abs_rho", ascending=False).head(top_n).sort_values("spearman_pred_true")
    domain_order = ["Functional/HF burden", "Structural remodeling", "Valvular regurgitation", "Valvular stenosis", "Other"]
    palette = dict(zip(domain_order, sns.color_palette("Set2", n_colors=len(domain_order)) if HAS_SEABORN else plt.cm.tab10(np.linspace(0, 1, len(domain_order)))))
    plt.figure(figsize=(9.8, max(5, 0.42 * len(d))))
    ax = plt.gca()
    y = np.arange(len(d))
    colors = [palette.get(lbl, "0.5") for lbl in d["domain_label"]]
    ax.hlines(y, 0, d["spearman_pred_true"].to_numpy(float), color=colors, lw=3, alpha=0.85)
    ax.scatter(d["spearman_pred_true"], y, c=colors, s=72, edgecolor="black", linewidth=0.5, zorder=3)
    if {"spearman_ci95_low", "spearman_ci95_high"}.issubset(d.columns):
        x = d["spearman_pred_true"].to_numpy(float)
        lo = d["spearman_ci95_low"].to_numpy(float)
        hi = d["spearman_ci95_high"].to_numpy(float)
        if np.isfinite(lo).any() and np.isfinite(hi).any():
            ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt="none", ecolor="black", elinewidth=1, capsize=2, zorder=2)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["variable"])
    ax.set_xlabel("Out-of-fold Spearman correlation\n(predicted clinical variable vs. observed variable)")
    ax.set_ylabel("")
    ax.set_title("Single-variable acoustic readout overview")
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[k], markeredgecolor='black', markersize=8, label=k) for k in domain_order if k in set(d["domain_label"])]
    if handles:
        ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=10)
    savefig(out_path)


def plot_panel_alignment(df: pd.DataFrame, out_path: Path) -> None:
    if len(df) == 0:
        return
    d = df[df["adjustment"] == "none"].copy() if "adjustment" in df.columns else df.copy()
    d = d[d["panel"].isin(DOMAIN_PLOT_PANELS)].copy()
    d = d[np.isfinite(d["spearman_acoustic_vs_clinical_axis"])]
    if len(d) == 0:
        return
    d["label"] = d["panel"] + " axis " + d["component"].astype(str)
    d["panel_order"] = d["panel"].map({p: i for i, p in enumerate(DOMAIN_PLOT_PANELS)})
    d = d.sort_values(["panel_order", "component"])
    plt.figure(figsize=(10, max(5, 0.48 * len(d))))
    ax = plt.gca()
    y = np.arange(len(d))
    x = d["spearman_acoustic_vs_clinical_axis"].to_numpy(float)
    lo = d["spearman_ci95_low"].to_numpy(float)
    hi = d["spearman_ci95_high"].to_numpy(float)
    ax.barh(y, x)
    if np.isfinite(lo).any() and np.isfinite(hi).any():
        ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt="none", color="black", capsize=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.set_xlabel("Out-of-fold Spearman correlation\n(acoustic axis score vs. clinical axis score)")
    ax.set_title("CCA alignment across domain-specific clinical panels")
    savefig(out_path)

def plot_axis_association_forest(assoc_df: pd.DataFrame, out_path: Path, panel: str, component: int = 1, top_n: int = 18) -> None:
    d = assoc_df[(assoc_df["panel"] == panel) & (assoc_df["adjustment"] == "none") & (assoc_df["component"] == component)].copy()
    d = d[np.isfinite(d["spearman_axis_variable"])]
    if len(d) == 0:
        return
    d["abs_rho"] = d["spearman_axis_variable"].abs()
    d = d.sort_values("abs_rho", ascending=False).head(top_n).sort_values("spearman_axis_variable")
    plt.figure(figsize=(9.5, max(5, 0.42 * len(d))))
    ax = plt.gca()
    y = np.arange(len(d))
    x = d["spearman_axis_variable"].to_numpy(float)
    lo = d["spearman_ci95_low"].to_numpy(float)
    hi = d["spearman_ci95_high"].to_numpy(float)
    ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt="o", capsize=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["variable"])
    ax.set_xlabel("Spearman correlation with acoustic axis score")
    ax.set_title(f"Clinical interpretation: {panel} CCA acoustic axis {component}")
    savefig(out_path)



def plot_gradient_heatmap(gradient_df: pd.DataFrame, assoc_df: pd.DataFrame, out_path: Path, panel: str, component: int = 1, top_n: int = 12) -> None:
    d = gradient_df[(gradient_df["panel"] == panel) & (gradient_df["adjustment"] == "none") & (gradient_df["component"] == component)].copy()
    if len(d) == 0:
        return
    assoc = assoc_df[(assoc_df["panel"] == panel) & (assoc_df["adjustment"] == "none") & (assoc_df["component"] == component)].copy()
    assoc = assoc[np.isfinite(assoc["spearman_axis_variable"])]
    if len(assoc) == 0:
        return
    assoc["abs_rho"] = assoc["spearman_axis_variable"].abs()
    top_vars = assoc.sort_values("abs_rho", ascending=False).head(top_n)["variable"].tolist()
    d = d[d["variable"].isin(top_vars)]
    if len(d) == 0:
        return
    value_col = "display_value" if "display_value" in d.columns else "median"
    mat = d.pivot_table(index="variable", columns="axis_group", values=value_col, aggfunc="first")
    mat = mat.loc[[v for v in top_vars if v in mat.index]]
    mat_z = mat.copy().astype(float)
    for idx in mat_z.index:
        vals = mat_z.loc[idx].to_numpy(dtype=float)
        sd = np.nanstd(vals)
        mu = np.nanmean(vals)
        mat_z.loc[idx] = (vals - mu) / sd if np.isfinite(sd) and sd > 1e-12 else 0.0
    plt.figure(figsize=(8.8, max(4.8, 0.45 * len(mat_z))))
    if HAS_SEABORN:
        ax = sns.heatmap(mat_z, annot=True, fmt=".2f", linewidths=0.5, cbar_kws={"label": "Row-wise z-scored display metric"})
    else:
        ax = plt.gca()
        im = ax.imshow(mat_z.values, aspect="auto")
        plt.colorbar(im, ax=ax, label="Row-wise z-scored display metric")
        ax.set_xticks(np.arange(mat_z.shape[1])); ax.set_xticklabels(mat_z.columns)
        ax.set_yticks(np.arange(mat_z.shape[0])); ax.set_yticklabels(mat_z.index)
    ax.set_xlabel(f"{panel} CCA acoustic axis {component} quantile")
    ax.set_ylabel("")
    ax.set_title("Clinical profile across acoustic-clinical axis")
    savefig(out_path)


def plot_gradient_small_multiples(gradient_df: pd.DataFrame, out_path: Path, panel: str, component: int = 1, max_vars: int = 8) -> None:
    d = gradient_df[(gradient_df["panel"] == panel) & (gradient_df["adjustment"] == "none") & (gradient_df["component"] == component)].copy()
    if len(d) == 0 or "display_value" not in d.columns:
        return
    # Rank variables by absolute axis association to show the most informative ones first.
    rank = d.groupby("variable")["axis_variable_spearman"].first().abs().sort_values(ascending=False)
    variables = rank.head(max_vars).index.tolist()
    if not variables:
        return
    n_vars = len(variables)
    n_cols = 2 if n_vars > 3 else 1
    n_rows = int(np.ceil(n_vars / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.2 * n_rows), squeeze=False)
    x_order = [f"Q{i+1}" for i in range(d["axis_group"].nunique())]
    for ax, var in zip(axes.ravel(), variables):
        sub = d[d["variable"] == var].copy()
        sub["axis_group"] = pd.Categorical(sub["axis_group"], categories=x_order, ordered=True)
        sub = sub.sort_values("axis_group")
        x = np.arange(len(sub))
        y = sub["display_value"].to_numpy(float)
        ax.plot(x, y, marker="o", lw=2)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["axis_group"].astype(str).tolist())
        metric_label = sub["display_metric_label"].iloc[0] if "display_metric_label" in sub.columns else "Value"
        ax.set_title(var)
        ax.set_ylabel(metric_label)
        ax.set_xlabel(f"CCA axis {component} quantile")
        # For proportions, keep y-axis in [0, 1] for readability.
        if sub["display_unit"].iloc[0] == "proportion":
            ax.set_ylim(0, max(0.05, min(1.0, np.nanmax(y) * 1.25 if np.isfinite(y).any() else 1.0)))
    for ax in axes.ravel()[n_vars:]:
        ax.axis("off")
    fig.suptitle(f"Clinical gradient across {panel} CCA acoustic axis {component}", y=1.02, fontsize=16)
    savefig(out_path)

def plot_endpoint_summary(df: pd.DataFrame, out_path: Path) -> None:
    if len(df) == 0 or "auroc" not in df.columns:
        return
    d = df[np.isfinite(df["auroc"])].copy()
    if len(d) == 0:
        return
    d = d.sort_values("endpoint")
    plt.figure(figsize=(max(7.5, 1.1 * len(d)), 5.5))
    ax = plt.gca()
    x = np.arange(len(d))
    y = d["auroc"].to_numpy(float)
    lo = d["auroc_ci95_low"].to_numpy(float)
    hi = d["auroc_ci95_high"].to_numpy(float)
    ax.bar(x, y)
    if np.isfinite(lo).any() and np.isfinite(hi).any():
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", color="black", capsize=3)
    ax.axhline(0.5, color="black", lw=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(d["endpoint"], rotation=35, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title("Endpoint validation using CCA acoustic axes 1+2")
    savefig(out_path)


def plot_negative_controls(ctrl: pd.DataFrame, out_path: Path, panel: str = "all_clinical") -> None:
    if len(ctrl) == 0:
        return
    d = ctrl[ctrl["panel"] == panel].copy()
    if len(d) == 0:
        return
    plt.figure(figsize=(8.5, 5.5))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.histplot(data=d, x="spearman", hue="control_type", bins=25, element="step", stat="count", common_norm=False, ax=ax)
    else:
        for ct, sub in d.groupby("control_type"):
            ax.hist(sub["spearman"].dropna(), bins=25, alpha=0.5, label=ct)
        ax.legend()
    ax.set_xlabel("Control out-of-fold Spearman correlation")
    ax.set_title(f"Negative controls for {panel} CCA axis 1")
    savefig(out_path)


# =============================================================================
# Report
# =============================================================================

def write_report(out_dir: Path, config: Dict, panels: Dict[str, List[str]], concept_df: pd.DataFrame, single_df: pd.DataFrame,
                 alignment_summary: pd.DataFrame, assoc_df: pd.DataFrame, endpoint_summary: pd.DataFrame,
                 neg_summary: pd.DataFrame, conf_summary: pd.DataFrame, lopo_summary: pd.DataFrame) -> None:
    lines = []
    lines.append("# Clinically anchored acoustic phenotyping V4 summary\n\n")
    lines.append("This analysis uses CCA only. Each CCA model is fitted inside the training fold; held-out patients are projected to acoustic and clinical latent scores. The main metric is the out-of-fold latent-score correlation.\n\n")
    lines.append("## Clinical panels\n")
    for p, vs in panels.items():
        lines.append(f"- **{p}**: {', '.join(vs)}\n")
    lines.append("\n## Detected clinical concepts\n")
    lines.append(concept_df[["concept", "domain", "column", "status", "filter_reason", "missing_fraction", "n_nonmissing", "n_unique"]].pipe(df_to_markdown_safe, index=False))

    if len(single_df) and "spearman_pred_true" in single_df.columns:
        d = single_df[single_df.get("status", "ok") == "ok"].copy()
        d = d[np.isfinite(d["spearman_pred_true"])]
        if len(d):
            d["abs_rho"] = d["spearman_pred_true"].abs()
            top = d.sort_values("abs_rho", ascending=False).head(8)
            lines.append("\n## Single-variable acoustic readout overview\n")
            lines.append(top[["variable", "n", "spearman_pred_true", "spearman_ci95_low", "spearman_ci95_high", "r2"]].pipe(df_to_markdown_safe, index=False))

    if len(alignment_summary):
        lines.append("\n## CCA panel alignment\n")
        cols = ["panel", "adjustment", "component", "n", "spearman_acoustic_vs_clinical_axis", "spearman_ci95_low", "spearman_ci95_high", "permutation_p_abs", "clinical_redundancy_mean_r2", "clinical_redundancy_axis1_2_mean_r2"]
        cols = [c for c in cols if c in alignment_summary.columns]
        lines.append(alignment_summary[cols].pipe(df_to_markdown_safe, index=False))

    if len(assoc_df):
        lines.append("\n## Domain-specific axis interpretation\n")
        for panel in DOMAIN_PLOT_PANELS:
            sub = assoc_df[(assoc_df["panel"] == panel) & (assoc_df["adjustment"] == "none") & (assoc_df["component"] == 1)].copy()
            if len(sub):
                sub["abs_rho"] = sub["spearman_axis_variable"].abs()
                top = sub.sort_values("abs_rho", ascending=False).head(10)
                cols = ["variable", "n", "spearman_axis_variable", "spearman_ci95_low", "spearman_ci95_high", "burden_orientation"]
                lines.append(f"\n### {panel}\n")
                lines.append(top[cols].pipe(df_to_markdown_safe, index=False))

    if len(endpoint_summary):
        lines.append("\n## Endpoint validation\n")
        cols = ["endpoint", "panel_type", "n", "n_positive", "positive_rate", "auroc", "auroc_ci95_low", "auroc_ci95_high", "balanced_accuracy", "accuracy"]
        cols = [c for c in cols if c in endpoint_summary.columns]
        lines.append(endpoint_summary[cols].pipe(df_to_markdown_safe, index=False))

    if len(neg_summary):
        lines.append("\n## Negative controls\n")
        lines.append(neg_summary.pipe(df_to_markdown_safe, index=False))

    if len(conf_summary):
        lines.append("\n## Confounder controls\n")
        cols = ["panel", "adjustment", "component", "n", "spearman_acoustic_vs_clinical_axis", "spearman_ci95_low", "spearman_ci95_high", "covariates_used"]
        cols = [c for c in cols if c in conf_summary.columns]
        lines.append(conf_summary[cols].pipe(df_to_markdown_safe, index=False))

    if len(lopo_summary):
        lines.append("\n## Leave-one-position-out control\n")
        cols = ["left_out_position", "component", "n", "spearman_acoustic_vs_clinical_axis", "spearman_ci95_low", "spearman_ci95_high"]
        cols = [c for c in cols if c in lopo_summary.columns]
        lines.append(lopo_summary[cols].pipe(df_to_markdown_safe, index=False))

    lines.append("\n## Configuration\n```json\n")
    lines.append(json.dumps(config, indent=2, ensure_ascii=False))
    lines.append("\n```\n")
    path = out_dir / "analysis_summary.md"
    path.write_text("".join(lines), encoding="utf-8")
    log(f"Saved Markdown summary: {path}")


# =============================================================================
# CLI and main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Clinically anchored acoustic phenotyping V4: CCA-only domain-panel analyses")
    p.add_argument("--embedding-dir", type=str, default="Representation_learning/embeddings_4_1/beats")
    p.add_argument("--patient-embedding-npy", type=str, default=None)
    p.add_argument("--patient-meta-csv", type=str, default=None)
    p.add_argument("--embedding-patient-id-col", type=str, default=None)

    p.add_argument("--position-embedding-npy", type=str, default=None)
    p.add_argument("--position-meta-csv", type=str, default=None)
    p.add_argument("--position-patient-id-col", type=str, default=None)
    p.add_argument("--position-col", type=str, default=None)
    p.add_argument("--positions", type=str, default="A,E,M,P,T")
    p.add_argument("--run-leave-one-position-out", action="store_true", default=True)
    p.add_argument("--no-leave-one-position-out", dest="run_leave_one_position_out", action="store_false")

    p.add_argument("--clinical-csv", type=str, default="Clinical_alignment/outputs/prepared/aligned_clinical_clean.csv")
    p.add_argument("--clinical-xlsx", type=str, default="Data/patient_info.xlsx")
    p.add_argument("--clinical-sheet", type=str, default=0)
    p.add_argument("--patient-id-col", type=str, default=None)

    p.add_argument("--out-dir", type=str, default="Clinical_alignment/outputs/clinically_anchored_acoustic_phenotyping_v4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-components", type=int, default=2)
    p.add_argument("--n-pca", type=int, default=50)
    p.add_argument("--cca-max-iter", type=int, default=3000)

    p.add_argument("--max-missing", type=float, default=0.50)
    p.add_argument("--min-target-n", type=int, default=50)
    p.add_argument("--min-panel-n", type=int, default=80)
    p.add_argument("--valve-max-missing", type=float, default=0.85,
                   help="More permissive missingness threshold for valve variables, which are often sparse.")
    p.add_argument("--valve-min-n", type=int, default=30,
                   help="Minimum non-missing samples for valve variables.")
    p.add_argument("--valve-min-unique", type=int, default=2,
                   help="Minimum unique values for valve variables.")
    p.add_argument("--min-nonmissing-clinical-vars", type=int, default=3)
    p.add_argument("--core-max-vars", type=int, default=8)
    p.add_argument("--n-axis-groups", type=int, default=4)

    p.add_argument("--min-endpoint-n", type=int, default=80)
    p.add_argument("--min-endpoint-class-n", type=int, default=5)
    p.add_argument("--leave-endpoint-out", action="store_true", default=True)
    p.add_argument("--no-leave-endpoint-out", dest="leave_endpoint_out", action="store_false")

    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-loading-bootstrap", type=int, default=200)
    p.add_argument("--n-permutations", type=int, default=100)
    p.add_argument("--n-random-controls", type=int, default=20)
    p.add_argument("--progress-every", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    setup_plotting()

    out_dir = Path(args.out_dir)
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    ensure_dir(table_dir); ensure_dir(fig_dir)
    config = vars(args).copy()
    (out_dir / "analysis_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    log("Loading and aligning data")
    X_raw, meta, emb_path, meta_path = load_patient_embeddings(args)
    clinical_raw = read_clinical_table(args)
    clinical_num = build_numeric_clinical_table(clinical_raw)
    X, clinical, patient_df = align_embeddings_and_clinical(X_raw, meta, clinical_num)
    finite_x = np.isfinite(X).all(axis=1)
    if not finite_x.all():
        log(f"Dropping {(~finite_x).sum()} patients with non-finite embeddings")
        X = X[finite_x]
        clinical = clinical.loc[finite_x].reset_index(drop=True)
        patient_df = patient_df.loc[finite_x].reset_index(drop=True)
    patient_ids = clinical["patient_id"].tolist()

    log("Detecting clinically relevant variables and covariates")
    concept_df = detect_clinical_concepts(clinical, args)
    concept_df.to_csv(table_dir / "clinical_concepts_detected.csv", index=False, encoding="utf-8-sig")
    # Save all regex-matched concept candidates to explain why a domain such as valve
    # may be absent: not found, too sparse, too many missing values, or constant.
    if hasattr(args, "_clinical_concept_candidates") and isinstance(args._clinical_concept_candidates, pd.DataFrame):
        args._clinical_concept_candidates.to_csv(table_dir / "clinical_concept_candidates_all.csv", index=False, encoding="utf-8-sig")
    cov_cols = find_covariate_columns(clinical)
    pd.DataFrame([{"covariate": k, "column": v} for k, v in cov_cols.items()]).to_csv(table_dir / "covariates_detected.csv", index=False, encoding="utf-8-sig")
    panels, panel_vars_df = build_clinical_panels(concept_df, pd.DataFrame(), args)
    if not any(p in panels for p in DOMAIN_PLOT_PANELS):
        raise ValueError("No domain-specific clinical panel could be built. Check clinical variables and missingness.")
    if "all_clinical" not in panels:
        raise ValueError("all_clinical panel could not be built from domain-specific panels.")

    # Single-variable readout is a complementary variable-level overview.
    # Heart rate is excluded from anchoring/readout targets and handled as a covariate.
    candidate_vars = panels["all_clinical"]
    pd.DataFrame({"patient_id": patient_ids}).to_csv(table_dir / "patients_aligned_before_panel_filter.csv", index=False, encoding="utf-8-sig")

    single_df = single_variable_readout(X, clinical, candidate_vars, args)
    single_df.to_csv(table_dir / "single_variable_readout.csv", index=False, encoding="utf-8-sig")
    plot_single_variable_readout(single_df, fig_dir / "single_variable_readout_lollipop_by_domain.png", concept_df=concept_df)

    panel_vars_df.to_csv(table_dir / "clinical_panels_used.csv", index=False, encoding="utf-8-sig")
    log("Clinical panels:")
    for name, vars_ in panels.items():
        log(f"  {name}: {vars_}")

    # Main CCA panel analyses.
    score_dfs = []
    fold_dfs = []
    summary_dfs = []
    assoc_dfs = []
    fold_assoc_dfs = []
    signstab_dfs = []
    gradient_dfs = []
    loading_boot_dfs = []
    loading_summary_dfs = []
    neg_ctrl_dfs = []
    neg_summary_rows = []

    for panel_name, vars_ in panels.items():
        try:
            Xp, Cp, pids_p, mask = filter_for_panel(X, clinical, patient_ids, vars_, args)
        except Exception as e:
            log(f"Skip panel={panel_name}: {e}")
            continue
        pd.DataFrame({"patient_id": pids_p}).to_csv(table_dir / f"patients_used_{panel_name}.csv", index=False, encoding="utf-8-sig")
        sdf, fdf = run_oof_cca_panel(Xp, Cp, vars_, pids_p, panel_name, args, concept_df, verbose=True)
        score_dfs.append(sdf)
        fold_dfs.append(fdf)
        summ = summarize_alignment(sdf, vars_, Cp, args)

        # Permutation p for axis 1 for each panel.
        obs = summ.loc[summ["component"] == 1, "spearman_acoustic_vs_clinical_axis"].iloc[0] if len(summ) else np.nan
        ctrl, ctrl_sum = permutation_test_panel(Xp, Cp, vars_, pids_p, panel_name, obs, args, concept_df)
        if len(ctrl): neg_ctrl_dfs.append(ctrl)
        if ctrl_sum: neg_summary_rows.append(ctrl_sum)
        if len(summ) and ctrl_sum:
            summ.loc[summ["component"] == 1, "permutation_p_abs"] = ctrl_sum.get("empirical_p_abs_ge_observed", np.nan)
        # Random embedding control only for all_clinical panel, to avoid excessive runtime.
        if panel_name == "all_clinical":
            rctrl, rsum = random_embedding_control_core(Xp, Cp, vars_, pids_p, panel_name, obs, args, concept_df)
            if len(rctrl): neg_ctrl_dfs.append(rctrl)
            if rsum: neg_summary_rows.append(rsum)
        summary_dfs.append(summ)

        assoc, fold_assoc, signstab = axis_clinical_associations(sdf, Cp, vars_, args, concept_df)
        assoc_dfs.append(assoc); fold_assoc_dfs.append(fold_assoc); signstab_dfs.append(signstab)
        grad = clinical_gradient_by_axis(sdf, Cp, vars_, args, concept_df=concept_df)
        gradient_dfs.append(grad)
        if panel_name in DOMAIN_PLOT_PANELS:
            plot_axis_association_forest(assoc, fig_dir / f"{clean_filename(panel_name)}_axis1_clinical_association_forest.png", panel=panel_name, component=1)
            plot_gradient_heatmap(grad, assoc, fig_dir / f"{clean_filename(panel_name)}_axis1_clinical_gradient_heatmap.png", panel=panel_name, component=1)
            plot_gradient_small_multiples(grad, fig_dir / f"{clean_filename(panel_name)}_axis1_clinical_gradient_small_multiples.png", panel=panel_name, component=1)
        boot, bootsum = bootstrap_loading_stability(Xp, Cp, vars_, panel_name, args, concept_df)
        if len(boot): loading_boot_dfs.append(boot)
        if len(bootsum): loading_summary_dfs.append(bootsum)

    scores_all = pd.concat(score_dfs, ignore_index=True) if score_dfs else pd.DataFrame()
    folds_all = pd.concat(fold_dfs, ignore_index=True) if fold_dfs else pd.DataFrame()
    alignment_summary = pd.concat(summary_dfs, ignore_index=True) if summary_dfs else pd.DataFrame()
    assoc_all = pd.concat(assoc_dfs, ignore_index=True) if assoc_dfs else pd.DataFrame()
    fold_assoc_all = pd.concat(fold_assoc_dfs, ignore_index=True) if fold_assoc_dfs else pd.DataFrame()
    signstab_all = pd.concat(signstab_dfs, ignore_index=True) if signstab_dfs else pd.DataFrame()
    gradient_all = pd.concat(gradient_dfs, ignore_index=True) if gradient_dfs else pd.DataFrame()
    loading_boot_all = pd.concat(loading_boot_dfs, ignore_index=True) if loading_boot_dfs else pd.DataFrame()
    loading_summary_all = pd.concat(loading_summary_dfs, ignore_index=True) if loading_summary_dfs else pd.DataFrame()
    neg_ctrl_all = pd.concat(neg_ctrl_dfs, ignore_index=True) if neg_ctrl_dfs else pd.DataFrame()
    neg_summary = pd.DataFrame(neg_summary_rows)

    scores_all.to_csv(table_dir / "oof_cca_axis_scores_by_panel.csv", index=False, encoding="utf-8-sig")
    folds_all.to_csv(table_dir / "fold_level_cca_alignment_summary.csv", index=False, encoding="utf-8-sig")
    alignment_summary.to_csv(table_dir / "cca_panel_alignment_summary.csv", index=False, encoding="utf-8-sig")
    assoc_all.to_csv(table_dir / "cca_axis_clinical_associations.csv", index=False, encoding="utf-8-sig")
    fold_assoc_all.to_csv(table_dir / "fold_level_axis_clinical_associations.csv", index=False, encoding="utf-8-sig")
    signstab_all.to_csv(table_dir / "cross_validation_loading_sign_stability.csv", index=False, encoding="utf-8-sig")
    gradient_all.to_csv(table_dir / "clinical_gradient_by_axis.csv", index=False, encoding="utf-8-sig")
    loading_boot_all.to_csv(table_dir / "bootstrap_axis_loading_values.csv", index=False, encoding="utf-8-sig")
    loading_summary_all.to_csv(table_dir / "bootstrap_axis_loading_stability_summary.csv", index=False, encoding="utf-8-sig")
    neg_ctrl_all.to_csv(table_dir / "negative_controls.csv", index=False, encoding="utf-8-sig")
    neg_summary.to_csv(table_dir / "negative_control_summary.csv", index=False, encoding="utf-8-sig")
    plot_panel_alignment(alignment_summary, fig_dir / "cca_panel_alignment_summary.png")
    plot_negative_controls(neg_ctrl_all, fig_dir / "negative_controls_all_clinical.png", panel="all_clinical")

    # Endpoint validation on all_clinical panel with leave-endpoint-out anchoring.
    all_clinical_vars = panels["all_clinical"]
    try:
        Xall, Call, pids_all, _ = filter_for_panel(X, clinical, patient_ids, all_clinical_vars, args)
        endpoint_summary, endpoint_pred, endpoint_info = run_endpoint_validation_all_clinical(Xall, Call, pids_all, all_clinical_vars, args, concept_df)
    except Exception as e:
        log(f"Endpoint validation failed/skipped: {e}")
        endpoint_summary, endpoint_pred, endpoint_info = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    endpoint_info.to_csv(table_dir / "endpoints_used.csv", index=False, encoding="utf-8-sig")
    endpoint_summary.to_csv(table_dir / "endpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
    endpoint_pred.to_csv(table_dir / "endpoint_validation_predictions.csv", index=False, encoding="utf-8-sig")
    plot_endpoint_summary(endpoint_summary, fig_dir / "endpoint_validation_auroc.png")

    # Confounder controls on all_clinical panel.
    try:
        conf_summary, conf_assoc = run_confounder_controls(Xall, Call, pids_all, all_clinical_vars, cov_cols, args, concept_df)
    except Exception as e:
        log(f"Confounder controls failed/skipped: {e}")
        conf_summary, conf_assoc = pd.DataFrame(), pd.DataFrame()
    conf_summary.to_csv(table_dir / "confounder_adjusted_alignment_summary.csv", index=False, encoding="utf-8-sig")
    conf_assoc.to_csv(table_dir / "confounder_adjusted_axis_clinical_associations.csv", index=False, encoding="utf-8-sig")

    # Leave-one-position-out position coverage control on all_clinical panel.
    try:
        lopo_summary = run_leave_one_position_control(X, clinical, patient_ids, all_clinical_vars, args, concept_df)
    except Exception as e:
        log(f"Leave-one-position-out control failed/skipped: {e}")
        lopo_summary = pd.DataFrame()
    lopo_summary.to_csv(table_dir / "leave_one_position_out_alignment_summary.csv", index=False, encoding="utf-8-sig")

    # Save aligned clinical panel values for checking.
    all_panel_cols = []
    for vs in panels.values():
        for v in vs:
            if v not in all_panel_cols:
                all_panel_cols.append(v)
    clinical[["patient_id"] + all_panel_cols].to_csv(table_dir / "aligned_clinical_values_for_detected_panels.csv", index=False, encoding="utf-8-sig")

    write_report(out_dir, config, panels, concept_df, single_df, alignment_summary, assoc_all, endpoint_summary, neg_summary, conf_summary, lopo_summary)
    log("Done.")
    log(f"All outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
