#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clinically anchored acoustic phenotyping from multi-site heart-sound embeddings.

This script runs the core experiments for the manuscript question:
"Do multi-site heart-sound representations contain patient-level acoustic phenotype
 structures that align with multidimensional clinical status?"

Main outputs
------------
1) Single-variable acoustic readout
   - Tests whether each clinical variable can be directly read out from BEATs patient embeddings.
   - Uses leakage-safe outer CV with scaler/PCA/Ridge fitted only on training folds.

2) Multivariate clinical anchoring with CCA and PLS
   - Learns low-dimensional acoustic-clinical axes from X=patient embeddings and Y=clinical panel.
   - Reports out-of-fold acoustic-score vs clinical-score alignment.

3) Axis clinical interpretation and stability summaries
   - Reports Spearman associations between out-of-fold acoustic axes and clinical variables.
   - Uses bootstrap CIs and fold-level sign consistency.

4) Clinical-gradient analysis
   - Groups patients by the primary acoustic-clinical axis and summarizes clinical gradients.

5) Endpoint validation
   - Tests whether aligned axes distinguish interpretable clinical endpoints, e.g. EF<40,
     NT-proBNP>=300, NYHA>=III.
   - Uses nested/leakage-safe CV: alignment and logistic readout are fitted only on training folds.
   - By default, the endpoint's own variable is excluded from the clinical anchoring panel for that endpoint.

6) Minimal negative controls
   - Patient-label permutation control.
   - Random-embedding control.

Example
-------
python run_clinically_anchored_acoustic_phenotyping.py ^
  --embedding-dir Representation_learning/embeddings_4_1/beats ^
  --clinical-csv Clinical_alignment/outputs/prepared/aligned_clinical_clean.csv ^
  --out-dir Clinical_alignment/outputs/clinically_anchored_acoustic_phenotyping ^
  --n-splits 5 ^
  --n-permutations 100 ^
  --n-random-controls 20 ^
  --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
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


# -----------------------------
# Utilities
# -----------------------------

def now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_patient_id(x) -> Optional[str]:
    """Normalize patient IDs for robust merging across CSV/Excel/meta files."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    # Common Excel artifact: integer IDs become "123.0".
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    s = re.sub(r"\s+", "", s)
    return s


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
    """Benjamini-Hochberg FDR correction without requiring statsmodels."""
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


def bootstrap_ci_spearman(
    x: Sequence[float],
    y: Sequence[float],
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
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
            vals.append(rho)
    if len(vals) < 20:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_ci_auc(
    y_true: Sequence[int],
    y_score: Sequence[float],
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
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
        try:
            vals.append(roc_auc_score(y_true[idx], y_score[idx]))
        except Exception:
            pass
    if len(vals) < 20:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def clean_filename(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", str(s))
    return s[:180]


# -----------------------------
# Loading and preprocessing
# -----------------------------

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
                "patient_embeddings.npy",
                "patient_embedding.npy",
                "patient_level_embeddings.npy",
                "patient_embeds.npy",
                "patient_emb.npy",
                "embeddings_patient.npy",
            ],
        )
        if emb_path is None:
            matches = sorted(emb_dir.glob("*patient*embed*.npy"))
            if matches:
                emb_path = matches[0]
    if emb_path is None or not emb_path.exists():
        raise FileNotFoundError(
            "Cannot find patient-level embedding .npy. Please pass --patient-embedding-npy explicitly."
        )

    if args.patient_meta_csv:
        meta_path = Path(args.patient_meta_csv)
    else:
        meta_path = find_first_existing(
            emb_dir,
            [
                "patient_meta.csv",
                "patient_metadata.csv",
                "patient_embeddings_meta.csv",
                "patient_order.csv",
                "patient_ids.csv",
                "patients.csv",
            ],
        )
        if meta_path is None:
            matches = sorted(emb_dir.glob("*patient*meta*.csv")) + sorted(emb_dir.glob("*patient*order*.csv"))
            if matches:
                meta_path = matches[0]
    if meta_path is None or not meta_path.exists():
        raise FileNotFoundError(
            "Cannot find patient-level meta/order CSV. Please pass --patient-meta-csv explicitly."
        )

    X = np.load(emb_path)
    meta = pd.read_csv(meta_path)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D patient embedding array, got shape={X.shape}")
    if len(meta) != X.shape[0]:
        raise ValueError(
            f"Meta rows ({len(meta)}) do not match embedding rows ({X.shape[0]}).\n"
            f"Embedding: {emb_path}\nMeta: {meta_path}"
        )

    pid_col = infer_patient_id_col(meta, explicit=args.embedding_patient_id_col, context="embedding meta")
    meta = meta.copy()
    meta["patient_id"] = meta[pid_col].map(normalize_patient_id)
    meta["embedding_row"] = np.arange(len(meta))
    if meta["patient_id"].duplicated().any():
        dup = meta.loc[meta["patient_id"].duplicated(), "patient_id"].head().tolist()
        raise ValueError(f"Duplicate patient_id found in embedding meta, e.g. {dup}")

    log(f"Loaded patient embeddings: {emb_path} with shape {X.shape}")
    log(f"Loaded patient meta: {meta_path} using patient ID column '{pid_col}'")
    return X.astype(np.float32), meta, emb_path, meta_path


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
    # If there is only one column in a patient_order.csv-like file, use it.
    if df.shape[1] == 1:
        return df.columns[0]
    # Try fuzzy.
    for c in df.columns:
        lc = str(c).lower()
        if ("patient" in lc or "subject" in lc or lc in {"pid", "id"}) and "embedding" not in lc:
            return c
    raise ValueError(
        f"Cannot infer patient ID column in {context}. Columns are:\n{list(df.columns)}\n"
        "Please pass --patient-id-col or --embedding-patient-id-col."
    )


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
    log(f"Loaded clinical table with {before} rows; retained {len(df)} unique patient IDs using column '{pid_col}'")
    return df


def coerce_to_numeric(s: pd.Series) -> pd.Series:
    """Coerce mixed clinical values into numeric where possible."""
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
        low = txt.lower().replace(" ", "")
        if low in mapping:
            out.append(mapping[low])
            continue
        # Handle common inequality values such as >300, ≥300, <5.
        low = low.replace(",", "")
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
        # Keep if at least a small number of numeric values exist.
        if coerced.notna().sum() >= 5:
            num[c] = coerced
    num.insert(0, "patient_id", df["patient_id"].values)
    return num


def align_embeddings_and_clinical(
    X: np.ndarray,
    meta: pd.DataFrame,
    clinical: pd.DataFrame,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    merged = meta[["patient_id", "embedding_row"]].merge(clinical, on="patient_id", how="inner")
    if len(merged) == 0:
        raise ValueError("No overlapping patient IDs between embeddings and clinical table.")
    row_idx = merged["embedding_row"].to_numpy(dtype=int)
    X_aligned = X[row_idx]
    clinical_aligned = merged.drop(columns=["embedding_row"]).reset_index(drop=True)
    patient_df = merged[["patient_id", "embedding_row"]].reset_index(drop=True)
    log(f"Aligned embeddings and clinical table: n={len(patient_df)} patients")
    return X_aligned, clinical_aligned, patient_df


# -----------------------------
# Clinical variable selection
# -----------------------------

CORE_PATTERNS: List[Tuple[str, List[str]]] = [
    ("EF", [r"^ef$", r"lvef", r"ejection", r"射血"]),
    ("NTproBNP", [r"nt[-_ ]?pro[-_ ]?bnp", r"ntprobnp", r"nt_pro_bnp", r"nt.*bnp", r"脑钠", r"bnp"]),
    ("NYHA", [r"nyha"]),
    ("HR", [r"^hr$", r"heart.*rate", r"心率"]),
    ("LA", [r"^la$", r"^lad$", r"left.*atri", r"左房", r"左心房"]),
    ("LVEDD", [r"lvedd", r"lvdd", r"left.*vent.*diast", r"左室.*舒张", r"左心室.*舒张"]),
    ("LVESD", [r"lvesd", r"left.*vent.*syst", r"左室.*收缩", r"左心室.*收缩"]),
    ("IVS", [r"ivs", r"室间隔"]),
    ("LVPW", [r"lvpw", r"后壁"]),
    ("MR", [r"^mr$", r"mitral.*regurg", r"二尖瓣.*反流"]),
    ("TR", [r"^tr$", r"tricuspid.*regurg", r"三尖瓣.*反流"]),
    ("AR", [r"^ar$", r"aortic.*regurg", r"主动脉.*反流"]),
    ("PR", [r"^pr$", r"pulmonary.*regurg", r"肺动脉.*反流"]),
    ("AS", [r"^as$", r"aortic.*sten", r"主动脉.*狭窄"]),
    ("MS", [r"^ms$", r"mitral.*sten", r"二尖瓣.*狭窄"]),
    ("SBP", [r"sbp", r"收缩压"]),
    ("DBP", [r"dbp", r"舒张压"]),
]

DEMOGRAPHIC_PATTERNS = [r"age", r"年龄", r"sex", r"gender", r"性别", r"男"]


def is_demographic_col(col: str) -> bool:
    lc = str(col).lower()
    return any(re.search(p, lc) for p in DEMOGRAPHIC_PATTERNS)


def select_first_matching_column(columns: Sequence[str], patterns: Sequence[str]) -> Optional[str]:
    for p in patterns:
        regex = re.compile(p, flags=re.I)
        matches = [c for c in columns if regex.search(str(c))]
        if matches:
            # Prefer shorter/exact-looking names to avoid accidental matches.
            matches = sorted(matches, key=lambda x: (len(str(x)), str(x)))
            return matches[0]
    return None


def select_clinical_variables(num_df: pd.DataFrame, args) -> List[str]:
    all_cols = [c for c in num_df.columns if c != "patient_id"]

    if args.clinical_vars:
        requested = [c.strip() for c in args.clinical_vars.split(",") if c.strip()]
        missing = [c for c in requested if c not in num_df.columns]
        if missing:
            raise ValueError(f"Requested --clinical-vars not found in clinical table: {missing}")
        selected = requested
    else:
        selected = []
        used = set()
        for label, patterns in CORE_PATTERNS:
            col = select_first_matching_column([c for c in all_cols if c not in used], patterns)
            if col is not None:
                selected.append(col)
                used.add(col)

        # If too few core variables were found, add numeric non-demographic variables with acceptable missingness.
        if len(selected) < args.min_clinical_vars:
            candidates = []
            for c in all_cols:
                if c in used:
                    continue
                if (not args.include_demographics) and is_demographic_col(c):
                    continue
                miss = num_df[c].isna().mean()
                uniq = num_df[c].nunique(dropna=True)
                if miss <= args.max_missing and uniq >= 2:
                    candidates.append((miss, -uniq, c))
            candidates = sorted(candidates)
            for _, _, c in candidates:
                selected.append(c)
                if len(selected) >= args.max_clinical_vars:
                    break

    # Filter by missingness and variability.
    final = []
    for c in selected:
        miss = num_df[c].isna().mean()
        uniq = num_df[c].nunique(dropna=True)
        if miss <= args.max_missing and uniq >= 2:
            final.append(c)
    # Limit max variables to keep CCA/PLS stable.
    final = final[: args.max_clinical_vars]
    if len(final) < args.min_clinical_vars:
        raise ValueError(
            f"Only {len(final)} clinical variables were selected after filtering. "
            f"Please pass --clinical-vars explicitly or relax --max-missing. Selected={final}"
        )
    return final


def filter_patients_for_clinical_panel(
    X: np.ndarray,
    clinical: pd.DataFrame,
    clinical_vars: List[str],
    patient_df: pd.DataFrame,
    min_nonmissing: int,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    n_nonmiss = clinical[clinical_vars].notna().sum(axis=1)
    mask = n_nonmiss >= min_nonmissing
    kept = int(mask.sum())
    if kept < 30:
        raise ValueError(
            f"Only {kept} patients have >= {min_nonmissing} non-missing clinical variables."
        )
    log(f"Clinical panel completeness filter: retained {kept}/{len(mask)} patients")
    return X[mask.values], clinical.loc[mask].reset_index(drop=True), patient_df.loc[mask].reset_index(drop=True)


# -----------------------------
# Modeling helpers
# -----------------------------

@dataclass
class AlignmentFitResult:
    x_scores_train: np.ndarray
    y_scores_train: np.ndarray
    x_scores_test: np.ndarray
    y_scores_test: np.ndarray
    used_clinical_vars: List[str]
    n_pca: int


def choose_n_pca(n_train: int, n_features: int, n_targets: int, requested: int) -> int:
    # Leave degrees of freedom for CCA/PLS. If X is already low-dimensional, PCA may keep all possible components.
    return int(max(1, min(requested, n_features, n_train - 2)))


def clinical_burden_direction(col: str) -> int:
    """
    Direction used only for sign orientation.
    +1 means higher clinical value usually means heavier cardiac burden.
    -1 means higher clinical value usually means better function, so axis should be reversed.
    0 means not used for orientation.
    """
    lc = str(col).lower()
    # EF/LVEF higher is generally better, so high burden axis should correlate negatively with EF.
    if re.search(r"(^ef$|lvef|ejection|射血)", lc):
        return -1
    # Most of these higher values imply heavier burden or abnormality.
    pos_patterns = [
        r"nt.*bnp", r"bnp", r"nyha", r"heart.*rate", r"^hr$", r"心率",
        r"^la$", r"lad", r"left.*atri", r"左房", r"左心房",
        r"lvedd", r"lvdd", r"lvesd", r"left.*vent", r"左室", r"左心室",
        r"mr", r"tr", r"ar", r"pr", r"as", r"ms", r"反流", r"狭窄",
        r"sbp", r"dbp", r"收缩压", r"舒张压",
    ]
    if any(re.search(p, lc) for p in pos_patterns):
        return +1
    return 0


def orient_scores_by_clinical_burden(
    scores_train: np.ndarray,
    scores_test: np.ndarray,
    y_scores_train: np.ndarray,
    y_scores_test: np.ndarray,
    clinical_train_raw: pd.DataFrame,
    clinical_vars: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Resolve fold-wise sign ambiguity so that axis 1 roughly points toward heavier cardiac burden."""
    signs = []
    xtr = scores_train.copy()
    xte = scores_test.copy()
    ytr = y_scores_train.copy()
    yte = y_scores_test.copy()

    n_components = xtr.shape[1]
    for k in range(n_components):
        evidence = []
        for c in clinical_vars:
            d = clinical_burden_direction(c)
            if d == 0:
                continue
            rho, _, n = safe_spearman(xtr[:, k], clinical_train_raw[c].values)
            if n >= 20 and np.isfinite(rho):
                evidence.append(d * rho)
        orientation_score = float(np.nanmean(evidence)) if evidence else 0.0
        sign = +1 if orientation_score >= 0 else -1
        signs.append(sign)
        if sign < 0:
            xtr[:, k] *= -1
            xte[:, k] *= -1
            ytr[:, k] *= -1
            yte[:, k] *= -1
    return xtr, xte, ytr, yte, signs


def fit_alignment_one_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train_raw: pd.DataFrame,
    Y_test_raw: pd.DataFrame,
    clinical_vars: List[str],
    method: str,
    n_components: int,
    n_pca_requested: int,
) -> AlignmentFitResult:
    """Fit scaler/PCA/imputer/scaler/CCA-or-PLS only on training data, then transform test data."""
    # Drop clinical variables that become constant within this training fold after imputation.
    Ytr_df = Y_train_raw[clinical_vars].copy()
    Yte_df = Y_test_raw[clinical_vars].copy()
    imputer = SimpleImputer(strategy="median")
    Ytr_imp = imputer.fit_transform(Ytr_df)
    Yte_imp = imputer.transform(Yte_df)
    keep = np.nanstd(Ytr_imp, axis=0) > 1e-12
    used_vars = [c for c, k in zip(clinical_vars, keep) if k]
    Ytr_imp = Ytr_imp[:, keep]
    Yte_imp = Yte_imp[:, keep]
    if Ytr_imp.shape[1] < 2:
        raise ValueError("Too few non-constant clinical variables in this fold after imputation.")

    x_scaler = StandardScaler()
    Xtr_s = x_scaler.fit_transform(X_train)
    Xte_s = x_scaler.transform(X_test)

    n_pca = choose_n_pca(len(X_train), X_train.shape[1], Ytr_imp.shape[1], n_pca_requested)
    pca = PCA(n_components=n_pca, random_state=0)
    Xtr_r = pca.fit_transform(Xtr_s)
    Xte_r = pca.transform(Xte_s)

    y_scaler = StandardScaler()
    Ytr_s = y_scaler.fit_transform(Ytr_imp)
    Yte_s = y_scaler.transform(Yte_imp)

    n_comp = int(min(n_components, Xtr_r.shape[1], Ytr_s.shape[1], len(X_train) - 2))
    if n_comp < 1:
        raise ValueError("n_components became <1 in this fold.")

    method_l = method.lower()
    if method_l == "cca":
        model = CCA(n_components=n_comp, max_iter=2000, scale=False)
    elif method_l == "pls":
        model = PLSRegression(n_components=n_comp, scale=False)
    else:
        raise ValueError(f"Unknown method: {method}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xtr_r, Ytr_s)
        transformed_train = model.transform(Xtr_r, Ytr_s)
        transformed_test = model.transform(Xte_r, Yte_s)

    # sklearn returns tuple when Y is provided.
    if isinstance(transformed_train, tuple):
        xtr_scores, ytr_scores = transformed_train
        xte_scores, yte_scores = transformed_test
    else:
        # Fallback: PLS/CCA should usually not reach here when Y is provided.
        xtr_scores = transformed_train
        xte_scores = transformed_test
        ytr_scores = np.full_like(xtr_scores, np.nan)
        yte_scores = np.full_like(xte_scores, np.nan)

    # Guarantee 2D arrays.
    xtr_scores = np.atleast_2d(xtr_scores)
    xte_scores = np.atleast_2d(xte_scores)
    ytr_scores = np.atleast_2d(ytr_scores)
    yte_scores = np.atleast_2d(yte_scores)
    if xtr_scores.shape[0] != len(X_train):
        xtr_scores = xtr_scores.T
    if xte_scores.shape[0] != len(X_test):
        xte_scores = xte_scores.T
    if ytr_scores.shape[0] != len(X_train):
        ytr_scores = ytr_scores.T
    if yte_scores.shape[0] != len(X_test):
        yte_scores = yte_scores.T

    xtr_scores, xte_scores, ytr_scores, yte_scores, _ = orient_scores_by_clinical_burden(
        xtr_scores, xte_scores, ytr_scores, yte_scores, Y_train_raw[used_vars], used_vars
    )

    return AlignmentFitResult(
        x_scores_train=xtr_scores,
        y_scores_train=ytr_scores,
        x_scores_test=xte_scores,
        y_scores_test=yte_scores,
        used_clinical_vars=used_vars,
        n_pca=n_pca,
    )


def run_oof_alignment(
    X: np.ndarray,
    clinical: pd.DataFrame,
    clinical_vars: List[str],
    patient_ids: Sequence[str],
    method: str,
    args,
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
        log(f"Running {method.upper()} acoustic-clinical alignment with {kf.get_n_splits()} outer folds")

    for fold, (tr, te) in enumerate(kf.split(X), start=1):
        if verbose:
            log(f"  {method.upper()} fold {fold}/{kf.get_n_splits()}: train={len(tr)}, test={len(te)}")
        res = fit_alignment_one_fold(
            X_train=X[tr],
            X_test=X[te],
            Y_train_raw=clinical.iloc[tr],
            Y_test_raw=clinical.iloc[te],
            clinical_vars=clinical_vars,
            method=method,
            n_components=n_comp,
            n_pca_requested=args.n_pca,
        )
        m = res.x_scores_test.shape[1]
        x_scores[te, :m] = res.x_scores_test[:, :m]
        y_scores[te, :m] = res.y_scores_test[:, :m]
        folds[te] = fold
        for comp in range(m):
            rho, p, nn = safe_spearman(res.x_scores_test[:, comp], res.y_scores_test[:, comp])
            r, rp, _ = safe_pearson(res.x_scores_test[:, comp], res.y_scores_test[:, comp])
            fold_rows.append({
                "method": method,
                "fold": fold,
                "component": comp + 1,
                "n_test": nn,
                "test_spearman_x_y_score": rho,
                "test_spearman_p": p,
                "test_pearson_x_y_score": r,
                "test_pearson_p": rp,
                "n_pca": res.n_pca,
                "used_clinical_vars": ";".join(res.used_clinical_vars),
            })

    score_df = pd.DataFrame({"patient_id": list(patient_ids), "fold": folds})
    for comp in range(n_comp):
        score_df[f"{method}_acoustic_axis{comp+1}"] = x_scores[:, comp]
        score_df[f"{method}_clinical_axis{comp+1}"] = y_scores[:, comp]
    fold_df = pd.DataFrame(fold_rows)
    return score_df, fold_df


# -----------------------------
# Experiment 1: single-variable readout
# -----------------------------

def single_variable_readout(
    X: np.ndarray,
    clinical: pd.DataFrame,
    clinical_vars: List[str],
    args,
) -> pd.DataFrame:
    log("Experiment: single clinical variables capture only limited acoustic correspondence")
    rows = []
    alphas = np.logspace(-3, 3, 13)

    for i, c in enumerate(clinical_vars, start=1):
        y_all = clinical[c].to_numpy(dtype=float)
        mask = np.isfinite(y_all)
        n = int(mask.sum())
        if n < args.min_target_n or np.nanstd(y_all[mask]) < 1e-12:
            rows.append({"variable": c, "n": n, "status": "skipped_too_few_or_constant"})
            continue
        Xc = X[mask]
        yc = y_all[mask]
        n_splits = min(args.n_splits, n)
        if n_splits < 3:
            rows.append({"variable": c, "n": n, "status": "skipped_too_few_folds"})
            continue
        preds = np.full(n, np.nan)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        log(f"  [{i}/{len(clinical_vars)}] CV Ridge readout: {c} (n={n})")
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
        try:
            r2 = r2_score(yc[np.isfinite(preds)], preds[np.isfinite(preds)])
        except Exception:
            r2 = np.nan
        try:
            mae = mean_absolute_error(yc[np.isfinite(preds)], preds[np.isfinite(preds)])
        except Exception:
            mae = np.nan
        ci_l, ci_u = bootstrap_ci_spearman(preds, yc, n_boot=args.n_bootstrap, seed=args.seed + i)
        rows.append({
            "variable": c,
            "n": nn,
            "spearman_pred_true": rho,
            "spearman_p": p,
            "spearman_ci95_low": ci_l,
            "spearman_ci95_high": ci_u,
            "pearson_pred_true": r,
            "pearson_p": rp,
            "r2": r2,
            "mae": mae,
            "status": "ok",
        })
    out = pd.DataFrame(rows)
    if "spearman_p" in out.columns:
        out["spearman_fdr"] = fdr_bh(out["spearman_p"].values)
    return out


# -----------------------------
# Experiment 2-4: alignment, interpretation, gradient
# -----------------------------

def summarize_alignment_scores(
    score_df: pd.DataFrame,
    methods: Sequence[str],
    args,
) -> pd.DataFrame:
    rows = []
    for method in methods:
        for comp in range(1, args.n_components + 1):
            xcol = f"{method}_acoustic_axis{comp}"
            ycol = f"{method}_clinical_axis{comp}"
            if xcol not in score_df.columns or ycol not in score_df.columns:
                continue
            rho, p, n = safe_spearman(score_df[xcol], score_df[ycol])
            r, rp, _ = safe_pearson(score_df[xcol], score_df[ycol])
            ci_l, ci_u = bootstrap_ci_spearman(score_df[xcol], score_df[ycol], args.n_bootstrap, args.seed + 100 + comp)
            rows.append({
                "method": method,
                "component": comp,
                "n": n,
                "spearman_acoustic_vs_clinical_axis": rho,
                "spearman_p": p,
                "spearman_ci95_low": ci_l,
                "spearman_ci95_high": ci_u,
                "pearson_acoustic_vs_clinical_axis": r,
                "pearson_p": rp,
            })
    out = pd.DataFrame(rows)
    if "spearman_p" in out.columns:
        out["spearman_fdr"] = fdr_bh(out["spearman_p"].values)
    return out


def acoustic_axis_clinical_associations(
    score_df: pd.DataFrame,
    clinical: pd.DataFrame,
    clinical_vars: List[str],
    methods: Sequence[str],
    args,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    fold_rows = []
    for method in methods:
        for comp in range(1, args.n_components + 1):
            xcol = f"{method}_acoustic_axis{comp}"
            if xcol not in score_df.columns:
                continue
            x = score_df[xcol].values
            for j, c in enumerate(clinical_vars):
                rho, p, n = safe_spearman(x, clinical[c].values)
                ci_l, ci_u = bootstrap_ci_spearman(
                    x, clinical[c].values, n_boot=args.n_bootstrap, seed=args.seed + 1000 + comp * 100 + j
                )
                rows.append({
                    "method": method,
                    "component": comp,
                    "variable": c,
                    "n": n,
                    "spearman_axis_variable": rho,
                    "spearman_p": p,
                    "spearman_ci95_low": ci_l,
                    "spearman_ci95_high": ci_u,
                    "burden_orientation": clinical_burden_direction(c),
                })
                # Fold-level sign consistency: association within each held-out fold.
                signs = []
                fold_rhos = []
                for fold in sorted(score_df["fold"].dropna().unique()):
                    m = score_df["fold"].values == fold
                    rr, pp, nn = safe_spearman(x[m], clinical.loc[m, c].values)
                    if np.isfinite(rr):
                        fold_rhos.append(rr)
                        signs.append(np.sign(rr))
                        fold_rows.append({
                            "method": method,
                            "component": comp,
                            "variable": c,
                            "fold": int(fold),
                            "n": nn,
                            "fold_spearman_axis_variable": rr,
                            "fold_spearman_p": pp,
                        })
    out = pd.DataFrame(rows)
    if len(out) > 0:
        out["spearman_fdr_within_method_component"] = np.nan
        for (m, c), idx in out.groupby(["method", "component"]).groups.items():
            out.loc[idx, "spearman_fdr_within_method_component"] = fdr_bh(out.loc[idx, "spearman_p"].values)
    fold_out = pd.DataFrame(fold_rows)
    return out, fold_out


def clinical_gradient_by_axis(
    score_df: pd.DataFrame,
    clinical: pd.DataFrame,
    clinical_vars: List[str],
    method: str,
    component: int = 1,
    n_groups: int = 4,
) -> pd.DataFrame:
    xcol = f"{method}_acoustic_axis{component}"
    scores = score_df[xcol].values
    valid = np.isfinite(scores)
    # qcut can fail with repeated values; rank first.
    ranks = pd.Series(scores[valid]).rank(method="first")
    groups = pd.qcut(ranks, q=n_groups, labels=[f"Q{i+1}" for i in range(n_groups)])
    tmp = clinical.loc[valid, ["patient_id"] + clinical_vars].copy()
    tmp["axis_score"] = scores[valid]
    tmp["axis_group"] = groups.to_numpy()

    rows = []
    for c in clinical_vars:
        vals = tmp[c].astype(float)
        rho, p, n = safe_spearman(tmp["axis_score"].values, vals.values)
        for g, sub in tmp.groupby("axis_group", observed=False):
            v = sub[c].astype(float)
            rows.append({
                "method": method,
                "component": component,
                "variable": c,
                "axis_group": str(g),
                "n": int(v.notna().sum()),
                "median": float(np.nanmedian(v)) if v.notna().sum() else np.nan,
                "mean": float(np.nanmean(v)) if v.notna().sum() else np.nan,
                "q25": float(np.nanpercentile(v.dropna(), 25)) if v.notna().sum() else np.nan,
                "q75": float(np.nanpercentile(v.dropna(), 75)) if v.notna().sum() else np.nan,
                "axis_variable_spearman": rho,
                "axis_variable_p": p,
                "axis_variable_n": n,
            })
    return pd.DataFrame(rows)


# -----------------------------
# Experiment 5: endpoint validation
# -----------------------------

def find_endpoint_columns(clinical: pd.DataFrame) -> Dict[str, str]:
    cols = [c for c in clinical.columns if c != "patient_id"]
    endpoint_cols = {}
    ef = select_first_matching_column(cols, [r"^ef$", r"lvef", r"ejection", r"射血"])
    if ef:
        endpoint_cols["EF_lt_40"] = ef
    nt = select_first_matching_column(cols, [r"nt[-_ ]?pro[-_ ]?bnp", r"ntprobnp", r"nt.*bnp", r"脑钠", r"^bnp$"])
    if nt:
        endpoint_cols["NTproBNP_ge_300"] = nt
    nyha = select_first_matching_column(cols, [r"nyha"])
    if nyha:
        endpoint_cols["NYHA_ge_3"] = nyha
    return endpoint_cols


def make_endpoint_series(name: str, col: str, clinical: pd.DataFrame) -> Tuple[pd.Series, float, str]:
    v = clinical[col].astype(float)
    if name == "EF_lt_40":
        maxv = np.nanmax(v.values)
        threshold = 0.40 if maxv <= 1.5 else 40.0
        y = (v < threshold).astype(float)
        return y, threshold, f"{col} < {threshold:g}"
    if name == "NTproBNP_ge_300":
        maxv = np.nanmax(v.values)
        lc = str(col).lower()
        if "log10" in lc or ("log" in lc and maxv <= 5.0):
            threshold = float(np.log10(300.0))
        elif "log" in lc or "ln" in lc:
            threshold = float(np.log1p(300.0))
        else:
            threshold = 300.0
        y = (v >= threshold).astype(float)
        return y, threshold, f"{col} >= {threshold:g}"
    if name == "NYHA_ge_3":
        threshold = 3.0
        y = (v >= threshold).astype(float)
        return y, threshold, f"{col} >= 3"
    raise ValueError(name)


def endpoint_exclude_vars(endpoint_name: str, endpoint_col: str, clinical_vars: List[str]) -> List[str]:
    exclude = {endpoint_col}
    if endpoint_name.startswith("EF"):
        pats = [r"^ef$", r"lvef", r"ejection", r"射血"]
    elif endpoint_name.startswith("NTproBNP"):
        pats = [r"nt.*bnp", r"ntprobnp", r"bnp", r"脑钠"]
    elif endpoint_name.startswith("NYHA"):
        pats = [r"nyha"]
    else:
        pats = []
    for c in clinical_vars:
        if any(re.search(p, str(c), flags=re.I) for p in pats):
            exclude.add(c)
    return [c for c in clinical_vars if c not in exclude]


def endpoint_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(prob) & np.isfinite(y_true)
    y = y_true[mask].astype(int)
    p = prob[mask]
    pred = (p >= 0.5).astype(int)
    out = {"n": int(len(y)), "n_positive": int(y.sum()), "positive_rate": float(y.mean()) if len(y) else np.nan}
    if len(y) == 0 or len(np.unique(y)) < 2:
        out.update({"auroc": np.nan, "accuracy": np.nan, "balanced_accuracy": np.nan, "sensitivity": np.nan, "specificity": np.nan})
        return out
    out["auroc"] = float(roc_auc_score(y, p))
    out["accuracy"] = float(accuracy_score(y, pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    return out


def run_endpoint_validation(
    X: np.ndarray,
    clinical: pd.DataFrame,
    clinical_vars: List[str],
    endpoints: Dict[str, str],
    methods: Sequence[str],
    args,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log("Experiment: endpoint validation with clinically anchored axes")
    summary_rows = []
    pred_rows = []
    endpoint_info_rows = []

    for endpoint_name, endpoint_col in endpoints.items():
        y_series, threshold, rule = make_endpoint_series(endpoint_name, endpoint_col, clinical)
        valid = y_series.notna() & clinical[endpoint_col].notna()
        y_all = y_series[valid].astype(int).to_numpy()
        if len(y_all) < args.min_endpoint_n or len(np.unique(y_all)) < 2:
            log(f"  Skip endpoint {endpoint_name}: too few samples or one class only")
            continue
        pos = int(y_all.sum())
        neg = int(len(y_all) - pos)
        if min(pos, neg) < args.min_endpoint_class_n:
            log(f"  Skip endpoint {endpoint_name}: min class count={min(pos, neg)} < {args.min_endpoint_class_n}")
            continue
        endpoint_info_rows.append({
            "endpoint": endpoint_name,
            "source_column": endpoint_col,
            "rule": rule,
            "threshold": threshold,
            "n": len(y_all),
            "n_positive": pos,
            "positive_rate": pos / len(y_all),
        })

        X_e = X[valid.values]
        clinical_e = clinical.loc[valid].reset_index(drop=True)
        patient_ids_e = clinical_e["patient_id"].tolist()

        for method in methods:
            # More rigorous endpoint validation: remove the endpoint's own variable from anchoring panel.
            if args.leave_endpoint_out:
                panel_vars = endpoint_exclude_vars(endpoint_name, endpoint_col, clinical_vars)
                panel_type = "leave_endpoint_out"
                if len(panel_vars) < args.min_clinical_vars:
                    panel_vars = clinical_vars
                    panel_type = "full_panel_fallback"
            else:
                panel_vars = clinical_vars
                panel_type = "full_panel"

            n_splits = min(args.n_splits, len(y_all), pos, neg)
            if n_splits < 3:
                log(f"  Skip endpoint {endpoint_name}/{method}: too few folds")
                continue
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

            max_k = min(args.n_components, 2)
            probs_by_k = {k: np.full(len(y_all), np.nan) for k in range(1, max_k + 1)}
            folds = np.full(len(y_all), -1)

            log(f"  Endpoint {endpoint_name}, method={method.upper()}, panel={panel_type}, n={len(y_all)}, pos={pos}")
            for fold, (tr, te) in enumerate(skf.split(X_e, y_all), start=1):
                res = fit_alignment_one_fold(
                    X_train=X_e[tr],
                    X_test=X_e[te],
                    Y_train_raw=clinical_e.iloc[tr],
                    Y_test_raw=clinical_e.iloc[te],
                    clinical_vars=panel_vars,
                    method=method,
                    n_components=args.n_components,
                    n_pca_requested=args.n_pca,
                )
                folds[te] = fold
                for k in range(1, max_k + 1):
                    if res.x_scores_train.shape[1] < k:
                        continue
                    clf = Pipeline([
                        ("scaler", StandardScaler()),
                        ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
                    ])
                    clf.fit(res.x_scores_train[:, :k], y_all[tr])
                    probs_by_k[k][te] = clf.predict_proba(res.x_scores_test[:, :k])[:, 1]

            for k, prob in probs_by_k.items():
                m = endpoint_metrics(y_all, prob)
                auc_l, auc_u = bootstrap_ci_auc(y_all, prob, n_boot=args.n_bootstrap, seed=args.seed + 7000 + k)
                summary_rows.append({
                    "endpoint": endpoint_name,
                    "source_column": endpoint_col,
                    "rule": rule,
                    "method": method,
                    "panel_type": panel_type,
                    "n_axis_features": k,
                    "clinical_panel_vars": ";".join(panel_vars),
                    **m,
                    "auroc_ci95_low": auc_l,
                    "auroc_ci95_high": auc_u,
                })
                for pid, y, p, fold in zip(patient_ids_e, y_all, prob, folds):
                    pred_rows.append({
                        "patient_id": pid,
                        "endpoint": endpoint_name,
                        "method": method,
                        "panel_type": panel_type,
                        "n_axis_features": k,
                        "fold": int(fold),
                        "y_true": int(y),
                        "y_prob": float(p) if np.isfinite(p) else np.nan,
                    })

    return pd.DataFrame(summary_rows), pd.DataFrame(pred_rows), pd.DataFrame(endpoint_info_rows)


# -----------------------------
# Negative controls
# -----------------------------

def run_negative_controls(
    X: np.ndarray,
    clinical: pd.DataFrame,
    clinical_vars: List[str],
    patient_ids: Sequence[str],
    observed_alignment: pd.DataFrame,
    args,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Minimal controls: permuted patient labels and random embeddings, for the primary method only."""
    method = args.control_method.lower()
    log(f"Negative controls for primary method={method.upper()}")
    rows = []
    rng = np.random.default_rng(args.seed)

    obs_row = observed_alignment[(observed_alignment["method"] == method) & (observed_alignment["component"] == 1)]
    observed = float(obs_row["spearman_acoustic_vs_clinical_axis"].iloc[0]) if len(obs_row) else np.nan

    # Control 1: permute clinical rows relative to acoustic rows.
    for i in range(args.n_permutations):
        if (i + 1) % max(1, args.progress_every) == 0 or i == 0:
            log(f"  Permutation control {i+1}/{args.n_permutations}")
        perm_idx = rng.permutation(len(clinical))
        clinical_perm = clinical.iloc[perm_idx].reset_index(drop=True)
        try:
            score_df, _ = run_oof_alignment(
                X, clinical_perm, clinical_vars, patient_ids, method, args, verbose=False
            )
            rho, p, n = safe_spearman(score_df[f"{method}_acoustic_axis1"], score_df[f"{method}_clinical_axis1"])
        except Exception as e:
            rho, p, n = np.nan, np.nan, 0
        rows.append({"control_type": "patient_label_permutation", "iteration": i + 1, "method": method, "component": 1, "spearman": rho, "p": p, "n": n})

    # Control 2: random embeddings with the same shape.
    for i in range(args.n_random_controls):
        log(f"  Random-embedding control {i+1}/{args.n_random_controls}")
        X_rand = rng.normal(size=X.shape).astype(np.float32)
        try:
            score_df, _ = run_oof_alignment(
                X_rand, clinical, clinical_vars, patient_ids, method, args, verbose=False
            )
            rho, p, n = safe_spearman(score_df[f"{method}_acoustic_axis1"], score_df[f"{method}_clinical_axis1"])
        except Exception:
            rho, p, n = np.nan, np.nan, 0
        rows.append({"control_type": "random_embedding", "iteration": i + 1, "method": method, "component": 1, "spearman": rho, "p": p, "n": n})

    ctrl = pd.DataFrame(rows)
    summary_rows = []
    for ct, sub in ctrl.groupby("control_type"):
        vals = sub["spearman"].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        # One-sided p: how often control reaches or exceeds observed absolute magnitude.
        empirical_p = (1 + np.sum(np.abs(vals) >= abs(observed))) / (1 + len(vals)) if np.isfinite(observed) else np.nan
        summary_rows.append({
            "method": method,
            "component": 1,
            "control_type": ct,
            "observed_spearman": observed,
            "control_n": len(vals),
            "control_mean": float(np.mean(vals)),
            "control_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "control_p2_5": float(np.percentile(vals, 2.5)),
            "control_p97_5": float(np.percentile(vals, 97.5)),
            "empirical_p_abs_ge_observed": empirical_p,
        })
    return ctrl, pd.DataFrame(summary_rows)


# -----------------------------
# Plotting
# -----------------------------

def setup_plotting() -> None:
    if HAS_SEABORN:
        sns.set_theme(style="white", context="talk")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.spines.top"] = True
    plt.rcParams["axes.spines.right"] = True
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.grid"] = False


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"Saved figure: {path}")


def plot_single_variable_readout(df: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    d = df[df.get("status", "ok") == "ok"].copy()
    if len(d) == 0 or "spearman_pred_true" not in d.columns:
        return
    d = d[np.isfinite(d["spearman_pred_true"])].copy()
    if len(d) == 0:
        return
    d["abs_rho"] = d["spearman_pred_true"].abs()
    d = d.sort_values("abs_rho", ascending=False).head(top_n).sort_values("spearman_pred_true")
    plt.figure(figsize=(9, max(5, 0.38 * len(d))))
    if HAS_SEABORN:
        ax = sns.barplot(data=d, x="spearman_pred_true", y="variable", orient="h")
    else:
        ax = plt.gca()
        ax.barh(d["variable"], d["spearman_pred_true"])
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Out-of-fold Spearman correlation\n(predicted clinical value vs. observed value)")
    ax.set_ylabel("")
    ax.set_title("Single-variable acoustic readout is limited")
    savefig(out_path)


def plot_alignment_summary(df: pd.DataFrame, out_path: Path) -> None:
    if len(df) == 0:
        return
    d = df.copy()
    d["label"] = d["method"].str.upper() + " axis " + d["component"].astype(str)
    plt.figure(figsize=(8.5, max(4, 0.6 * len(d))))
    ax = plt.gca()
    y = np.arange(len(d))
    x = d["spearman_acoustic_vs_clinical_axis"].to_numpy(dtype=float)
    lo = d["spearman_ci95_low"].to_numpy(dtype=float)
    hi = d["spearman_ci95_high"].to_numpy(dtype=float)
    ax.barh(y, x)
    if np.isfinite(lo).any() and np.isfinite(hi).any():
        ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt="none", color="black", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Out-of-fold Spearman correlation\n(acoustic axis score vs. clinical axis score)")
    ax.set_title("Multivariate acoustic-clinical alignment")
    savefig(out_path)


def plot_axis_clinical_forest(
    assoc_df: pd.DataFrame,
    out_path: Path,
    method: str,
    component: int = 1,
    top_n: int = 18,
) -> None:
    d = assoc_df[(assoc_df["method"] == method) & (assoc_df["component"] == component)].copy()
    d = d[np.isfinite(d["spearman_axis_variable"])]
    if len(d) == 0:
        return
    d["abs_rho"] = d["spearman_axis_variable"].abs()
    d = d.sort_values("abs_rho", ascending=False).head(top_n).sort_values("spearman_axis_variable")
    plt.figure(figsize=(9.5, max(5, 0.42 * len(d))))
    ax = plt.gca()
    y = np.arange(len(d))
    x = d["spearman_axis_variable"].to_numpy(dtype=float)
    lo = d["spearman_ci95_low"].to_numpy(dtype=float)
    hi = d["spearman_ci95_high"].to_numpy(dtype=float)
    ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt="o", capsize=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["variable"])
    ax.set_xlabel("Spearman correlation with acoustic axis score")
    ax.set_title(f"Clinical interpretation of {method.upper()} acoustic axis {component}")
    savefig(out_path)


def plot_clinical_gradient_heatmap(
    gradient_df: pd.DataFrame,
    assoc_df: pd.DataFrame,
    out_path: Path,
    method: str,
    component: int = 1,
    top_n: int = 10,
) -> None:
    assoc = assoc_df[(assoc_df["method"] == method) & (assoc_df["component"] == component)].copy()
    assoc = assoc[np.isfinite(assoc["spearman_axis_variable"])]
    if len(assoc) == 0:
        return
    assoc["abs_rho"] = assoc["spearman_axis_variable"].abs()
    top_vars = assoc.sort_values("abs_rho", ascending=False).head(top_n)["variable"].tolist()
    d = gradient_df[(gradient_df["method"] == method) & (gradient_df["component"] == component) & (gradient_df["variable"].isin(top_vars))]
    if len(d) == 0:
        return
    mat = d.pivot_table(index="variable", columns="axis_group", values="median", aggfunc="first")
    # Row-wise z-score for visualizing monotonic gradients across axis quantiles.
    mat_z = mat.copy().astype(float)
    for idx in mat_z.index:
        vals = mat_z.loc[idx].to_numpy(dtype=float)
        mu = np.nanmean(vals)
        sd = np.nanstd(vals)
        if np.isfinite(sd) and sd > 1e-12:
            mat_z.loc[idx] = (vals - mu) / sd
        else:
            mat_z.loc[idx] = 0.0
    mat_z = mat_z.loc[top_vars]
    plt.figure(figsize=(8.5, max(4.5, 0.45 * len(mat_z))))
    if HAS_SEABORN:
        ax = sns.heatmap(mat_z, annot=True, fmt=".2f", linewidths=0.5, cbar_kws={"label": "Row-wise z-scored median"})
    else:
        ax = plt.gca()
        im = ax.imshow(mat_z.values, aspect="auto")
        plt.colorbar(im, ax=ax, label="Row-wise z-scored median")
        ax.set_xticks(np.arange(mat_z.shape[1])); ax.set_xticklabels(mat_z.columns)
        ax.set_yticks(np.arange(mat_z.shape[0])); ax.set_yticklabels(mat_z.index)
    ax.set_xlabel(f"{method.upper()} acoustic axis {component} quantile")
    ax.set_ylabel("")
    ax.set_title("Clinical gradient across acoustic-clinical axis")
    savefig(out_path)


def plot_endpoint_summary(df: pd.DataFrame, out_path: Path) -> None:
    if len(df) == 0:
        return
    d = df.copy()
    d = d[np.isfinite(d["auroc"])]
    if len(d) == 0:
        return
    d["label"] = d["endpoint"] + "\n" + d["method"].str.upper() + f" axis" + d["n_axis_features"].astype(str)
    d = d.sort_values(["endpoint", "method", "n_axis_features"])
    plt.figure(figsize=(max(8, 0.65 * len(d)), 5.5))
    ax = plt.gca()
    x = np.arange(len(d))
    y = d["auroc"].to_numpy(dtype=float)
    lo = d["auroc_ci95_low"].to_numpy(dtype=float)
    hi = d["auroc_ci95_high"].to_numpy(dtype=float)
    ax.bar(x, y)
    if np.isfinite(lo).any() and np.isfinite(hi).any():
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", color="black", capsize=3)
    ax.axhline(0.5, color="black", lw=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(d["label"], rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title("Endpoint validation of clinically anchored axes")
    savefig(out_path)


def plot_negative_controls(ctrl: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    if len(ctrl) == 0 or len(summary) == 0:
        return
    method = summary["method"].iloc[0]
    observed = summary["observed_spearman"].iloc[0]
    plt.figure(figsize=(8, 5.5))
    if HAS_SEABORN:
        ax = sns.histplot(data=ctrl, x="spearman", hue="control_type", bins=25, element="step", stat="count", common_norm=False)
    else:
        ax = plt.gca()
        for ct, sub in ctrl.groupby("control_type"):
            ax.hist(sub["spearman"].dropna(), bins=25, alpha=0.5, label=ct)
        ax.legend()
    ax.axvline(observed, color="black", lw=2, label="observed")
    ax.axvline(-observed, color="black", lw=1, linestyle="--")
    ax.set_xlabel("Out-of-fold Spearman correlation\n(acoustic axis score vs. clinical axis score)")
    ax.set_title(f"Negative controls for {method.upper()} axis 1")
    savefig(out_path)


# -----------------------------
# Report
# -----------------------------

def write_markdown_report(
    out_dir: Path,
    config: Dict,
    clinical_vars: List[str],
    single_df: pd.DataFrame,
    alignment_summary: pd.DataFrame,
    assoc_df: pd.DataFrame,
    endpoint_summary: pd.DataFrame,
    neg_summary: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# Clinically anchored acoustic phenotyping summary\n")
    lines.append("## Analysis design\n")
    lines.append("This script evaluates whether patient-level multi-site heart-sound embeddings contain low-dimensional acoustic phenotype axes that align with multidimensional clinical status.\n")
    lines.append("Main tests: single-variable readout, multivariate CCA/PLS alignment, clinical-axis interpretation, endpoint validation, and minimal negative controls.\n")
    lines.append("## Clinical variables used for anchoring\n")
    for c in clinical_vars:
        lines.append(f"- {c}\n")

    lines.append("\n## Key result tables\n")
    lines.append("- `tables/single_variable_readout.csv`: leakage-safe direct readout of each clinical variable from acoustic embeddings.\n")
    lines.append("- `tables/alignment_summary.csv`: out-of-fold acoustic-clinical axis alignment.\n")
    lines.append("- `tables/acoustic_axis_clinical_associations.csv`: clinical interpretation of acoustic axes.\n")
    lines.append("- `tables/clinical_gradient_by_axis.csv`: clinical profile across axis quantiles.\n")
    lines.append("- `tables/endpoint_validation_summary.csv`: EF/NT-proBNP/NYHA endpoint validation.\n")
    lines.append("- `tables/negative_control_summary.csv`: patient-label permutation and random-embedding controls.\n")

    if len(single_df) and "spearman_pred_true" in single_df.columns:
        d = single_df[single_df.get("status", "ok") == "ok"].copy()
        d = d[np.isfinite(d["spearman_pred_true"])]
        if len(d):
            top = d.reindex(d["spearman_pred_true"].abs().sort_values(ascending=False).index).head(5)
            lines.append("\n## Top single-variable acoustic readouts\n")
            lines.append(top[["variable", "n", "spearman_pred_true", "spearman_ci95_low", "spearman_ci95_high"]].pipe(df_to_markdown_safe, index=False))
            lines.append("\n")

    if len(alignment_summary):
        lines.append("\n## Multivariate alignment summary\n")
        cols = ["method", "component", "n", "spearman_acoustic_vs_clinical_axis", "spearman_ci95_low", "spearman_ci95_high", "spearman_p", "spearman_fdr"]
        cols = [c for c in cols if c in alignment_summary.columns]
        lines.append(alignment_summary[cols].pipe(df_to_markdown_safe, index=False))
        lines.append("\n")

    if len(assoc_df):
        lines.append("\n## Top clinical associations for acoustic axis 1\n")
        for method in sorted(assoc_df["method"].dropna().unique()):
            sub = assoc_df[(assoc_df["method"] == method) & (assoc_df["component"] == 1)].copy()
            if len(sub):
                sub["abs_rho"] = sub["spearman_axis_variable"].abs()
                top = sub.sort_values("abs_rho", ascending=False).head(8)
                lines.append(f"\n### {method.upper()} axis 1\n")
                cols = ["variable", "n", "spearman_axis_variable", "spearman_ci95_low", "spearman_ci95_high", "spearman_p", "spearman_fdr_within_method_component"]
                lines.append(top[cols].pipe(df_to_markdown_safe, index=False))
                lines.append("\n")

    if len(endpoint_summary):
        lines.append("\n## Endpoint validation\n")
        cols = ["endpoint", "method", "panel_type", "n_axis_features", "n", "n_positive", "auroc", "auroc_ci95_low", "auroc_ci95_high", "balanced_accuracy", "accuracy"]
        cols = [c for c in cols if c in endpoint_summary.columns]
        lines.append(endpoint_summary[cols].pipe(df_to_markdown_safe, index=False))
        lines.append("\n")

    if len(neg_summary):
        lines.append("\n## Negative controls\n")
        lines.append(neg_summary.pipe(df_to_markdown_safe, index=False))
        lines.append("\n")

    lines.append("\n## Configuration\n")
    lines.append("```json\n")
    lines.append(json.dumps(config, indent=2, ensure_ascii=False))
    lines.append("\n```\n")

    report_path = out_dir / "analysis_summary.md"
    report_path.write_text("".join(lines), encoding="utf-8")
    log(f"Saved Markdown summary: {report_path}")


# -----------------------------
# CLI / main
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Clinically anchored acoustic phenotyping from BEATs patient embeddings")

    p.add_argument("--embedding-dir", type=str, default="Representation_learning/embeddings_4_1/beats",
                   help="Directory containing patient_embeddings.npy and patient_meta.csv/patient_order.csv.")
    p.add_argument("--patient-embedding-npy", type=str, default=None,
                   help="Explicit patient-level embedding .npy path, if auto-detection fails.")
    p.add_argument("--patient-meta-csv", type=str, default=None,
                   help="Explicit patient-level meta/order CSV path, if auto-detection fails.")
    p.add_argument("--embedding-patient-id-col", type=str, default=None,
                   help="Patient ID column in embedding meta/order CSV.")

    p.add_argument("--clinical-csv", type=str, default="Clinical_alignment/outputs/prepared/aligned_clinical_clean.csv",
                   help="Processed clinical CSV. Recommended input.")
    p.add_argument("--clinical-xlsx", type=str, default="Data/patient_info.xlsx",
                   help="Raw clinical Excel fallback if --clinical-csv is not used.")
    p.add_argument("--clinical-sheet", type=str, default=0,
                   help="Excel sheet name/index when using --clinical-xlsx.")
    p.add_argument("--patient-id-col", type=str, default=None,
                   help="Patient ID column in clinical table.")
    p.add_argument("--clinical-vars", type=str, default=None,
                   help="Comma-separated clinical variables to use. If omitted, script auto-selects a core clinical panel.")

    p.add_argument("--out-dir", type=str, default="Clinical_alignment/outputs/clinically_anchored_acoustic_phenotyping")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-components", type=int, default=2)
    p.add_argument("--n-pca", type=int, default=50,
                   help="Number of PCA components fitted inside each training fold before CCA/PLS/Ridge.")
    p.add_argument("--methods", type=str, default="cca,pls",
                   help="Comma-separated methods for clinical anchoring: cca,pls")

    p.add_argument("--max-missing", type=float, default=0.50,
                   help="Maximum missing fraction allowed for automatically selected clinical variables.")
    p.add_argument("--min-clinical-vars", type=int, default=5)
    p.add_argument("--max-clinical-vars", type=int, default=20)
    p.add_argument("--min-nonmissing-clinical-vars", type=int, default=3)
    p.add_argument("--include-demographics", action="store_true",
                   help="Include age/sex-like variables in auto-selected clinical panel. Default: exclude.")

    p.add_argument("--min-target-n", type=int, default=50,
                   help="Minimum non-missing samples for single-variable readout.")
    p.add_argument("--min-endpoint-n", type=int, default=80)
    p.add_argument("--min-endpoint-class-n", type=int, default=20)
    p.add_argument("--leave-endpoint-out", action="store_true", default=True,
                   help="For endpoint validation, exclude the endpoint's own variable from the clinical anchoring panel. Default: True.")
    p.add_argument("--no-leave-endpoint-out", dest="leave_endpoint_out", action="store_false")

    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-permutations", type=int, default=100,
                   help="Number of patient-label permutation negative controls. Set 0 to skip.")
    p.add_argument("--n-random-controls", type=int, default=20,
                   help="Number of random-embedding negative controls. Set 0 to skip.")
    p.add_argument("--control-method", type=str, default="cca", choices=["cca", "pls"],
                   help="Primary method used for negative controls.")
    p.add_argument("--progress-every", type=int, default=10)

    return p.parse_args()


def main():
    args = parse_args()
    setup_plotting()
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    ensure_dir(table_dir)
    ensure_dir(fig_dir)

    config = vars(args).copy()
    (out_dir / "analysis_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    # 1) Load and align data.
    log("Loading data")
    X_raw, meta, emb_path, meta_path = load_patient_embeddings(args)
    clinical_raw = read_clinical_table(args)
    clinical_num = build_numeric_clinical_table(clinical_raw)
    X, clinical, patient_df = align_embeddings_and_clinical(X_raw, meta, clinical_num)

    # Remove rows with invalid embeddings.
    finite_x = np.isfinite(X).all(axis=1)
    if not finite_x.all():
        log(f"Dropping {(~finite_x).sum()} patients with non-finite embedding values")
        X = X[finite_x]
        clinical = clinical.loc[finite_x].reset_index(drop=True)
        patient_df = patient_df.loc[finite_x].reset_index(drop=True)

    # 2) Select clinical variables and apply completeness filter.
    clinical_vars = select_clinical_variables(clinical, args)
    X, clinical, patient_df = filter_patients_for_clinical_panel(
        X, clinical, clinical_vars, patient_df, args.min_nonmissing_clinical_vars
    )
    patient_ids = clinical["patient_id"].tolist()

    used_vars_df = pd.DataFrame({
        "variable": clinical_vars,
        "missing_fraction": [clinical[c].isna().mean() for c in clinical_vars],
        "n_nonmissing": [clinical[c].notna().sum() for c in clinical_vars],
        "n_unique": [clinical[c].nunique(dropna=True) for c in clinical_vars],
        "burden_orientation": [clinical_burden_direction(c) for c in clinical_vars],
    })
    used_vars_df.to_csv(table_dir / "clinical_variables_used.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"patient_id": patient_ids}).to_csv(table_dir / "patients_used.csv", index=False, encoding="utf-8-sig")
    log(f"Selected {len(clinical_vars)} clinical variables for anchoring: {clinical_vars}")

    # 3) Single-variable readout.
    single_df = single_variable_readout(X, clinical, clinical_vars, args)
    single_df.to_csv(table_dir / "single_variable_readout.csv", index=False, encoding="utf-8-sig")
    plot_single_variable_readout(single_df, fig_dir / "single_variable_readout_top_spearman.png")

    # 4) CCA/PLS multivariate alignment.
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    all_scores = pd.DataFrame({"patient_id": patient_ids})
    all_scores["fold"] = np.nan
    fold_all = []
    for method in methods:
        score_df, fold_df = run_oof_alignment(X, clinical, clinical_vars, patient_ids, method, args, verbose=True)
        # Merge score columns. Use fold from first method.
        if all_scores["fold"].isna().all():
            all_scores["fold"] = score_df["fold"]
        score_cols = [c for c in score_df.columns if c not in {"patient_id", "fold"}]
        all_scores = all_scores.merge(score_df[["patient_id"] + score_cols], on="patient_id", how="left")
        fold_all.append(fold_df)

    fold_df_all = pd.concat(fold_all, ignore_index=True) if fold_all else pd.DataFrame()
    all_scores.to_csv(table_dir / "oof_acoustic_clinical_axis_scores.csv", index=False, encoding="utf-8-sig")
    fold_df_all.to_csv(table_dir / "fold_level_alignment_summary.csv", index=False, encoding="utf-8-sig")

    alignment_summary = summarize_alignment_scores(all_scores, methods, args)
    alignment_summary.to_csv(table_dir / "alignment_summary.csv", index=False, encoding="utf-8-sig")
    plot_alignment_summary(alignment_summary, fig_dir / "multivariate_alignment_summary.png")

    # 5) Clinical interpretation and gradient.
    assoc_df, fold_assoc_df = acoustic_axis_clinical_associations(all_scores, clinical, clinical_vars, methods, args)
    assoc_df.to_csv(table_dir / "acoustic_axis_clinical_associations.csv", index=False, encoding="utf-8-sig")
    fold_assoc_df.to_csv(table_dir / "fold_level_axis_clinical_associations.csv", index=False, encoding="utf-8-sig")

    gradient_all = []
    for method in methods:
        for comp in range(1, args.n_components + 1):
            if f"{method}_acoustic_axis{comp}" in all_scores.columns:
                grad = clinical_gradient_by_axis(all_scores, clinical, clinical_vars, method=method, component=comp, n_groups=4)
                gradient_all.append(grad)
                if comp == 1:
                    plot_axis_clinical_forest(
                        assoc_df,
                        fig_dir / f"{method}_axis{comp}_clinical_association_forest.png",
                        method=method,
                        component=comp,
                    )
                    plot_clinical_gradient_heatmap(
                        grad,
                        assoc_df,
                        fig_dir / f"{method}_axis{comp}_clinical_gradient_heatmap.png",
                        method=method,
                        component=comp,
                    )
    gradient_df = pd.concat(gradient_all, ignore_index=True) if gradient_all else pd.DataFrame()
    gradient_df.to_csv(table_dir / "clinical_gradient_by_axis.csv", index=False, encoding="utf-8-sig")

    # 6) Endpoint validation.
    endpoints = find_endpoint_columns(clinical)
    endpoint_summary, endpoint_pred, endpoint_info = run_endpoint_validation(
        X, clinical, clinical_vars, endpoints, methods, args
    )
    endpoint_info.to_csv(table_dir / "endpoints_used.csv", index=False, encoding="utf-8-sig")
    endpoint_summary.to_csv(table_dir / "endpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
    endpoint_pred.to_csv(table_dir / "endpoint_validation_predictions.csv", index=False, encoding="utf-8-sig")
    plot_endpoint_summary(endpoint_summary, fig_dir / "endpoint_validation_auroc.png")

    # 7) Negative controls.
    if args.n_permutations > 0 or args.n_random_controls > 0:
        neg_ctrl, neg_summary = run_negative_controls(X, clinical, clinical_vars, patient_ids, alignment_summary, args)
    else:
        neg_ctrl, neg_summary = pd.DataFrame(), pd.DataFrame()
    neg_ctrl.to_csv(table_dir / "negative_controls.csv", index=False, encoding="utf-8-sig")
    neg_summary.to_csv(table_dir / "negative_control_summary.csv", index=False, encoding="utf-8-sig")
    plot_negative_controls(neg_ctrl, neg_summary, fig_dir / "negative_controls_alignment.png")

    # 8) Save clinical table aligned to the analysis cohort for later checking.
    clinical[["patient_id"] + clinical_vars].to_csv(table_dir / "aligned_clinical_panel_used.csv", index=False, encoding="utf-8-sig")

    write_markdown_report(
        out_dir=out_dir,
        config=config,
        clinical_vars=clinical_vars,
        single_df=single_df,
        alignment_summary=alignment_summary,
        assoc_df=assoc_df,
        endpoint_summary=endpoint_summary,
        neg_summary=neg_summary,
    )

    log("Done.")
    log(f"All outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
