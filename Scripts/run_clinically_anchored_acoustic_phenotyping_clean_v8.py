#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean clinically anchored acoustic phenotyping analysis.

This version intentionally uses the fixed column names in
Clinical_alignment/outputs/prepared/aligned_clinical_clean.csv and removes the
old regex-based variable discovery / core-panel logic.

Main leakage control
--------------------
For every cross-validated analysis, all preprocessing steps are fit only in the
training fold:
    - clinical imputation
    - X and Y standardization
    - PCA on acoustic embeddings
    - optional covariate residualization
    - CCA
    - endpoint logistic classifier
The held-out fold is only transformed/predicted.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
import warnings
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
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    roc_auc_score,
    roc_curve,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    "age_residualized": ["age_years"],
    "sex_residualized": ["sex_male"],
    "heart_rate_residualized": ["heart_rate"],
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


# =============================================================================
# Utilities
# =============================================================================

def now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def clean_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", str(s))[:180]


def normalize_patient_id(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return re.sub(r"\s+", "", s)


def safe_spearman(x, y) -> Tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 5 or np.nanstd(x[mask]) < 1e-12 or np.nanstd(y[mask]) < 1e-12:
        return np.nan, np.nan, n
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p), n


def safe_pearson(x, y) -> Tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 5 or np.nanstd(x[mask]) < 1e-12 or np.nanstd(y[mask]) < 1e-12:
        return np.nan, np.nan, n
    r, p = stats.pearsonr(x[mask], y[mask])
    return float(r), float(p), n


def bootstrap_ci_spearman(x, y, n_boot: int, seed: int) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 10 or n_boot <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        if np.nanstd(x[idx]) < 1e-12 or np.nanstd(y[idx]) < 1e-12:
            continue
        rho = stats.spearmanr(x[idx], y[idx]).correlation
        if np.isfinite(rho):
            vals.append(float(rho))
    if len(vals) < 20:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_auc_ci(y_true, y_score, n_boot: int, seed: int) -> Tuple[float, float]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if len(y) < 20 or len(np.unique(y)) < 2 or n_boot <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(float(roc_auc_score(y[idx], s[idx])))
    if len(vals) < 20:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


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


# =============================================================================
# Loading and validation
# =============================================================================

def find_first_existing(base: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        p = base / name
        if p.exists():
            return p
    return None


def infer_patient_col(df: pd.DataFrame, explicit: Optional[str] = None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Patient ID column not found: {explicit}")
        return explicit
    for c in ["patient_id", "patient", "pid", "ID", "id", "subject_id"]:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = str(c).lower()
        if "patient" in lc or lc in {"pid", "id"}:
            return c
    raise ValueError(f"Cannot infer patient ID column. Columns={list(df.columns)}")


def load_embeddings(args) -> Tuple[np.ndarray, pd.DataFrame]:
    emb_dir = Path(args.embedding_dir)
    emb_path = Path(args.patient_embedding_npy) if args.patient_embedding_npy else find_first_existing(
        emb_dir,
        ["patient_embeddings.npy", "patient_embedding.npy", "patient_level_embeddings.npy", "patient_embeds.npy"],
    )
    if emb_path is None or not emb_path.exists():
        raise FileNotFoundError("Cannot find patient-level embedding .npy. Use --patient-embedding-npy.")
    meta_path = Path(args.patient_meta_csv) if args.patient_meta_csv else find_first_existing(
        emb_dir,
        ["patient_meta.csv", "patient_metadata.csv", "patient_embeddings_meta.csv", "patient_order.csv", "patient_ids.csv"],
    )
    if meta_path is None or not meta_path.exists():
        raise FileNotFoundError("Cannot find patient-level meta/order CSV. Use --patient-meta-csv.")
    X = np.load(emb_path).astype(np.float32)
    meta = pd.read_csv(meta_path)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got {X.shape}")
    if len(meta) != X.shape[0]:
        raise ValueError(f"Meta rows {len(meta)} != embedding rows {X.shape[0]}")
    pid_col = infer_patient_col(meta, args.embedding_patient_id_col)
    meta = meta.copy()
    meta[PATIENT_ID_COL] = meta[pid_col].map(normalize_patient_id)
    meta["embedding_row"] = np.arange(len(meta))
    if meta[PATIENT_ID_COL].duplicated().any():
        raise ValueError("Duplicate patient_id in embedding meta")
    log(f"Loaded embeddings: {emb_path}, shape={X.shape}")
    log(f"Loaded embedding meta: {meta_path}, patient column={pid_col}")
    return X, meta[[PATIENT_ID_COL, "embedding_row"]]


def load_clinical(args) -> pd.DataFrame:
    df = pd.read_csv(args.clinical_csv)
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    if PATIENT_ID_COL not in df.columns:
        raise ValueError(f"Clinical CSV must contain '{PATIENT_ID_COL}'")
    df = df.copy()
    df[PATIENT_ID_COL] = df[PATIENT_ID_COL].map(normalize_patient_id)
    needed = [PATIENT_ID_COL] + ALL_CLINICAL_VARS + ["age_years", "sex_male", "heart_rate"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required fixed columns in clinical CSV: {missing}")
    for c in needed:
        if c != PATIENT_ID_COL:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[PATIENT_ID_COL]).drop_duplicates(PATIENT_ID_COL, keep="first")
    log(f"Loaded clinical data: {len(df)} unique patients")
    return df


def align_data(X: np.ndarray, meta: pd.DataFrame, clinical: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    merged = meta.merge(clinical, on=PATIENT_ID_COL, how="inner")
    if len(merged) == 0:
        raise ValueError("No overlapping patient IDs between embeddings and clinical table")
    X_aligned = X[merged["embedding_row"].to_numpy(dtype=int)]
    finite = np.isfinite(X_aligned).all(axis=1)
    if not finite.all():
        log(f"Dropping {(~finite).sum()} patients with non-finite embeddings")
        X_aligned = X_aligned[finite]
        merged = merged.loc[finite].reset_index(drop=True)
    clinical_aligned = merged.drop(columns=["embedding_row"]).reset_index(drop=True)
    patient_ids = clinical_aligned[PATIENT_ID_COL].tolist()
    log(f"Aligned data: n={len(patient_ids)} patients")
    return X_aligned, clinical_aligned, patient_ids


# =============================================================================
# CCA modeling
# =============================================================================

def choose_n_pca(n_train: int, n_features: int, requested: int) -> int:
    return int(max(1, min(requested, n_features, n_train - 2)))


def prepare_covariates(C_train_raw: Optional[pd.DataFrame], C_test_raw: Optional[pd.DataFrame]):
    if C_train_raw is None or C_test_raw is None or C_train_raw.shape[1] == 0:
        return None, None
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    Ctr = imp.fit_transform(C_train_raw)
    Cte = imp.transform(C_test_raw)
    Ctr = sc.fit_transform(Ctr)
    Cte = sc.transform(Cte)
    keep = np.nanstd(Ctr, axis=0) > 1e-12
    if keep.sum() == 0:
        return None, None
    return Ctr[:, keep], Cte[:, keep]


def residualize_train_test(A_train, A_test, C_train, C_test):
    if C_train is None or C_test is None:
        return A_train, A_test
    Dtr = np.column_stack([np.ones(len(C_train)), C_train])
    Dte = np.column_stack([np.ones(len(C_test)), C_test])
    beta, *_ = np.linalg.lstsq(Dtr, A_train, rcond=None)
    return A_train - Dtr @ beta, A_test - Dte @ beta


def orient_axes(xtr, xte, ytr, yte, clinical_train: pd.DataFrame, clinical_vars: List[str]):
    xtr = xtr.copy(); xte = xte.copy(); ytr = ytr.copy(); yte = yte.copy()
    signs = []
    for k in range(xtr.shape[1]):
        evidence = []
        for v in clinical_vars:
            direction = BURDEN_DIRECTION.get(v, 0)
            if direction == 0:
                continue
            rho, _, n = safe_spearman(xtr[:, k], clinical_train[v].values)
            if n >= 20 and np.isfinite(rho):
                evidence.append(direction * rho)
        sign = 1 if (np.nanmean(evidence) if evidence else 0) >= 0 else -1
        signs.append(sign)
        if sign < 0:
            xtr[:, k] *= -1; xte[:, k] *= -1; ytr[:, k] *= -1; yte[:, k] *= -1
    return xtr, xte, ytr, yte, signs


def fit_cca_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    clinical_train: pd.DataFrame,
    clinical_test: pd.DataFrame,
    clinical_vars: List[str],
    args,
    covariates_train: Optional[pd.DataFrame] = None,
    covariates_test: Optional[pd.DataFrame] = None,
):
    # Y imputation is fit on train only.
    y_imp = SimpleImputer(strategy="median")
    Ytr = y_imp.fit_transform(clinical_train[clinical_vars])
    Yte = y_imp.transform(clinical_test[clinical_vars])
    keep = np.nanstd(Ytr, axis=0) > 1e-12
    used_vars = [v for v, k in zip(clinical_vars, keep) if k]
    Ytr = Ytr[:, keep]
    Yte = Yte[:, keep]
    if Ytr.shape[1] < 2:
        raise ValueError("Too few non-constant clinical variables for CCA")

    Ctr, Cte = prepare_covariates(covariates_train, covariates_test)
    Xtr_res, Xte_res = residualize_train_test(np.asarray(X_train, float), np.asarray(X_test, float), Ctr, Cte)
    Ytr_res, Yte_res = residualize_train_test(Ytr, Yte, Ctr, Cte)

    # X scaling and PCA are fit on train only.
    x_scaler = StandardScaler()
    Xtr_s = x_scaler.fit_transform(Xtr_res)
    Xte_s = x_scaler.transform(Xte_res)
    n_pca = choose_n_pca(len(X_train), X_train.shape[1], args.n_pca)
    pca = PCA(n_components=n_pca, random_state=args.seed)
    Xtr_r = pca.fit_transform(Xtr_s)
    Xte_r = pca.transform(Xte_s)

    y_scaler = StandardScaler()
    Ytr_s = y_scaler.fit_transform(Ytr_res)
    Yte_s = y_scaler.transform(Yte_res)

    n_comp = int(min(args.n_components, Xtr_r.shape[1], Ytr_s.shape[1], len(X_train) - 2))
    if n_comp < 1:
        raise ValueError("n_components became <1")
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

    xtr, xte, ytr, yte, signs = orient_axes(xtr, xte, ytr, yte, clinical_train[used_vars], used_vars)
    return {"xtr": xtr, "xte": xte, "ytr": ytr, "yte": yte, "used_vars": used_vars, "n_pca": n_pca, "signs": signs}


def run_oof_cca(
    X: np.ndarray,
    clinical: pd.DataFrame,
    patient_ids: Sequence[str],
    panel_name: str,
    clinical_vars: List[str],
    args,
    adjustment: str = "none",
    covariate_cols: Optional[List[str]] = None,
    verbose: bool = True,
):
    min_nonmissing = min(args.min_nonmissing_clinical_vars, len(clinical_vars))
    valid = clinical[clinical_vars].notna().sum(axis=1).to_numpy() >= min_nonmissing
    Xp = X[valid]
    Cp = clinical.loc[valid].reset_index(drop=True)
    pids = [pid for pid, keep in zip(patient_ids, valid) if keep]
    if len(Cp) < args.min_panel_n:
        raise ValueError(f"Panel {panel_name} retained n={len(Cp)}, below min_panel_n={args.min_panel_n}")
    n = len(Cp)
    kf = KFold(n_splits=min(args.n_splits, n), shuffle=True, random_state=args.seed)
    x_scores = np.full((n, args.n_components), np.nan)
    y_scores = np.full((n, args.n_components), np.nan)
    folds = np.full(n, -1, dtype=int)
    fold_rows = []
    if verbose:
        log(f"OOF CCA: {panel_name}, adjustment={adjustment}, n={n}, vars={clinical_vars}")
    for fold, (tr, te) in enumerate(kf.split(Xp), start=1):
        if verbose:
            log(f"  fold {fold}/{kf.get_n_splits()}: train={len(tr)}, test={len(te)}")
        cov_tr = Cp.iloc[tr][covariate_cols] if covariate_cols else None
        cov_te = Cp.iloc[te][covariate_cols] if covariate_cols else None
        res = fit_cca_fold(Xp[tr], Xp[te], Cp.iloc[tr], Cp.iloc[te], clinical_vars, args, cov_tr, cov_te)
        m = res["xte"].shape[1]
        x_scores[te, :m] = res["xte"][:, :m]
        y_scores[te, :m] = res["yte"][:, :m]
        folds[te] = fold
        for c in range(m):
            rho, p, nn = safe_spearman(res["xte"][:, c], res["yte"][:, c])
            pr, pp, _ = safe_pearson(res["xte"][:, c], res["yte"][:, c])
            fold_rows.append({
                "panel": panel_name,
                "adjustment": adjustment,
                "fold": fold,
                "component": c + 1,
                "n_test": nn,
                "test_spearman_x_y_score": rho,
                "test_spearman_p": p,
                "test_pearson_x_y_score": pr,
                "test_pearson_p": pp,
                "used_clinical_vars": ";".join(res["used_vars"]),
                "n_pca": res["n_pca"],
                "axis_orientation_sign": res["signs"][c] if c < len(res["signs"]) else np.nan,
            })
    score = pd.DataFrame({PATIENT_ID_COL: pids, "fold": folds, "panel": panel_name, "adjustment": adjustment})
    for c in range(args.n_components):
        score[f"cca_acoustic_axis{c+1}"] = x_scores[:, c]
        score[f"cca_clinical_axis{c+1}"] = y_scores[:, c]
    return score, pd.DataFrame(fold_rows), Cp, pids


def summarize_alignment(score_df: pd.DataFrame, clinical: pd.DataFrame, clinical_vars: List[str], args, seed_offset: int = 0) -> pd.DataFrame:
    rows = []
    for c in range(1, args.n_components + 1):
        x = score_df[f"cca_acoustic_axis{c}"]
        y = score_df[f"cca_clinical_axis{c}"]
        rho, p, n = safe_spearman(x, y)
        pr, pp, _ = safe_pearson(x, y)
        lo, hi = bootstrap_ci_spearman(x, y, args.n_bootstrap, args.seed + 101 + seed_offset + c)
        red = []
        for v in clinical_vars:
            r, _, nn = safe_pearson(x, clinical[v].values)
            if nn >= 10 and np.isfinite(r):
                red.append(r ** 2)
        row = {
            "panel": score_df["panel"].iloc[0],
            "adjustment": score_df["adjustment"].iloc[0],
            "component": c,
            "n": n,
            "spearman_acoustic_vs_clinical_axis": rho,
            "spearman_p": p,
            "spearman_ci95_low": lo,
            "spearman_ci95_high": hi,
            "pearson_acoustic_vs_clinical_axis": pr,
            "pearson_p": pp,
            "clinical_redundancy_mean_r2": float(np.nanmean(red)) if red else np.nan,
            "clinical_redundancy_max_abs_r": float(np.sqrt(np.nanmax(red))) if red else np.nan,
        }
        rows.append(row)
    if args.n_components >= 2 and {"cca_acoustic_axis1", "cca_acoustic_axis2"}.issubset(score_df.columns):
        Z = score_df[["cca_acoustic_axis1", "cca_acoustic_axis2"]].to_numpy(float)
        vals = []
        for v in clinical_vars:
            y = clinical[v].to_numpy(float)
            mask = np.isfinite(y) & np.isfinite(Z).all(axis=1)
            if mask.sum() >= 20 and np.nanstd(y[mask]) > 1e-12:
                lr = LinearRegression().fit(Z[mask], y[mask])
                vals.append(r2_score(y[mask], lr.predict(Z[mask])))
        if rows:
            rows[0]["clinical_redundancy_axis1_2_mean_r2"] = float(np.nanmean(vals)) if vals else np.nan
    return pd.DataFrame(rows)


# =============================================================================
# Variable-level readout and axis interpretation
# =============================================================================

def single_variable_readout(X: np.ndarray, clinical: pd.DataFrame, patient_ids: Sequence[str], args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows, pred_rows = [], []
    alphas = np.logspace(-3, 3, 13)
    log("Single-variable acoustic readout")
    for i, v in enumerate(ALL_CLINICAL_VARS, start=1):
        y_all = clinical[v].to_numpy(float)
        mask = np.isfinite(y_all)
        n = int(mask.sum())
        if n < args.min_target_n or np.nanstd(y_all[mask]) < 1e-12:
            rows.append({"variable": v, "n": n, "status": "skipped"})
            continue
        Xv, yv = X[mask], y_all[mask]
        pids = [pid for pid, keep in zip(patient_ids, mask) if keep]
        pred = np.full(n, np.nan)
        kf = KFold(n_splits=min(args.n_splits, n), shuffle=True, random_state=args.seed)
        log(f"  [{i}/{len(ALL_CLINICAL_VARS)}] {v}: n={n}")
        for tr, te in kf.split(Xv):
            n_pca = choose_n_pca(len(tr), Xv.shape[1], args.n_pca)
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_pca, random_state=args.seed)),
                ("ridge", RidgeCV(alphas=alphas)),
            ])
            pipe.fit(Xv[tr], yv[tr])
            pred[te] = pipe.predict(Xv[te])
        rho, p, nn = safe_spearman(pred, yv)
        pr, pp, _ = safe_pearson(pred, yv)
        lo, hi = bootstrap_ci_spearman(pred, yv, args.n_bootstrap, args.seed + 1000 + i)
        rows.append({
            "variable": v,
            "domain": VARIABLE_DOMAIN[v],
            "n": nn,
            "status": "ok",
            "spearman_pred_true": rho,
            "spearman_p": p,
            "spearman_ci95_low": lo,
            "spearman_ci95_high": hi,
            "pearson_pred_true": pr,
            "pearson_p": pp,
            "r2": r2_score(yv[np.isfinite(pred)], pred[np.isfinite(pred)]),
            "mae": mean_absolute_error(yv[np.isfinite(pred)], pred[np.isfinite(pred)]),
        })
        for pid, yt, yp in zip(pids, yv, pred):
            pred_rows.append({PATIENT_ID_COL: pid, "variable": v, "y_true": yt, "y_pred": yp})
    out = pd.DataFrame(rows)
    if "spearman_p" in out.columns:
        out["spearman_fdr"] = fdr_bh(out["spearman_p"].values)
    return out, pd.DataFrame(pred_rows)


def axis_clinical_associations(score_df: pd.DataFrame, clinical: pd.DataFrame, clinical_vars: List[str], args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, fold_rows = [], []
    for c in range(1, args.n_components + 1):
        x = score_df[f"cca_acoustic_axis{c}"]
        for j, v in enumerate(clinical_vars):
            rho, p, n = safe_spearman(x, clinical[v].values)
            lo, hi = bootstrap_ci_spearman(x, clinical[v].values, args.n_bootstrap, args.seed + 2000 + c * 100 + j)
            rows.append({
                "panel": score_df["panel"].iloc[0],
                "adjustment": score_df["adjustment"].iloc[0],
                "component": c,
                "variable": v,
                "domain": VARIABLE_DOMAIN.get(v, "Other"),
                "n": n,
                "spearman_axis_variable": rho,
                "spearman_p": p,
                "spearman_ci95_low": lo,
                "spearman_ci95_high": hi,
                "burden_orientation": BURDEN_DIRECTION.get(v, 0),
                "oriented_spearman": rho * BURDEN_DIRECTION.get(v, 0) if np.isfinite(rho) else np.nan,
            })
            for fold in sorted(pd.unique(score_df["fold"])):
                m = score_df["fold"].to_numpy() == fold
                rr, pp, nn = safe_spearman(score_df.loc[m, f"cca_acoustic_axis{c}"], clinical.loc[m, v])
                fold_rows.append({
                    "panel": score_df["panel"].iloc[0],
                    "adjustment": score_df["adjustment"].iloc[0],
                    "component": c,
                    "variable": v,
                    "fold": int(fold),
                    "n": nn,
                    "fold_spearman_axis_variable": rr,
                    "fold_spearman_p": pp,
                    "fold_oriented_spearman": rr * BURDEN_DIRECTION.get(v, 0) if np.isfinite(rr) else np.nan,
                })
    assoc = pd.DataFrame(rows)
    if len(assoc):
        assoc["spearman_fdr_within_panel_component"] = np.nan
        for _, idx in assoc.groupby(["panel", "adjustment", "component"]).groups.items():
            assoc.loc[idx, "spearman_fdr_within_panel_component"] = fdr_bh(assoc.loc[idx, "spearman_p"].values)
    fold_assoc = pd.DataFrame(fold_rows)
    stab_rows = []
    for (panel, adj, comp, var), sub in fold_assoc.groupby(["panel", "adjustment", "component", "variable"]):
        vals = sub["fold_spearman_axis_variable"].dropna().to_numpy(float)
        oriented = sub["fold_oriented_spearman"].dropna().to_numpy(float)
        stab_rows.append({
            "panel": panel,
            "adjustment": adj,
            "component": comp,
            "variable": var,
            "n_folds_with_valid_rho": len(vals),
            "mean_fold_spearman": float(np.mean(vals)) if len(vals) else np.nan,
            "sd_fold_spearman": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "expected_direction_positive_rate": float(np.mean(oriented > 0)) if len(oriented) else np.nan,
        })
    return assoc, fold_assoc, pd.DataFrame(stab_rows)


def fit_full_cca_scores(X: np.ndarray, clinical: pd.DataFrame, clinical_vars: List[str], args) -> np.ndarray:
    Y = SimpleImputer(strategy="median").fit_transform(clinical[clinical_vars])
    keep = np.nanstd(Y, axis=0) > 1e-12
    vars_used = [v for v, k in zip(clinical_vars, keep) if k]
    Y = Y[:, keep]
    Xs = StandardScaler().fit_transform(X)
    n_pca = choose_n_pca(len(X), X.shape[1], args.n_pca)
    Xr = PCA(n_components=n_pca, random_state=args.seed).fit_transform(Xs)
    Ys = StandardScaler().fit_transform(Y)
    n_comp = int(min(args.n_components, Xr.shape[1], Ys.shape[1], len(X) - 2))
    cca = CCA(n_components=n_comp, max_iter=args.cca_max_iter, scale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cca.fit(Xr, Ys)
        xs, ys = cca.transform(Xr, Ys)
    xs = np.atleast_2d(xs)
    ys = np.atleast_2d(ys)
    if xs.shape[0] != len(X): xs = xs.T
    if ys.shape[0] != len(X): ys = ys.T
    xs, _, ys, _, _ = orient_axes(xs, xs.copy(), ys, ys.copy(), clinical[vars_used], vars_used)
    return xs


def bootstrap_loading_stability(X: np.ndarray, clinical: pd.DataFrame, panel: str, clinical_vars: List[str], args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if args.n_loading_bootstrap <= 0:
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(args.seed + 333)
    rows = []
    n = len(clinical)
    log(f"Bootstrap loading stability: {panel}, B={args.n_loading_bootstrap}")
    for b in range(args.n_loading_bootstrap):
        if (b + 1) % max(1, args.progress_every) == 0 or b == 0:
            log(f"  bootstrap {b+1}/{args.n_loading_bootstrap}")
        idx = rng.integers(0, n, size=n)
        try:
            xs = fit_full_cca_scores(X[idx], clinical.iloc[idx].reset_index(drop=True), clinical_vars, args)
        except Exception:
            continue
        Cb = clinical.iloc[idx].reset_index(drop=True)
        for comp in range(min(args.n_components, xs.shape[1])):
            abs_pairs = []
            local_rows = []
            for v in clinical_vars:
                rho, _, nn = safe_spearman(xs[:, comp], Cb[v].values)
                abs_pairs.append((v, abs(rho) if np.isfinite(rho) else -np.inf))
                local_rows.append({
                    "panel": panel,
                    "bootstrap": b + 1,
                    "component": comp + 1,
                    "variable": v,
                    "n": nn,
                    "spearman_axis_variable": rho,
                    "oriented_spearman": rho * BURDEN_DIRECTION.get(v, 0) if np.isfinite(rho) else np.nan,
                })
            rank_map = {v: r + 1 for r, (v, _) in enumerate(sorted(abs_pairs, key=lambda z: z[1], reverse=True))}
            for row in local_rows:
                row["abs_loading_rank"] = rank_map.get(row["variable"], np.nan)
                rows.append(row)
    boot = pd.DataFrame(rows)
    summary = []
    if len(boot):
        for (panel, comp, var), sub in boot.groupby(["panel", "component", "variable"]):
            vals = sub["spearman_axis_variable"].dropna().to_numpy(float)
            oriented = sub["oriented_spearman"].dropna().to_numpy(float)
            ranks = sub["abs_loading_rank"].dropna().to_numpy(float)
            summary.append({
                "panel": panel,
                "component": comp,
                "variable": var,
                "domain": VARIABLE_DOMAIN.get(var, "Other"),
                "n_boot_valid": len(vals),
                "mean_spearman": float(np.mean(vals)) if len(vals) else np.nan,
                "ci95_low": float(np.percentile(vals, 2.5)) if len(vals) else np.nan,
                "ci95_high": float(np.percentile(vals, 97.5)) if len(vals) else np.nan,
                "positive_rate_raw": float(np.mean(vals > 0)) if len(vals) else np.nan,
                "expected_direction_positive_rate": float(np.mean(oriented > 0)) if len(oriented) else np.nan,
                "median_abs_loading_rank": float(np.median(ranks)) if len(ranks) else np.nan,
            })
    return boot, pd.DataFrame(summary)


# =============================================================================
# Clinical gradient
# =============================================================================

def clinical_gradient_patient_values(score_df: pd.DataFrame, clinical: pd.DataFrame, clinical_vars: List[str], args) -> pd.DataFrame:
    rows = []
    for comp in range(1, args.n_components + 1):
        score_col = f"cca_acoustic_axis{comp}"
        if score_col not in score_df.columns:
            continue
        scores = score_df[score_col].to_numpy(float)
        valid = np.isfinite(scores)
        if valid.sum() < 20:
            continue
        rank = pd.Series(scores[valid]).rank(method="first")
        labels = [f"Q{i+1}" for i in range(args.n_axis_groups)]
        groups = pd.qcut(rank, q=args.n_axis_groups, labels=labels)
        tmp = clinical.loc[valid, [PATIENT_ID_COL] + clinical_vars].copy()
        tmp["axis_score"] = scores[valid]
        tmp["axis_group"] = groups.to_numpy()
        tmp["panel"] = score_df["panel"].iloc[0]
        tmp["adjustment"] = score_df["adjustment"].iloc[0]
        tmp["component"] = comp
        for _, row in tmp.iterrows():
            for v in clinical_vars:
                val = row[v]
                if pd.isna(val):
                    continue
                rows.append({
                    "patient_id": row[PATIENT_ID_COL],
                    "panel": row["panel"],
                    "adjustment": row["adjustment"],
                    "component": comp,
                    "axis_score": row["axis_score"],
                    "axis_group": row["axis_group"],
                    "variable": v,
                    "domain": VARIABLE_DOMAIN.get(v, "Other"),
                    "value": float(val),
                    "is_ordinal": v in ORDINAL_VARS,
                    "is_valve": v in VALVE_VARS,
                })
    return pd.DataFrame(rows)


def clinical_gradient_summary(patient_long: pd.DataFrame) -> pd.DataFrame:
    """Summarize clinical gradients and add trend / group-difference tests.

    trend_spearman_p tests monotonic association between the continuous CCA axis
    score and the clinical variable. kruskal_p tests whether Q1-Q4 groups differ.
    Both are descriptive validation statistics; they are not used to fit CCA.
    """
    rows = []
    if len(patient_long) == 0:
        return pd.DataFrame()
    for (panel, adj, comp, var), sub_all in patient_long.groupby(["panel", "adjustment", "component", "variable"]):
        axis = sub_all["axis_score"].to_numpy(float)
        vals_all = sub_all["value"].to_numpy(float)
        rho, sp_p, nn = safe_spearman(axis, vals_all)
        grouped_vals = []
        for g in [f"Q{i+1}" for i in range(4)]:
            arr = sub_all.loc[sub_all["axis_group"] == g, "value"].dropna().to_numpy(float)
            if len(arr):
                grouped_vals.append(arr)
        if len(grouped_vals) >= 2 and all(len(a) >= 2 for a in grouped_vals):
            try:
                kw_stat, kw_p = stats.kruskal(*grouped_vals)
            except Exception:
                kw_stat, kw_p = np.nan, np.nan
        else:
            kw_stat, kw_p = np.nan, np.nan

        for group, sub in sub_all.groupby("axis_group"):
            vals = sub["value"].to_numpy(float)
            n = len(vals)
            is_ord = bool(sub["is_ordinal"].iloc[0]) if n else var in ORDINAL_VARS
            rows.append({
                "panel": panel,
                "adjustment": adj,
                "component": comp,
                "variable": var,
                "domain": VARIABLE_DOMAIN.get(var, "Other"),
                "axis_group": group,
                "n": n,
                "is_ordinal": is_ord,
                "mean": float(np.mean(vals)) if n else np.nan,
                "std": float(np.std(vals, ddof=1)) if n > 1 else np.nan,
                "sem": float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
                "median": float(np.median(vals)) if n else np.nan,
                "q25": float(np.percentile(vals, 25)) if n else np.nan,
                "q75": float(np.percentile(vals, 75)) if n else np.nan,
                "prop_ge_1": float(np.mean(vals >= 1)) if n else np.nan,
                "prop_ge_2": float(np.mean(vals >= 2)) if n else np.nan,
                "prop_ge_3": float(np.mean(vals >= 3)) if n else np.nan,
                "trend_spearman": rho,
                "trend_spearman_p": sp_p,
                "trend_spearman_n": nn,
                "kruskal_stat": kw_stat,
                "kruskal_p": kw_p,
            })
    out = pd.DataFrame(rows)
    if len(out):
        # FDR over unique variable-level tests, then copy values back to Q rows.
        uniq = out.drop_duplicates(["panel", "adjustment", "component", "variable"]).copy()
        uniq["trend_spearman_fdr"] = fdr_bh(uniq["trend_spearman_p"].values)
        uniq["kruskal_fdr"] = fdr_bh(uniq["kruskal_p"].values)
        out = out.merge(
            uniq[["panel", "adjustment", "component", "variable", "trend_spearman_fdr", "kruskal_fdr"]],
            on=["panel", "adjustment", "component", "variable"],
            how="left",
        )
    return out


# =============================================================================
# Negative controls and endpoint validation
# =============================================================================

def permutation_test(X: np.ndarray, clinical: pd.DataFrame, patient_ids: Sequence[str], panel: str, clinical_vars: List[str], observed: float, args):
    rng = np.random.default_rng(args.seed + 444)
    rows = []
    log(f"Permutation control: {panel}, B={args.n_permutations}")
    for b in range(args.n_permutations):
        if (b + 1) % max(1, args.progress_every) == 0 or b == 0:
            log(f"  permutation {b+1}/{args.n_permutations}")
        perm = clinical.iloc[rng.permutation(len(clinical))].reset_index(drop=True)
        try:
            sdf, _, _, _ = run_oof_cca(X, perm, patient_ids, panel, clinical_vars, args, verbose=False)
            rho, p, n = safe_spearman(sdf["cca_acoustic_axis1"], sdf["cca_clinical_axis1"])
        except Exception:
            rho, p, n = np.nan, np.nan, 0
        rows.append({"panel": panel, "control_type": "patient_label_permutation", "iteration": b + 1, "component": 1, "spearman": rho, "p": p, "n": n})
    vals = pd.Series([r["spearman"] for r in rows]).dropna().to_numpy(float)
    summary = {
        "panel": panel,
        "component": 1,
        "control_type": "patient_label_permutation",
        "observed_spearman": observed,
        "control_n": len(vals),
        "control_mean": float(np.mean(vals)) if len(vals) else np.nan,
        "control_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
        "control_p2_5": float(np.percentile(vals, 2.5)) if len(vals) else np.nan,
        "control_p97_5": float(np.percentile(vals, 97.5)) if len(vals) else np.nan,
        "empirical_p_abs_ge_observed": float((1 + np.sum(np.abs(vals) >= abs(observed))) / (1 + len(vals))) if len(vals) and np.isfinite(observed) else np.nan,
    }
    return pd.DataFrame(rows), summary


def random_embedding_control(X: np.ndarray, clinical: pd.DataFrame, patient_ids: Sequence[str], panel: str, clinical_vars: List[str], observed: float, args):
    rng = np.random.default_rng(args.seed + 555)
    rows = []
    log(f"Random-embedding control: {panel}, B={args.n_random_controls}")
    for b in range(args.n_random_controls):
        log(f"  random embedding {b+1}/{args.n_random_controls}")
        Xrand = rng.normal(size=X.shape).astype(np.float32)
        try:
            sdf, _, _, _ = run_oof_cca(Xrand, clinical, patient_ids, panel, clinical_vars, args, verbose=False)
            rho, p, n = safe_spearman(sdf["cca_acoustic_axis1"], sdf["cca_clinical_axis1"])
        except Exception:
            rho, p, n = np.nan, np.nan, 0
        rows.append({"panel": panel, "control_type": "random_embedding", "iteration": b + 1, "component": 1, "spearman": rho, "p": p, "n": n})
    vals = pd.Series([r["spearman"] for r in rows]).dropna().to_numpy(float)
    summary = {
        "panel": panel,
        "component": 1,
        "control_type": "random_embedding",
        "observed_spearman": observed,
        "control_n": len(vals),
        "control_mean": float(np.mean(vals)) if len(vals) else np.nan,
        "control_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
        "control_p2_5": float(np.percentile(vals, 2.5)) if len(vals) else np.nan,
        "control_p97_5": float(np.percentile(vals, 97.5)) if len(vals) else np.nan,
        "empirical_p_abs_ge_observed": float((1 + np.sum(np.abs(vals) >= abs(observed))) / (1 + len(vals))) if len(vals) and np.isfinite(observed) else np.nan,
    }
    return pd.DataFrame(rows), summary


def make_endpoint(clinical: pd.DataFrame, name: str) -> Tuple[pd.Series, str, str]:
    y = pd.Series(np.nan, index=clinical.index, dtype=float)
    if name == "EF_lt_40":
        v = clinical["EF_Teich"]
        y.loc[v.notna()] = (v.loc[v.notna()] < 40).astype(float)
        return y, "EF_Teich", "EF_Teich < 40"
    if name == "NTproBNP_ge_900":
        v = clinical["NTproBNP"]
        threshold = np.log1p(900.0) if np.nanmax(v.values) <= 20 else 900.0
        y.loc[v.notna()] = (v.loc[v.notna()] >= threshold).astype(float)
        return y, "NTproBNP", f"NTproBNP >= {threshold:.4g} (NT-proBNP >=900 if log1p transformed)"
    if name == "NYHA_ge_3":
        v = clinical["NYHA"]
        y.loc[v.notna()] = (v.loc[v.notna()] >= 3).astype(float)
        return y, "NYHA", "NYHA >= 3"
    if name == "LA_ge_40":
        v = clinical["LA_mm"]
        y.loc[v.notna()] = (v.loc[v.notna()] >= 40).astype(float)
        return y, "LA_mm", "LA_mm >= 40 (exploratory LA diameter endpoint)"
    if name == "LVEDD_dilated":
        v = clinical["LVEDD_mm"]
        sex = clinical["sex_male"]
        valid = v.notna() & sex.notna()
        thr = pd.Series(np.where(sex >= 0.5, 58.0, 52.0), index=clinical.index)
        y.loc[valid] = (v.loc[valid] > thr.loc[valid]).astype(float)
        return y, "LVEDD_mm", "LVEDD >58 mm male or >52 mm female"
    raise ValueError(name)


def exclude_endpoint_var(endpoint_source: str, clinical_vars: List[str]) -> List[str]:
    return [v for v in clinical_vars if v != endpoint_source]


def endpoint_metrics(y, prob) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    mask = np.isfinite(prob)
    y = y[mask]
    prob = prob[mask]
    out = {"n": int(len(y)), "n_positive": int(y.sum()) if len(y) else 0, "positive_rate": float(y.mean()) if len(y) else np.nan}
    if len(y) == 0 or len(np.unique(y)) < 2:
        out.update({"auroc": np.nan, "accuracy": np.nan, "balanced_accuracy": np.nan, "sensitivity": np.nan, "specificity": np.nan})
        return out
    pred = (prob >= 0.5).astype(int)
    out["auroc"] = float(roc_auc_score(y, prob))
    out["accuracy"] = float(accuracy_score(y, pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out["sensitivity"] = float(tp / (tp + fn)) if tp + fn else np.nan
    out["specificity"] = float(tn / (tn + fp)) if tn + fp else np.nan
    return out


def run_endpoint_validation(X: np.ndarray, clinical: pd.DataFrame, patient_ids: Sequence[str], args):
    """Leave-endpoint-out endpoint validation comparing CCA axis 1 vs axes 1+2.

    Leakage control is identical for both feature sets:
      1. In each outer training fold, the endpoint-defining clinical variable is
         removed from the CCA anchoring matrix Y.
      2. CCA, all preprocessing, and the logistic classifier are fit on the
         training fold only.
      3. Held-out patients are only transformed and predicted.

    The only difference between the two variants is the number of acoustic CCA
    scores given to the logistic classifier: axis 1 only, or axes 1 and 2.
    """
    endpoints = ["EF_lt_40", "NTproBNP_ge_900", "NYHA_ge_3", "LA_ge_40", "LVEDD_dilated"]
    axis_feature_options = [1, 2]
    summary, pred_rows, info_rows = [], [], []
    for endpoint in endpoints:
        y_series, source, rule = make_endpoint(clinical, endpoint)
        valid = y_series.notna()
        y_all = y_series.loc[valid].astype(int).to_numpy()
        info = {
            "endpoint": endpoint,
            "source_column": source,
            "rule": rule,
            "n": int(valid.sum()),
            "n_positive": int(y_all.sum()) if len(y_all) else 0,
            "positive_rate": float(y_all.mean()) if len(y_all) else np.nan,
        }
        min_class = int(min(y_all.sum(), len(y_all) - y_all.sum())) if len(y_all) else 0
        if len(y_all) < args.min_endpoint_n or len(np.unique(y_all)) < 2 or min_class < args.min_endpoint_class_n:
            info["status"] = "skipped_too_few_for_oof_auc"
            info["min_class_count"] = min_class
            info_rows.append(info)
            log(f"Endpoint skipped: {endpoint}, n={len(y_all)}, pos={int(y_all.sum()) if len(y_all) else 0}")
            continue
        info["status"] = "ok"
        info["min_class_count"] = min_class
        info_rows.append(info)

        Xv = X[valid.to_numpy()]
        Cv = clinical.loc[valid].reset_index(drop=True)
        pids = [pid for pid, keep in zip(patient_ids, valid.to_numpy()) if keep]
        panel_vars = exclude_endpoint_var(source, ALL_CLINICAL_VARS)
        panel_type = "all_clinical_leave_endpoint_out"
        n_splits = min(args.n_splits, int(y_all.sum()), int(len(y_all) - y_all.sum()), len(y_all))
        folds = np.full(len(y_all), -1, dtype=int)
        prob_by_k = {k: np.full(len(y_all), np.nan) for k in axis_feature_options}

        log(f"Endpoint validation: {endpoint}, n={len(y_all)}, pos={int(y_all.sum())}, panel vars={panel_vars}")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        for fold, (tr, te) in enumerate(skf.split(Xv, y_all), start=1):
            # CCA is fit once per outer training fold, then both classifiers use
            # the same fold-specific CCA scores. This avoids any extra leakage.
            res = fit_cca_fold(Xv[tr], Xv[te], Cv.iloc[tr], Cv.iloc[te], panel_vars, args)
            max_k = int(res["xtr"].shape[1])
            for k_requested in axis_feature_options:
                k = min(k_requested, max_k)
                if k < k_requested:
                    continue
                clf = Pipeline([
                    ("scaler", StandardScaler()),
                    ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
                ])
                clf.fit(res["xtr"][:, :k], y_all[tr])
                prob_by_k[k_requested][te] = clf.predict_proba(res["xte"][:, :k])[:, 1]
            folds[te] = fold

        for k_requested in axis_feature_options:
            prob = prob_by_k[k_requested]
            if not np.isfinite(prob).any():
                continue
            m = endpoint_metrics(y_all, prob)
            lo, hi = bootstrap_auc_ci(y_all, prob, args.n_bootstrap, args.seed + 6000 + 17 * k_requested)
            summary.append({
                "endpoint": endpoint,
                "source_column": source,
                "rule": rule,
                "panel": "all_clinical",
                "panel_type": panel_type,
                "n_axis_features": k_requested,
                "axis_feature_set": "axis1" if k_requested == 1 else "axis1_2",
                "clinical_panel_vars": ";".join(panel_vars),
                **m,
                "auroc_ci95_low": lo,
                "auroc_ci95_high": hi,
            })
            for pid, yt, yp, f in zip(pids, y_all, prob, folds):
                pred_rows.append({
                    PATIENT_ID_COL: pid,
                    "endpoint": endpoint,
                    "y_true": int(yt),
                    "y_prob": float(yp),
                    "fold": int(f),
                    "panel_type": panel_type,
                    "n_axis_features": k_requested,
                    "axis_feature_set": "axis1" if k_requested == 1 else "axis1_2",
                })
    return pd.DataFrame(summary), pd.DataFrame(pred_rows), pd.DataFrame(info_rows)


def build_endpoint_axis_feature_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact axis1 vs axis1+2 endpoint comparison table."""
    if summary_df is None or len(summary_df) == 0 or "n_axis_features" not in summary_df.columns:
        return pd.DataFrame()
    rows = []
    metric_cols = [c for c in [
        "auroc", "balanced_accuracy", "accuracy", "sensitivity", "specificity",
        "auroc_mean", "balanced_accuracy_mean", "auroc_median", "balanced_accuracy_median",
        "auroc_p2_5", "auroc_p97_5", "balanced_accuracy_p2_5", "balanced_accuracy_p97_5",
    ] if c in summary_df.columns]
    base_cols = [c for c in ["endpoint", "source_column", "rule", "n", "n_positive", "positive_rate"] if c in summary_df.columns]
    for endpoint, sub in summary_df.groupby("endpoint"):
        rec = {"endpoint": endpoint}
        for c in base_cols:
            if c != "endpoint" and c in sub.columns and len(sub[c].dropna()):
                rec[c] = sub[c].dropna().iloc[0]
        for metric in metric_cols:
            for k in [1, 2]:
                hit = sub[sub["n_axis_features"].astype(int) == k]
                rec[f"{metric}_axis{k if k == 1 else '1_2'}"] = float(hit[metric].iloc[0]) if len(hit) and pd.notna(hit[metric].iloc[0]) else np.nan
            a1 = rec.get(f"{metric}_axis1", np.nan)
            a12 = rec.get(f"{metric}_axis1_2", np.nan)
            rec[f"delta_{metric}_axis1_2_minus_axis1"] = float(a12 - a1) if np.isfinite(a1) and np.isfinite(a12) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# =============================================================================
# Confounder and LOPO controls
# =============================================================================

def run_confounder_controls(X: np.ndarray, clinical: pd.DataFrame, patient_ids: Sequence[str], args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows, assoc_rows = [], []
    for adj, covs in COVARIATE_SETS.items():
        log(f"Confounder control: {adj}")
        sdf, _, Cp, _ = run_oof_cca(X, clinical, patient_ids, "all_clinical", ALL_CLINICAL_VARS, args, adjustment=adj, covariate_cols=covs)
        summ = summarize_alignment(sdf, Cp, ALL_CLINICAL_VARS, args, seed_offset=7000)
        summ["covariates_used"] = ";".join(covs)
        rows.append(summ)
        assoc, _, _ = axis_clinical_associations(sdf, Cp, ALL_CLINICAL_VARS, args)
        assoc["covariates_used"] = ";".join(covs)
        assoc_rows.append(assoc)
    return pd.concat(rows, ignore_index=True), pd.concat(assoc_rows, ignore_index=True)


def load_position_embeddings(args):
    emb_dir = Path(args.embedding_dir)
    emb_path = Path(args.position_embedding_npy) if args.position_embedding_npy else find_first_existing(
        emb_dir, ["position_embeddings.npy", "position_embedding.npy", "position_embeds.npy"]
    )
    meta_path = Path(args.position_meta_csv) if args.position_meta_csv else find_first_existing(
        emb_dir, ["position_meta.csv", "position_metadata.csv", "position_embeddings_meta.csv"]
    )
    if emb_path is None or meta_path is None or not emb_path.exists() or not meta_path.exists():
        log("Position embeddings/meta not found; LOPO skipped")
        return None, None
    Xpos = np.load(emb_path).astype(np.float32)
    meta = pd.read_csv(meta_path)
    if len(meta) != Xpos.shape[0]:
        log("Position meta length mismatch; LOPO skipped")
        return None, None
    pid_col = infer_patient_col(meta, args.position_patient_id_col)
    pos_col = args.position_col if args.position_col and args.position_col in meta.columns else None
    if pos_col is None:
        for c in ["position", "pos", "auscultation_position", "site", "location"]:
            if c in meta.columns:
                pos_col = c
                break
    if pos_col is None:
        log("Cannot infer position column; LOPO skipped")
        return None, None
    meta = meta.copy()
    meta[PATIENT_ID_COL] = meta[pid_col].map(normalize_patient_id)
    meta["position"] = meta[pos_col].astype(str).str.upper().str.strip()
    meta["row"] = np.arange(len(meta))
    log(f"Loaded position embeddings: {emb_path}, shape={Xpos.shape}")
    return Xpos, meta


def build_lopo_X(Xpos: np.ndarray, meta: pd.DataFrame, patient_ids: Sequence[str], leave_pos: str, positions: Sequence[str]):
    rows, pids = [], []
    for pid in patient_ids:
        parts = []
        ok = True
        for pos in positions:
            if pos == leave_pos:
                continue
            sub = meta[(meta[PATIENT_ID_COL] == pid) & (meta["position"] == pos)]
            if len(sub) == 0:
                ok = False
                break
            parts.append(Xpos[sub["row"].to_numpy(dtype=int)].mean(axis=0))
        if ok:
            rows.append(np.concatenate(parts))
            pids.append(pid)
    if not rows:
        return np.empty((0, 0), dtype=np.float32), []
    return np.vstack(rows).astype(np.float32), pids


def run_lopo_control(X: np.ndarray, clinical: pd.DataFrame, patient_ids: Sequence[str], args) -> pd.DataFrame:
    if not args.run_leave_one_position_out:
        return pd.DataFrame()
    Xpos, meta = load_position_embeddings(args)
    if Xpos is None:
        return pd.DataFrame()
    clinical_by_pid = clinical.set_index(PATIENT_ID_COL, drop=False)
    positions = [p.strip().upper() for p in args.positions.split(",") if p.strip()]
    rows = []
    for leave in positions:
        Xl, pids = build_lopo_X(Xpos, meta, patient_ids, leave, positions)
        if len(pids) < args.min_panel_n:
            log(f"LOPO skipped leave={leave}: n={len(pids)}")
            continue
        Cl = clinical_by_pid.loc[pids].reset_index(drop=True)
        try:
            sdf, _, Cp, _ = run_oof_cca(Xl, Cl, pids, f"all_clinical_leave_{leave}", ALL_CLINICAL_VARS, args)
            summ = summarize_alignment(sdf, Cp, ALL_CLINICAL_VARS, args, seed_offset=8000)
            summ["left_out_position"] = leave
            summ["n_positions_used"] = len(positions) - 1
            rows.append(summ)
        except Exception as e:
            log(f"LOPO failed leave={leave}: {e}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()



# =============================================================================
# Anti-circularity analyses: leave-one-variable-out and cross-domain characterization
# =============================================================================

CROSS_DOMAIN_TARGETS: Dict[str, List[str]] = {
    "functional_impairment_hf_burden": CLINICAL_PANELS["structural_remodeling"],
    "structural_remodeling": CLINICAL_PANELS["functional_impairment_hf_burden"],
    "valvular_regurgitation": CLINICAL_PANELS["functional_impairment_hf_burden"] + CLINICAL_PANELS["structural_remodeling"],
}


def _valid_rows_for_vars(clinical: pd.DataFrame, clinical_vars: List[str], args) -> np.ndarray:
    min_non = min(args.min_nonmissing_clinical_vars, len(clinical_vars))
    return (clinical[clinical_vars].notna().sum(axis=1).to_numpy() >= min_non)


def _axis_grouped_values(score_df: pd.DataFrame, clinical: pd.DataFrame, variable: str, args) -> pd.DataFrame:
    """Create patient-level values for one variable across acoustic-axis quantiles."""
    rows = []
    for comp in range(1, args.n_components + 1):
        score_col = f"cca_acoustic_axis{comp}"
        if score_col not in score_df.columns:
            continue
        scores = score_df[score_col].to_numpy(float)
        values = clinical[variable].to_numpy(float)
        valid = np.isfinite(scores) & np.isfinite(values)
        if valid.sum() < 20:
            continue
        rank = pd.Series(scores[valid]).rank(method="first")
        labels = [f"Q{i+1}" for i in range(args.n_axis_groups)]
        groups = pd.qcut(rank, q=args.n_axis_groups, labels=labels)
        for pid, score, group, value in zip(clinical.loc[valid, PATIENT_ID_COL], scores[valid], groups.to_numpy(), values[valid]):
            rows.append({
                "patient_id": pid,
                "component": comp,
                "axis_score": float(score),
                "axis_group": str(group),
                "variable": variable,
                "domain": VARIABLE_DOMAIN.get(variable, "Other"),
                "value": float(value),
                "is_ordinal": variable in ORDINAL_VARS,
                "is_valve": variable in VALVE_VARS,
            })
    return pd.DataFrame(rows)


def summarize_one_variable_gradient(patient_long: pd.DataFrame, panel_name: str, adjustment: str = "none") -> pd.DataFrame:
    """Summarize Q1-Q4 gradient for one held-out or cross-domain variable."""
    if len(patient_long) == 0:
        return pd.DataFrame()
    patient_long = patient_long.copy()
    patient_long["panel"] = panel_name
    patient_long["adjustment"] = adjustment
    return clinical_gradient_summary(patient_long)


def run_leave_one_variable_out_cca(
    X: np.ndarray,
    clinical: pd.DataFrame,
    patient_ids: Sequence[str],
    args,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Within each domain panel, leave one clinical variable out of CCA.

    The removed variable is never used in the CCA anchoring matrix. It is only
    evaluated on held-out fold acoustic scores. This directly addresses whether
    the learned axis relates to a variable because that variable was used to
    define the axis.
    """
    log("Leave-one-variable-out CCA anti-circularity analysis")
    summary_rows = []
    score_rows = []
    gpat_rows = []
    gsum_rows = []

    for panel, vars_all in CLINICAL_PANELS.items():
        if len(vars_all) < 3:
            log(f"  Skip leave-one-variable-out for {panel}: only {len(vars_all)} variables")
            continue
        for heldout in vars_all:
            train_vars = [v for v in vars_all if v != heldout]
            analysis_name = f"{panel}_leaveout_{heldout}"
            try:
                score, fold, Cp, pids = run_oof_cca(
                    X, clinical, patient_ids,
                    analysis_name, train_vars, args,
                    adjustment="none", covariate_cols=None, verbose=True,
                )
            except Exception as e:
                log(f"  LOO-CV failed: panel={panel}, heldout={heldout}: {e}")
                summary_rows.append({
                    "source_panel": panel,
                    "heldout_variable": heldout,
                    "training_variables": ";".join(train_vars),
                    "component": np.nan,
                    "status": f"failed: {e}",
                })
                continue

            score_out = score.copy()
            score_out["source_panel"] = panel
            score_out["heldout_variable"] = heldout
            score_out["training_variables"] = ";".join(train_vars)
            score_rows.append(score_out)

            for comp in range(1, args.n_components + 1):
                x = score[f"cca_acoustic_axis{comp}"]
                y = Cp[heldout]
                rho, pval, n = safe_spearman(x, y)
                lo, hi = bootstrap_ci_spearman(x, y, args.n_bootstrap, args.seed + 12000 + comp + 37 * len(summary_rows))
                summary_rows.append({
                    "source_panel": panel,
                    "heldout_variable": heldout,
                    "heldout_domain": VARIABLE_DOMAIN.get(heldout, "Other"),
                    "training_variables": ";".join(train_vars),
                    "analysis_panel": analysis_name,
                    "component": comp,
                    "n": n,
                    "status": "ok",
                    "spearman_acoustic_axis_vs_heldout": rho,
                    "spearman_p": pval,
                    "spearman_ci95_low": lo,
                    "spearman_ci95_high": hi,
                    "burden_orientation": BURDEN_DIRECTION.get(heldout, 0),
                    "oriented_spearman": rho * BURDEN_DIRECTION.get(heldout, 0) if np.isfinite(rho) else np.nan,
                })

            gpat = _axis_grouped_values(score, Cp, heldout, args)
            if len(gpat):
                gpat["source_panel"] = panel
                gpat["analysis_panel"] = analysis_name
                gpat["heldout_variable"] = heldout
                gpat["training_variables"] = ";".join(train_vars)
                gpat["panel"] = analysis_name
                gpat["adjustment"] = "none"
                gpat_rows.append(gpat)
                gsum = summarize_one_variable_gradient(gpat, analysis_name, adjustment="none")
                if len(gsum):
                    gsum["source_panel"] = panel
                    gsum["analysis_panel"] = analysis_name
                    gsum["heldout_variable"] = heldout
                    gsum["training_variables"] = ";".join(train_vars)
                    gsum_rows.append(gsum)

    summary = pd.DataFrame(summary_rows)
    if len(summary) and "spearman_p" in summary.columns:
        ok = summary["status"].eq("ok") & summary["spearman_p"].notna()
        summary.loc[ok, "spearman_fdr"] = fdr_bh(summary.loc[ok, "spearman_p"].values)
    score_all = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    gpat_all = pd.concat(gpat_rows, ignore_index=True) if gpat_rows else pd.DataFrame()
    gsum_all = pd.concat(gsum_rows, ignore_index=True) if gsum_rows else pd.DataFrame()
    return summary, score_all, gpat_all, gsum_all


def run_cross_domain_characterization(
    scores_all: pd.DataFrame,
    clinical: pd.DataFrame,
    args,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate non-anchoring clinical variables using domain-specific acoustic axes.

    For example, the functional axis is characterized against structural variables,
    and the structural axis is characterized against functional variables. These
    variables were not used to fit the source domain CCA.
    """
    log("Cross-domain clinical characterization")
    clinical_by_pid = clinical.set_index(PATIENT_ID_COL, drop=False)
    rows = []
    gpat_rows = []
    gsum_rows = []

    for source_panel, target_vars in CROSS_DOMAIN_TARGETS.items():
        sdf = scores_all[(scores_all["panel"] == source_panel) & (scores_all["adjustment"] == "none")].copy()
        if len(sdf) == 0:
            log(f"  Cross-domain skipped: no score rows for {source_panel}")
            continue
        sdf = sdf.dropna(subset=[PATIENT_ID_COL]).copy()
        C = clinical_by_pid.loc[sdf[PATIENT_ID_COL].values].reset_index(drop=True)
        sdf = sdf.reset_index(drop=True)

        for target in target_vars:
            for comp in range(1, args.n_components + 1):
                rho, pval, n = safe_spearman(sdf[f"cca_acoustic_axis{comp}"], C[target])
                lo, hi = bootstrap_ci_spearman(
                    sdf[f"cca_acoustic_axis{comp}"], C[target],
                    args.n_bootstrap, args.seed + 15000 + comp + 53 * len(rows)
                )
                rows.append({
                    "source_panel": source_panel,
                    "source_domain": PANEL_DOMAIN.get(source_panel, source_panel),
                    "target_variable": target,
                    "target_domain": VARIABLE_DOMAIN.get(target, "Other"),
                    "component": comp,
                    "n": n,
                    "spearman_acoustic_axis_vs_target": rho,
                    "spearman_p": pval,
                    "spearman_ci95_low": lo,
                    "spearman_ci95_high": hi,
                    "burden_orientation": BURDEN_DIRECTION.get(target, 0),
                    "oriented_spearman": rho * BURDEN_DIRECTION.get(target, 0) if np.isfinite(rho) else np.nan,
                    "target_was_used_in_source_cca": target in CLINICAL_PANELS.get(source_panel, []),
                })

            gpat = _axis_grouped_values(sdf, C, target, args)
            if len(gpat):
                analysis_panel = f"{source_panel}_to_{target}"
                gpat["source_panel"] = source_panel
                gpat["target_variable"] = target
                gpat["target_domain"] = VARIABLE_DOMAIN.get(target, "Other")
                gpat["panel"] = analysis_panel
                gpat["adjustment"] = "none"
                gpat_rows.append(gpat)
                gsum = summarize_one_variable_gradient(gpat, analysis_panel, adjustment="none")
                if len(gsum):
                    gsum["source_panel"] = source_panel
                    gsum["target_variable"] = target
                    gsum["target_domain"] = VARIABLE_DOMAIN.get(target, "Other")
                    gsum["target_was_used_in_source_cca"] = False
                    gsum_rows.append(gsum)

    out = pd.DataFrame(rows)
    if len(out) and "spearman_p" in out.columns:
        out["spearman_fdr"] = np.nan
        ok = out["spearman_p"].notna()
        out.loc[ok, "spearman_fdr"] = fdr_bh(out.loc[ok, "spearman_p"].values)
    gpat_all = pd.concat(gpat_rows, ignore_index=True) if gpat_rows else pd.DataFrame()
    gsum_all = pd.concat(gsum_rows, ignore_index=True) if gsum_rows else pd.DataFrame()
    return out, gpat_all, gsum_all



# =============================================================================
# Repeated random-split robustness analyses
# =============================================================================

def _args_with_seed(args, seed: int):
    """Clone argparse Namespace and replace seed for repeated split analyses."""
    new_args = copy.copy(args)
    new_args.seed = int(seed)
    return new_args


def _summarize_numeric_values(values: np.ndarray, prefix: str = "") -> Dict[str, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    out = {
        f"{prefix}n_valid": int(len(vals)),
        f"{prefix}mean": float(np.mean(vals)) if len(vals) else np.nan,
        f"{prefix}sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
        f"{prefix}median": float(np.median(vals)) if len(vals) else np.nan,
        f"{prefix}p2_5": float(np.percentile(vals, 2.5)) if len(vals) else np.nan,
        f"{prefix}p25": float(np.percentile(vals, 25)) if len(vals) else np.nan,
        f"{prefix}p75": float(np.percentile(vals, 75)) if len(vals) else np.nan,
        f"{prefix}p97_5": float(np.percentile(vals, 97.5)) if len(vals) else np.nan,
        f"{prefix}min": float(np.min(vals)) if len(vals) else np.nan,
        f"{prefix}max": float(np.max(vals)) if len(vals) else np.nan,
    }
    return out


def run_repeated_cca_alignment(
    X: np.ndarray,
    clinical: pd.DataFrame,
    patient_ids: Sequence[str],
    panels: Dict[str, List[str]],
    args,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat the full 5-fold OOF CCA with different random seeds.

    This is a random-split robustness analysis. Each repeat refits all fold-level
    preprocessing and CCA models from scratch using a different KFold split.
    """
    n_repeats = int(args.n_random_split_repeats)
    rows = []
    if n_repeats <= 0:
        return pd.DataFrame(), pd.DataFrame()

    log(f"Repeated random-split CCA robustness: repeats={n_repeats}, folds={args.n_splits}")
    for rep in range(n_repeats):
        split_seed = int(args.seed + 20000 + rep)
        local_args = _args_with_seed(args, split_seed)
        log(f"  repeated CCA split {rep + 1}/{n_repeats}, seed={split_seed}")
        for panel, vars_ in panels.items():
            try:
                score, _, Cp, _ = run_oof_cca(
                    X, clinical, patient_ids,
                    panel_name=panel,
                    clinical_vars=vars_,
                    args=local_args,
                    adjustment="none",
                    covariate_cols=None,
                    verbose=False,
                )
                for comp in range(1, args.n_components + 1):
                    x = score[f"cca_acoustic_axis{comp}"]
                    y = score[f"cca_clinical_axis{comp}"]
                    rho, pval, n = safe_spearman(x, y)
                    pr, pp, _ = safe_pearson(x, y)
                    rows.append({
                        "repeat": rep + 1,
                        "split_seed": split_seed,
                        "panel": panel,
                        "domain": PANEL_DOMAIN.get(panel, "All domains"),
                        "component": comp,
                        "n": n,
                        "status": "ok",
                        "spearman_acoustic_vs_clinical_axis": rho,
                        "spearman_p": pval,
                        "pearson_acoustic_vs_clinical_axis": pr,
                        "pearson_p": pp,
                    })
            except Exception as e:
                log(f"    repeated CCA failed: panel={panel}, seed={split_seed}: {e}")
                rows.append({
                    "repeat": rep + 1,
                    "split_seed": split_seed,
                    "panel": panel,
                    "domain": PANEL_DOMAIN.get(panel, "All domains"),
                    "component": np.nan,
                    "n": 0,
                    "status": f"failed: {e}",
                    "spearman_acoustic_vs_clinical_axis": np.nan,
                    "spearman_p": np.nan,
                    "pearson_acoustic_vs_clinical_axis": np.nan,
                    "pearson_p": np.nan,
                })

    values = pd.DataFrame(rows)
    summary_rows = []
    if len(values):
        ok = values["status"].eq("ok")
        for (panel, comp), sub in values[ok].groupby(["panel", "component"]):
            vals = sub["spearman_acoustic_vs_clinical_axis"].to_numpy(float)
            rec = {
                "panel": panel,
                "domain": PANEL_DOMAIN.get(panel, "All domains"),
                "component": comp,
                "n_repeats_requested": n_repeats,
            }
            rec.update(_summarize_numeric_values(vals, prefix="spearman_"))
            rec["positive_rate"] = float(np.mean(vals > 0)) if len(vals) else np.nan
            rec["all_repeats_positive"] = bool(np.all(vals > 0)) if len(vals) else False
            summary_rows.append(rec)
    return values, pd.DataFrame(summary_rows)


def run_repeated_endpoint_validation(
    X: np.ndarray,
    clinical: pd.DataFrame,
    patient_ids: Sequence[str],
    args,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat leave-endpoint-out endpoint validation with different stratified splits.

    Each repeat performs a fresh StratifiedKFold split. Within every fold, the
    endpoint-defining clinical variable is removed from the CCA Y matrix, CCA is
    fitted on the training fold only, and two simple logistic classifiers are
    trained/evaluated: one using acoustic CCA axis 1 only and one using axes 1+2.
    """
    n_repeats = int(args.n_random_split_repeats)
    endpoints = ["EF_lt_40", "NTproBNP_ge_900", "NYHA_ge_3", "LA_ge_40", "LVEDD_dilated"]
    axis_feature_options = [1, 2]
    rows = []
    if n_repeats <= 0:
        return pd.DataFrame(), pd.DataFrame()

    log(f"Repeated StratifiedKFold endpoint robustness: repeats={n_repeats}, folds={args.n_splits}")
    for endpoint in endpoints:
        y_series, source, rule = make_endpoint(clinical, endpoint)
        valid = y_series.notna()
        y_all = y_series.loc[valid].astype(int).to_numpy()
        Xv = X[valid.to_numpy()]
        Cv = clinical.loc[valid].reset_index(drop=True)
        n = len(y_all)
        n_pos = int(y_all.sum()) if n else 0
        n_neg = int(n - n_pos)
        min_class = min(n_pos, n_neg) if n else 0

        if n < args.min_endpoint_n or len(np.unique(y_all)) < 2 or min_class < args.min_endpoint_class_n:
            for k_requested in axis_feature_options:
                rows.append({
                    "endpoint": endpoint,
                    "source_column": source,
                    "rule": rule,
                    "repeat": np.nan,
                    "split_seed": np.nan,
                    "n_axis_features": k_requested,
                    "axis_feature_set": "axis1" if k_requested == 1 else "axis1_2",
                    "n": n,
                    "n_positive": n_pos,
                    "positive_rate": float(np.mean(y_all)) if n else np.nan,
                    "status": "skipped_too_few_for_repeated_auc",
                    "auroc": np.nan,
                    "balanced_accuracy": np.nan,
                    "accuracy": np.nan,
                    "sensitivity": np.nan,
                    "specificity": np.nan,
                })
            continue

        panel_vars = exclude_endpoint_var(source, ALL_CLINICAL_VARS)
        n_splits = min(args.n_splits, n_pos, n_neg, n)
        if n_splits < 2:
            for k_requested in axis_feature_options:
                rows.append({
                    "endpoint": endpoint,
                    "source_column": source,
                    "rule": rule,
                    "repeat": np.nan,
                    "split_seed": np.nan,
                    "n_axis_features": k_requested,
                    "axis_feature_set": "axis1" if k_requested == 1 else "axis1_2",
                    "n": n,
                    "n_positive": n_pos,
                    "positive_rate": float(np.mean(y_all)),
                    "status": "skipped_n_splits_lt_2",
                    "auroc": np.nan,
                    "balanced_accuracy": np.nan,
                    "accuracy": np.nan,
                    "sensitivity": np.nan,
                    "specificity": np.nan,
                })
            continue

        log(f"  repeated endpoint: {endpoint}, n={n}, pos={n_pos}, folds={n_splits}")
        for rep in range(n_repeats):
            split_seed = int(args.seed + 30000 + 97 * rep)
            local_args = _args_with_seed(args, split_seed)
            prob_by_k = {k: np.full(n, np.nan) for k in axis_feature_options}
            try:
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
                for fold, (tr, te) in enumerate(skf.split(Xv, y_all), start=1):
                    # Fold-specific CCA is fit on training patients only. Both
                    # classifiers use the same leakage-safe fold projection.
                    res = fit_cca_fold(Xv[tr], Xv[te], Cv.iloc[tr], Cv.iloc[te], panel_vars, local_args)
                    max_k = int(res["xtr"].shape[1])
                    for k_requested in axis_feature_options:
                        k = min(k_requested, max_k)
                        if k < k_requested:
                            continue
                        clf = Pipeline([
                            ("scaler", StandardScaler()),
                            ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
                        ])
                        clf.fit(res["xtr"][:, :k], y_all[tr])
                        prob_by_k[k_requested][te] = clf.predict_proba(res["xte"][:, :k])[:, 1]

                for k_requested, prob in prob_by_k.items():
                    m = endpoint_metrics(y_all, prob)
                    rows.append({
                        "endpoint": endpoint,
                        "source_column": source,
                        "rule": rule,
                        "repeat": rep + 1,
                        "split_seed": split_seed,
                        "panel": "all_clinical",
                        "panel_type": "all_clinical_leave_endpoint_out",
                        "n_axis_features": k_requested,
                        "axis_feature_set": "axis1" if k_requested == 1 else "axis1_2",
                        "clinical_panel_vars": ";".join(panel_vars),
                        "status": "ok",
                        **m,
                    })
            except Exception as e:
                log(f"    repeated endpoint failed: endpoint={endpoint}, seed={split_seed}: {e}")
                for k_requested in axis_feature_options:
                    rows.append({
                        "endpoint": endpoint,
                        "source_column": source,
                        "rule": rule,
                        "repeat": rep + 1,
                        "split_seed": split_seed,
                        "panel": "all_clinical",
                        "panel_type": "all_clinical_leave_endpoint_out",
                        "n_axis_features": k_requested,
                        "axis_feature_set": "axis1" if k_requested == 1 else "axis1_2",
                        "clinical_panel_vars": ";".join(panel_vars),
                        "n": n,
                        "n_positive": n_pos,
                        "positive_rate": float(np.mean(y_all)),
                        "status": f"failed: {e}",
                        "auroc": np.nan,
                        "balanced_accuracy": np.nan,
                        "accuracy": np.nan,
                        "sensitivity": np.nan,
                        "specificity": np.nan,
                    })

    values = pd.DataFrame(rows)
    summary_rows = []
    if len(values):
        ok = values["status"].eq("ok")
        for (endpoint, k_requested), sub in values[ok].groupby(["endpoint", "n_axis_features"]):
            aucs = sub["auroc"].to_numpy(float)
            bas = sub["balanced_accuracy"].to_numpy(float)
            rec = {
                "endpoint": endpoint,
                "source_column": sub["source_column"].iloc[0],
                "rule": sub["rule"].iloc[0],
                "n_axis_features": int(k_requested),
                "axis_feature_set": "axis1" if int(k_requested) == 1 else "axis1_2",
                "n": int(sub["n"].iloc[0]),
                "n_positive": int(sub["n_positive"].iloc[0]),
                "positive_rate": float(sub["positive_rate"].iloc[0]),
                "n_repeats_requested": n_repeats,
            }
            rec.update(_summarize_numeric_values(aucs, prefix="auroc_"))
            rec.update(_summarize_numeric_values(bas, prefix="balanced_accuracy_"))
            rec["auroc_gt_0_5_rate"] = float(np.mean(aucs > 0.5)) if len(aucs) else np.nan
            summary_rows.append(rec)
    return values, pd.DataFrame(summary_rows)


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
    """Plot repeated StratifiedKFold endpoint AUROC distributions for axis1 vs axes1+2."""
    if len(values_df) == 0:
        return
    d = values_df[values_df["status"].eq("ok")].copy()
    d = d[np.isfinite(d["auroc"])]
    if len(d) == 0:
        return
    if "n_axis_features" not in d.columns:
        d["n_axis_features"] = 2
    d["axis_feature_set"] = d["n_axis_features"].astype(int).map({1: "Axis 1", 2: "Axis 1+2"}).fillna(d.get("axis_feature_set", "Axis"))
    endpoint_order = ["EF_lt_40", "NTproBNP_ge_900", "NYHA_ge_3", "LA_ge_40", "LVEDD_dilated"]
    endpoint_order = [e for e in endpoint_order if e in set(d["endpoint"])]
    hue_order = [h for h in ["Axis 1", "Axis 1+2"] if h in set(d["axis_feature_set"])]
    plt.figure(figsize=(11.2, 6.0))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.boxplot(
            data=d, x="endpoint", y="auroc", hue="axis_feature_set",
            order=endpoint_order, hue_order=hue_order, showfliers=False, linewidth=1.25, ax=ax,
        )
        sns.stripplot(
            data=d, x="endpoint", y="auroc", hue="axis_feature_set",
            order=endpoint_order, hue_order=hue_order, dodge=True,
            color="black", size=3.3, alpha=0.50, jitter=0.16, ax=ax,
        )
        # Deduplicate legends caused by boxplot + stripplot.
        handles, labels = ax.get_legend_handles_labels()
        seen = set(); handles2=[]; labels2=[]
        for h,l in zip(handles, labels):
            if l not in seen and l in hue_order:
                handles2.append(h); labels2.append(l); seen.add(l)
        ax.legend(handles2, labels2, title="CCA features", frameon=False, loc="lower right", fontsize=10)
    else:
        width = 0.35
        xbase = np.arange(len(endpoint_order))
        for j, feat in enumerate(hue_order):
            groups = [d.loc[(d["endpoint"] == ep) & (d["axis_feature_set"] == feat), "auroc"].dropna().to_numpy(float) for ep in endpoint_order]
            pos = xbase + (j - (len(hue_order)-1)/2) * width
            ax.boxplot(groups, positions=pos, widths=width*0.85, showfliers=False)
        ax.set_xticks(xbase); ax.set_xticklabels(endpoint_order)
    ax.axhline(0.5, color="black", linestyle="--", lw=1)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("AUROC")
    ax.set_title("Repeated StratifiedKFold endpoint robustness: axis 1 vs axis 1+2")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    savefig(out_path)


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
    """Plot ROC curves, overlaying axis 1 and axes 1+2 for each endpoint."""
    if len(pred_df) == 0:
        return
    d = pred_df.copy()
    if "n_axis_features" not in d.columns:
        d["n_axis_features"] = 2
    d["axis_feature_set"] = d["n_axis_features"].astype(int).map({1: "Axis 1", 2: "Axis 1+2"}).fillna(d.get("axis_feature_set", "Axis"))
    summary = summary_df.copy() if summary_df is not None else pd.DataFrame()
    if len(summary) and "n_axis_features" not in summary.columns:
        summary["n_axis_features"] = 2
    for endpoint, sub_ep in d.groupby("endpoint"):
        if len(sub_ep) == 0 or sub_ep["y_true"].nunique() < 2:
            continue
        plt.figure(figsize=(6.8, 6.2))
        ax = plt.gca()
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", lw=1)
        for k, label in [(1, "Axis 1"), (2, "Axis 1+2")]:
            sub = sub_ep[sub_ep["n_axis_features"].astype(int) == k]
            if len(sub) == 0 or sub["y_true"].nunique() < 2:
                continue
            y = sub["y_true"].astype(int).to_numpy()
            p = sub["y_prob"].astype(float).to_numpy()
            fpr, tpr, lo, hi = roc_curve_with_ci(y, p, args.n_bootstrap, args.seed + 9000 + 19 * k)
            auc = roc_auc_score(y, p)
            row = summary[(summary["endpoint"] == endpoint) & (summary["n_axis_features"].astype(int) == k)] if len(summary) else pd.DataFrame()
            if len(row):
                auc_lo = row["auroc_ci95_low"].iloc[0]
                auc_hi = row["auroc_ci95_high"].iloc[0]
            else:
                auc_lo, auc_hi = np.nan, np.nan
            if np.isfinite(lo).any():
                ax.fill_between(fpr, lo, hi, alpha=0.12)
            lab = f"{label}: AUROC={auc:.3f}" + (f" ({auc_lo:.3f}–{auc_hi:.3f})" if np.isfinite(auc_lo) else "")
            ax.plot(fpr, tpr, lw=2.3, label=lab)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC curve: {endpoint}\naxis 1 vs axis 1+2")
        ax.legend(frameon=False, loc="lower right", fontsize=10)
        savefig(fig_dir / f"roc_curve_{clean_filename(endpoint)}_axis1_vs_axis12.png")


def plot_endpoint_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Endpoint AUROC forest comparing acoustic CCA axis 1 vs axes 1+2."""
    d = summary_df[np.isfinite(summary_df.get("auroc", np.nan))].copy()
    if len(d) == 0:
        return
    if "n_axis_features" not in d.columns:
        d["n_axis_features"] = 2
    d["axis_feature_set"] = d["n_axis_features"].astype(int).map({1: "Axis 1", 2: "Axis 1+2"}).fillna(d.get("axis_feature_set", "Axis"))
    endpoint_order = ["EF_lt_40", "NTproBNP_ge_900", "NYHA_ge_3", "LA_ge_40", "LVEDD_dilated"]
    endpoint_order = [e for e in endpoint_order if e in set(d["endpoint"])]
    feat_order = [h for h in ["Axis 1", "Axis 1+2"] if h in set(d["axis_feature_set"])]

    plt.figure(figsize=(9.8, max(4.8, 0.72 * len(endpoint_order))))
    ax = plt.gca()
    y_base = np.arange(len(endpoint_order))
    offsets = {"Axis 1": -0.13, "Axis 1+2": 0.13}
    colors = {"Axis 1": "0.35", "Axis 1+2": "black"}
    for feat in feat_order:
        sub = d[d["axis_feature_set"] == feat].copy()
        xs=[]; ys=[]; xerr_low=[]; xerr_high=[]
        for i, ep in enumerate(endpoint_order):
            hit = sub[sub["endpoint"] == ep]
            if len(hit) == 0:
                continue
            row = hit.iloc[0]
            x = float(row["auroc"])
            lo = float(row["auroc_ci95_low"]) if "auroc_ci95_low" in row and pd.notna(row["auroc_ci95_low"]) else np.nan
            hi = float(row["auroc_ci95_high"]) if "auroc_ci95_high" in row and pd.notna(row["auroc_ci95_high"]) else np.nan
            xs.append(x); ys.append(i + offsets.get(feat, 0));
            xerr_low.append(x - lo if np.isfinite(lo) else 0)
            xerr_high.append(hi - x if np.isfinite(hi) else 0)
        if xs:
            ax.errorbar(xs, ys, xerr=[xerr_low, xerr_high], fmt="o", capsize=3, color=colors.get(feat, None), label=feat)
    ax.axvline(0.5, color="black", linestyle="--", lw=1)
    ax.set_yticks(y_base)
    ax.set_yticklabels(endpoint_order)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("AUROC")
    ax.set_title("Endpoint validation: axis 1 vs axis 1+2")
    ax.legend(frameon=False, loc="lower right")
    savefig(out_path)


def plot_confounder(summary_df: pd.DataFrame, unadjusted: pd.DataFrame, out_path: Path) -> None:
    rows = []
    base = unadjusted[(unadjusted["panel"] == "all_clinical") & (unadjusted["adjustment"] == "none") & (unadjusted["component"] == 1)]
    if len(base):
        row = base.iloc[0].to_dict()
        row["plot_label"] = "Unadjusted"
        rows.append(row)
    for label, nice in [
        ("age_residualized", "Age"),
        ("sex_residualized", "Sex"),
        ("heart_rate_residualized", "Heart rate"),
    ]:
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
    ax.set_title("Single-covariate-adjusted CCA alignment")
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
# Added robustness: position contribution, hyperparameter sensitivity, and threshold endpoints
# =============================================================================

def parse_int_list(s: str) -> List[int]:
    vals = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return vals


def build_position_mode_X(
    Xpos: np.ndarray,
    meta: pd.DataFrame,
    patient_ids: Sequence[str],
    mode: str,
    positions_to_use: Sequence[str],
):
    """Build patient-level X from position embeddings for contribution analysis.

    mode="concat" concatenates all selected positions in fixed order.
    mode="single" uses one selected position only.
    """
    rows, pids = [], []
    for pid in patient_ids:
        parts = []
        ok = True
        for pos in positions_to_use:
            sub = meta[(meta[PATIENT_ID_COL] == pid) & (meta["position"] == pos)]
            if len(sub) == 0:
                ok = False
                break
            parts.append(Xpos[sub["row"].to_numpy(dtype=int)].mean(axis=0))
        if ok:
            if mode == "single":
                rows.append(parts[0])
            else:
                rows.append(np.concatenate(parts))
            pids.append(pid)
    if not rows:
        return np.empty((0, 0), dtype=np.float32), []
    return np.vstack(rows).astype(np.float32), pids


def run_position_contribution_analysis(
    clinical: pd.DataFrame,
    patient_ids: Sequence[str],
    args,
) -> pd.DataFrame:
    """Only-position contribution analysis.

    This complements leave-one-position-out. LOPO asks whether alignment remains
    after removing one site. Only-position asks how much all-clinical alignment can
    be recovered from each site alone. A full position-concat baseline is also
    included so that only-A/E/M/P/T are compared to a representation built from the
    same position-level embeddings.
    """
    if not args.run_position_contribution:
        return pd.DataFrame()
    Xpos, meta = load_position_embeddings(args)
    if Xpos is None:
        return pd.DataFrame()

    clinical_by_pid = clinical.set_index(PATIENT_ID_COL, drop=False)
    positions = [p.strip().upper() for p in args.positions.split(",") if p.strip()]
    configs = [("full_position_concat", positions, "concat")]
    configs += [(f"only_{p}", [p], "single") for p in positions]

    rows = []
    for name, pos_list, mode in configs:
        Xc, pids = build_position_mode_X(Xpos, meta, patient_ids, mode=mode, positions_to_use=pos_list)
        if len(pids) < args.min_panel_n:
            log(f"Position contribution skipped: {name}, n={len(pids)}")
            continue
        Cl = clinical_by_pid.loc[pids].reset_index(drop=True)
        try:
            sdf, _, Cp, _ = run_oof_cca(
                Xc, Cl, pids, f"all_clinical_{name}", ALL_CLINICAL_VARS, args, verbose=True
            )
            summ = summarize_alignment(sdf, Cp, ALL_CLINICAL_VARS, args, seed_offset=8100)
            summ["position_analysis"] = name
            summ["included_positions"] = ",".join(pos_list)
            summ["n_positions_used"] = len(pos_list)
            summ["position_mode"] = mode
            rows.append(summ)
        except Exception as e:
            log(f"Position contribution failed: {name}: {e}")
            rows.append(pd.DataFrame([{
                "panel": f"all_clinical_{name}",
                "adjustment": "none",
                "component": np.nan,
                "n": len(pids),
                "spearman_acoustic_vs_clinical_axis": np.nan,
                "spearman_p": np.nan,
                "spearman_ci95_low": np.nan,
                "spearman_ci95_high": np.nan,
                "position_analysis": name,
                "included_positions": ",".join(pos_list),
                "n_positions_used": len(pos_list),
                "position_mode": mode,
                "status": f"failed: {e}",
            }]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_position_contribution(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Plot full-position-concat vs only-position alignment."""
    if len(summary_df) == 0:
        return
    d = summary_df[summary_df["component"] == 1].copy()
    d = d[np.isfinite(d["spearman_acoustic_vs_clinical_axis"])]
    if len(d) == 0:
        return
    order = ["full_position_concat", "only_A", "only_E", "only_M", "only_P", "only_T"]
    d["plot_order"] = d["position_analysis"].map({v: i for i, v in enumerate(order)})
    d = d.sort_values("plot_order")
    labels = {
        "full_position_concat": "Full A/E/M/P/T\n(position concat)",
        "only_A": "Only A",
        "only_E": "Only E",
        "only_M": "Only M",
        "only_P": "Only P",
        "only_T": "Only T",
    }
    plt.figure(figsize=(8.8, 5.3))
    ax = plt.gca()
    y = np.arange(len(d))
    x = d["spearman_acoustic_vs_clinical_axis"].to_numpy(float)
    lo = d["spearman_ci95_low"].to_numpy(float)
    hi = d["spearman_ci95_high"].to_numpy(float)
    colors = ["black" if a == "full_position_concat" else "#0072B2" for a in d["position_analysis"]]
    for yi, xi, li, hi_i, color in zip(y, x, lo, hi, colors):
        ax.plot([li, hi_i], [yi, yi], color=color, lw=2.2)
        ax.scatter([xi], [yi], color=color, edgecolor="black", s=75, zorder=3)
    ax.axvline(0, color="black", lw=1)
    full = d[d["position_analysis"] == "full_position_concat"]
    if len(full):
        ax.axvline(full["spearman_acoustic_vs_clinical_axis"].iloc[0], color="black", linestyle="--", lw=1, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([labels.get(a, a) for a in d["position_analysis"]])
    ax.set_xlabel("OOF Spearman correlation\n(all-clinical acoustic vs. clinical axis)")
    ax.set_title("Position contribution: single-site vs full position-concat alignment")
    savefig(out_path)


def run_hyperparameter_sensitivity(
    X: np.ndarray,
    clinical: pd.DataFrame,
    patient_ids: Sequence[str],
    args,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sensitivity to CCA hyperparameters.

    Tests n_pca = 20/50/100 and n_components = 1/2. For each setting, all
    fold-level preprocessing and CCA fitting are repeated inside the same
    leakage-safe OOF framework. Endpoint sensitivity uses the same leave-endpoint-
    out logic as the main endpoint validation.
    """
    if args.skip_hyperparameter_sensitivity:
        return pd.DataFrame(), pd.DataFrame()

    n_pca_values = parse_int_list(args.sensitivity_n_pca_values)
    n_comp_values = parse_int_list(args.sensitivity_n_components_values)
    align_rows = []
    endpoint_rows = []

    for n_pca in n_pca_values:
        for n_comp in n_comp_values:
            local_args = copy.copy(args)
            local_args.n_pca = int(n_pca)
            local_args.n_components = int(n_comp)
            tag = f"pca{n_pca}_comp{n_comp}"
            log(f"Hyperparameter sensitivity: {tag}")

            try:
                score, _, Cp, _ = run_oof_cca(
                    X, clinical, patient_ids,
                    panel_name=f"all_clinical_{tag}",
                    clinical_vars=ALL_CLINICAL_VARS,
                    args=local_args,
                    adjustment="none",
                    covariate_cols=None,
                    verbose=False,
                )
                summ = summarize_alignment(score, Cp, ALL_CLINICAL_VARS, local_args, seed_offset=9000 + n_pca * 10 + n_comp)
                summ["sensitivity_type"] = "hyperparameter"
                summ["n_pca_setting"] = n_pca
                summ["n_components_setting"] = n_comp
                summ["setting"] = tag
                align_rows.append(summ)
            except Exception as e:
                log(f"  hyperparameter alignment failed: {tag}: {e}")
                align_rows.append(pd.DataFrame([{
                    "panel": f"all_clinical_{tag}",
                    "adjustment": "none",
                    "component": np.nan,
                    "n": np.nan,
                    "spearman_acoustic_vs_clinical_axis": np.nan,
                    "spearman_p": np.nan,
                    "spearman_ci95_low": np.nan,
                    "spearman_ci95_high": np.nan,
                    "sensitivity_type": "hyperparameter",
                    "n_pca_setting": n_pca,
                    "n_components_setting": n_comp,
                    "setting": tag,
                    "status": f"failed: {e}",
                }]))

            try:
                ep_summary, _, _ = run_endpoint_validation(X, clinical, patient_ids, local_args)
                if len(ep_summary):
                    ep_summary["sensitivity_type"] = "hyperparameter"
                    ep_summary["n_pca_setting"] = n_pca
                    ep_summary["n_components_setting"] = n_comp
                    ep_summary["setting"] = tag
                    endpoint_rows.append(ep_summary)
            except Exception as e:
                log(f"  hyperparameter endpoint failed: {tag}: {e}")

    align = pd.concat(align_rows, ignore_index=True) if align_rows else pd.DataFrame()
    endpoint = pd.concat(endpoint_rows, ignore_index=True) if endpoint_rows else pd.DataFrame()
    return align, endpoint


def plot_hyperparameter_alignment(align_df: pd.DataFrame, out_path: Path) -> None:
    if len(align_df) == 0:
        return
    d = align_df[(align_df["component"] == 1)].copy()
    d = d[np.isfinite(d["spearman_acoustic_vs_clinical_axis"])]
    if len(d) == 0:
        return
    plt.figure(figsize=(7.8, 5.2))
    ax = plt.gca()
    if HAS_SEABORN:
        sns.lineplot(
            data=d,
            x="n_pca_setting",
            y="spearman_acoustic_vs_clinical_axis",
            hue="n_components_setting",
            marker="o",
            linewidth=2.2,
            ax=ax,
        )
    else:
        for comp, sub in d.groupby("n_components_setting"):
            ax.plot(sub["n_pca_setting"], sub["spearman_acoustic_vs_clinical_axis"], marker="o", label=f"n_components={comp}")
        ax.legend()
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("PCA components before CCA")
    ax.set_ylabel("OOF Spearman correlation\n(acoustic axis 1 vs clinical axis 1)")
    ax.set_title("Hyperparameter sensitivity of all-clinical CCA alignment")
    savefig(out_path)


def plot_hyperparameter_endpoint(endpoint_df: pd.DataFrame, out_path: Path) -> None:
    if len(endpoint_df) == 0:
        return
    d = endpoint_df[np.isfinite(endpoint_df.get("auroc", np.nan))].copy()
    if len(d) == 0:
        return
    d["axis_feature_set"] = d["n_axis_features"].astype(int).map({1: "Axis 1", 2: "Axis 1+2"})
    d["setting_label"] = "PCA" + d["n_pca_setting"].astype(str) + "/C" + d["n_components_setting"].astype(str)
    # Keep default endpoints, summarized as a compact heatmap by endpoint and setting.
    for axis_label in ["Axis 1", "Axis 1+2"]:
        sub = d[d["axis_feature_set"] == axis_label].copy()
        if len(sub) == 0:
            continue
        pivot = sub.pivot_table(index="endpoint", columns="setting_label", values="auroc", aggfunc="mean")
        if pivot.empty:
            continue
        plt.figure(figsize=(max(8.5, 0.75 * len(pivot.columns)), max(4.5, 0.45 * len(pivot.index))))
        ax = plt.gca()
        if HAS_SEABORN:
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.5, vmax=0.9, linewidths=0.4, ax=ax)
        else:
            im = ax.imshow(pivot.values, aspect="auto", vmin=0.5, vmax=0.9)
            plt.colorbar(im, ax=ax, label="AUROC")
            ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
            ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
        ax.set_title(f"Endpoint AUROC hyperparameter sensitivity ({axis_label})")
        ax.set_xlabel("Hyperparameter setting")
        ax.set_ylabel("")
        savefig(out_path.with_name(out_path.stem + f"_{clean_filename(axis_label)}.png"))


def make_threshold_endpoint(clinical: pd.DataFrame, family: str, threshold: float) -> Tuple[pd.Series, str, str, str]:
    y = pd.Series(np.nan, index=clinical.index, dtype=float)
    if family == "EF_lt":
        v = clinical["EF_Teich"]
        y.loc[v.notna()] = (v.loc[v.notna()] < threshold).astype(float)
        name = f"EF_lt_{int(threshold)}"
        return y, "EF_Teich", f"EF_Teich < {threshold:g}", name
    if family == "NTproBNP_ge":
        v = clinical["NTproBNP"]
        thr_used = np.log1p(threshold) if np.nanmax(v.values) <= 20 else threshold
        y.loc[v.notna()] = (v.loc[v.notna()] >= thr_used).astype(float)
        name = f"NTproBNP_ge_{int(threshold)}"
        rule = f"NTproBNP >= {thr_used:.4g} (NT-proBNP >= {threshold:g} if log1p transformed)"
        return y, "NTproBNP", rule, name
    if family == "NYHA_ge":
        v = clinical["NYHA"]
        y.loc[v.notna()] = (v.loc[v.notna()] >= threshold).astype(float)
        name = f"NYHA_ge_{int(threshold)}"
        return y, "NYHA", f"NYHA >= {threshold:g}", name
    raise ValueError(family)


def run_endpoint_threshold_sensitivity(
    X: np.ndarray,
    clinical: pd.DataFrame,
    patient_ids: Sequence[str],
    args,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Endpoint-threshold sensitivity with leakage-safe leave-endpoint-out CCA.

    For each endpoint threshold, the source clinical variable is removed from the
    CCA Y matrix inside every training fold. CCA and the logistic classifier are
    fit only on the training fold; held-out patients are only transformed and
    predicted.
    """
    if args.skip_endpoint_threshold_sensitivity:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    specs = [
        ("EF_lt", 40.0), ("EF_lt", 50.0),
        ("NTproBNP_ge", 125.0), ("NTproBNP_ge", 300.0), ("NTproBNP_ge", 900.0),
        ("NYHA_ge", 1.0), ("NYHA_ge", 2.0), ("NYHA_ge", 3.0),
    ]
    axis_options = [1, 2]
    summary, pred_rows, info_rows = [], [], []

    for family, thr in specs:
        y_series, source, rule, endpoint = make_threshold_endpoint(clinical, family, thr)
        valid = y_series.notna()
        y_all = y_series.loc[valid].astype(int).to_numpy()
        info = {
            "endpoint": endpoint,
            "endpoint_family": family,
            "threshold": thr,
            "source_column": source,
            "rule": rule,
            "n": int(valid.sum()),
            "n_positive": int(y_all.sum()) if len(y_all) else 0,
            "positive_rate": float(y_all.mean()) if len(y_all) else np.nan,
        }
        min_class = int(min(y_all.sum(), len(y_all) - y_all.sum())) if len(y_all) else 0
        if len(y_all) < args.min_endpoint_n or len(np.unique(y_all)) < 2 or min_class < args.min_endpoint_class_n:
            info["status"] = "skipped_too_few_for_oof_auc"
            info["min_class_count"] = min_class
            info_rows.append(info)
            log(f"Threshold endpoint skipped: {endpoint}, n={len(y_all)}, pos={int(y_all.sum()) if len(y_all) else 0}")
            continue
        info["status"] = "ok"
        info["min_class_count"] = min_class
        info_rows.append(info)

        Xv = X[valid.to_numpy()]
        Cv = clinical.loc[valid].reset_index(drop=True)
        pids = [pid for pid, keep in zip(patient_ids, valid.to_numpy()) if keep]
        panel_vars = exclude_endpoint_var(source, ALL_CLINICAL_VARS)
        n_splits = min(args.n_splits, int(y_all.sum()), int(len(y_all) - y_all.sum()), len(y_all))
        prob_by_k = {k: np.full(len(y_all), np.nan) for k in axis_options}
        folds = np.full(len(y_all), -1, dtype=int)

        log(f"Threshold endpoint validation: {endpoint}, n={len(y_all)}, pos={int(y_all.sum())}")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        for fold, (tr, te) in enumerate(skf.split(Xv, y_all), start=1):
            res = fit_cca_fold(Xv[tr], Xv[te], Cv.iloc[tr], Cv.iloc[te], panel_vars, args)
            max_k = int(res["xtr"].shape[1])
            for k_requested in axis_options:
                k = min(k_requested, max_k)
                if k < k_requested:
                    continue
                clf = Pipeline([
                    ("scaler", StandardScaler()),
                    ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
                ])
                clf.fit(res["xtr"][:, :k], y_all[tr])
                prob_by_k[k_requested][te] = clf.predict_proba(res["xte"][:, :k])[:, 1]
            folds[te] = fold

        for k_requested, prob in prob_by_k.items():
            if not np.isfinite(prob).any():
                continue
            m = endpoint_metrics(y_all, prob)
            lo, hi = bootstrap_auc_ci(y_all, prob, args.n_bootstrap, args.seed + 7000 + k_requested + int(thr))
            summary.append({
                "endpoint": endpoint,
                "endpoint_family": family,
                "threshold": thr,
                "source_column": source,
                "rule": rule,
                "panel": "all_clinical",
                "panel_type": "all_clinical_leave_endpoint_out_threshold_sensitivity",
                "n_axis_features": k_requested,
                "axis_feature_set": "axis1" if k_requested == 1 else "axis1_2",
                "clinical_panel_vars": ";".join(panel_vars),
                **m,
                "auroc_ci95_low": lo,
                "auroc_ci95_high": hi,
            })
            for pid, yt, yp, f in zip(pids, y_all, prob, folds):
                pred_rows.append({
                    PATIENT_ID_COL: pid,
                    "endpoint": endpoint,
                    "endpoint_family": family,
                    "threshold": thr,
                    "source_column": source,
                    "y_true": int(yt),
                    "y_prob": float(yp),
                    "fold": int(f),
                    "n_axis_features": k_requested,
                    "axis_feature_set": "axis1" if k_requested == 1 else "axis1_2",
                })

    return pd.DataFrame(summary), pd.DataFrame(pred_rows), pd.DataFrame(info_rows)


def plot_endpoint_threshold_sensitivity(summary_df: pd.DataFrame, out_path: Path) -> None:
    if len(summary_df) == 0:
        return
    d = summary_df[np.isfinite(summary_df.get("auroc", np.nan))].copy()
    if len(d) == 0:
        return
    d["axis_feature_set"] = d["n_axis_features"].astype(int).map({1: "Axis 1", 2: "Axis 1+2"})
    d["endpoint_label"] = d["endpoint"].str.replace("_", " ", regex=False)
    families = ["EF_lt", "NTproBNP_ge", "NYHA_ge"]
    fig, axes = plt.subplots(1, len(families), figsize=(5.2 * len(families), 4.8), squeeze=False)
    for ax, fam in zip(axes.ravel(), families):
        sub = d[d["endpoint_family"] == fam].copy()
        if len(sub) == 0:
            ax.axis("off")
            continue
        sub = sub.sort_values(["threshold", "n_axis_features"])
        if HAS_SEABORN:
            sns.lineplot(
                data=sub,
                x="threshold",
                y="auroc",
                hue="axis_feature_set",
                marker="o",
                linewidth=2.2,
                ax=ax,
            )
        else:
            for feat, ss in sub.groupby("axis_feature_set"):
                ax.plot(ss["threshold"], ss["auroc"], marker="o", label=feat)
            ax.legend()
        ax.axhline(0.5, color="black", linestyle="--", lw=1)
        ax.set_ylim(0.4, 1.0)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("AUROC")
        ax.set_title(fam.replace("_", " "))
    savefig(out_path)


# =============================================================================
# Report
# =============================================================================

def write_markdown(out_dir: Path, config: Dict, panels: Dict[str, List[str]], single_df: pd.DataFrame, align_df: pd.DataFrame, endpoint_df: pd.DataFrame, neg_df: pd.DataFrame):
    lines = ["# Clinically anchored acoustic phenotyping clean summary\n\n"]
    lines.append("CCA is fitted inside each training fold; held-out patients are only transformed/predicted. Fixed clinical columns are used.\n\n")
    lines.append("## Panels\n")
    for p, vs in panels.items():
        lines.append(f"- **{p}**: {', '.join(vs)}\n")
    lines.append("\n## Top single-variable readouts\n")
    if len(single_df):
        top = single_df[single_df["status"] == "ok"].copy()
        top["abs_rho"] = top["spearman_pred_true"].abs()
        lines.append(top.sort_values("abs_rho", ascending=False).head(8).to_markdown(index=False))
    lines.append("\n\n## Alignment summary\n")
    lines.append(align_df.to_markdown(index=False) if len(align_df) else "empty")
    lines.append("\n\n## Endpoint summary\n")
    lines.append(endpoint_df.to_markdown(index=False) if len(endpoint_df) else "empty")
    lines.append("\n\n## Negative control summary\n")
    lines.append(neg_df.to_markdown(index=False) if len(neg_df) else "empty")
    lines.append("\n\n## Config\n```json\n")
    lines.append(json.dumps(config, indent=2, ensure_ascii=False))
    lines.append("\n```\n")
    (out_dir / "analysis_summary.md").write_text("".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Clean clinically anchored acoustic phenotyping with fixed aligned_clinical_clean schema")
    p.add_argument("--embedding-dir", type=str, default="Representation_learning/embeddings_4_1/beats")
    p.add_argument("--patient-embedding-npy", type=str, default=None)
    p.add_argument("--patient-meta-csv", type=str, default=None)
    p.add_argument("--embedding-patient-id-col", type=str, default=None)
    p.add_argument("--clinical-csv", type=str, default="Clinical_alignment/outputs/prepared/aligned_clinical_clean.csv")
    p.add_argument("--out-dir", type=str, default="Clinical_alignment/outputs/clinically_anchored_acoustic_phenotyping_clean")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-components", type=int, default=2)
    p.add_argument("--n-pca", type=int, default=50)
    p.add_argument("--cca-max-iter", type=int, default=3000)
    p.add_argument("--min-panel-n", type=int, default=80)
    p.add_argument("--min-target-n", type=int, default=50)
    p.add_argument("--min-nonmissing-clinical-vars", type=int, default=2)
    p.add_argument("--n-axis-groups", type=int, default=4)
    p.add_argument("--min-endpoint-n", type=int, default=80)
    p.add_argument("--min-endpoint-class-n", type=int, default=5)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-loading-bootstrap", type=int, default=200)
    p.add_argument("--n-permutations", type=int, default=100)
    p.add_argument("--n-random-controls", type=int, default=100)
    p.add_argument(
        "--n-random-split-repeats",
        type=int,
        default=10,
        help="Number of repeated random 5-fold split runs for CCA and endpoint robustness.",
    )
    p.add_argument("--skip-repeated-random-split", action="store_true", help="Skip repeated random-split robustness analyses.")
    p.add_argument("--skip-hyperparameter-sensitivity", action="store_true", help="Skip n_pca/n_components hyperparameter sensitivity analyses.")
    p.add_argument("--sensitivity-n-pca-values", type=str, default="20,50,100")
    p.add_argument("--sensitivity-n-components-values", type=str, default="1,2")
    p.add_argument("--skip-endpoint-threshold-sensitivity", action="store_true", help="Skip endpoint-threshold sensitivity analysis.")
    p.add_argument("--run-position-contribution", action="store_true", default=True)
    p.add_argument("--no-position-contribution", dest="run_position_contribution", action="store_false")
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--position-embedding-npy", type=str, default=None)
    p.add_argument("--position-meta-csv", type=str, default=None)
    p.add_argument("--position-patient-id-col", type=str, default=None)
    p.add_argument("--position-col", type=str, default=None)
    p.add_argument("--positions", type=str, default="A,E,M,P,T")
    p.add_argument("--run-leave-one-position-out", action="store_true", default=True)
    p.add_argument("--no-leave-one-position-out", dest="run_leave_one_position_out", action="store_false")
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

    Xraw, meta = load_embeddings(args)
    clinical_raw = load_clinical(args)
    X, clinical, patient_ids = align_data(Xraw, meta, clinical_raw)

    panels = {**CLINICAL_PANELS, "all_clinical": ALL_CLINICAL_VARS}
    pd.DataFrame([{"panel": p, "variable": v, "domain": VARIABLE_DOMAIN.get(v, "All")}
                  for p, vs in panels.items() for v in vs]).to_csv(table_dir / "clinical_panels_used.csv", index=False, encoding="utf-8-sig")

    # Single-variable overview.
    single_df, single_pred = single_variable_readout(X, clinical, patient_ids, args)
    single_df.to_csv(table_dir / "single_variable_readout.csv", index=False, encoding="utf-8-sig")
    single_pred.to_csv(table_dir / "single_variable_predictions.csv", index=False, encoding="utf-8-sig")
    plot_single_variable_lollipop(single_df, fig_dir / "single_variable_readout_lollipop_by_domain.png")

    # Domain and all-clinical CCA analyses.
    score_dfs, fold_dfs, align_dfs, assoc_dfs, fold_assoc_dfs, signstab_dfs = [], [], [], [], [], []
    gradient_patient_dfs, gradient_summary_dfs = [], []
    boot_dfs, boot_summary_dfs = [], []
    neg_ctrl_dfs, neg_summary_rows = [], []

    for panel, vars_ in panels.items():
        log(f"===== Panel: {panel} =====")
        score, fold, Cp, pids = run_oof_cca(X, clinical, patient_ids, panel, vars_, args)
        score_dfs.append(score); fold_dfs.append(fold)
        align = summarize_alignment(score, Cp, vars_, args)
        # Negative control for every panel: permutation; random embedding only for all_clinical.
        obs = align.loc[align["component"] == 1, "spearman_acoustic_vs_clinical_axis"].iloc[0]
        pctrl, psum = permutation_test(X, clinical, patient_ids, panel, vars_, obs, args)
        neg_ctrl_dfs.append(pctrl); neg_summary_rows.append(psum)
        align.loc[align["component"] == 1, "permutation_p_abs"] = psum["empirical_p_abs_ge_observed"]
        if panel == "all_clinical":
            rctrl, rsum = random_embedding_control(X, clinical, patient_ids, panel, vars_, obs, args)
            neg_ctrl_dfs.append(rctrl); neg_summary_rows.append(rsum)
        align_dfs.append(align)

        assoc, fold_assoc, signstab = axis_clinical_associations(score, Cp, vars_, args)
        assoc_dfs.append(assoc); fold_assoc_dfs.append(fold_assoc); signstab_dfs.append(signstab)
        gpat = clinical_gradient_patient_values(score, Cp, vars_, args)
        gsum = clinical_gradient_summary(gpat)
        gradient_patient_dfs.append(gpat); gradient_summary_dfs.append(gsum)
        try:
            min_non = min(args.min_nonmissing_clinical_vars, len(vars_))
            valid = clinical[vars_].notna().sum(axis=1).to_numpy() >= min_non
            boot, bootsum = bootstrap_loading_stability(X[valid], clinical.loc[valid].reset_index(drop=True), panel, vars_, args)
        except Exception as e:
            log(f"Bootstrap loading skipped for {panel}: {e}")
            boot, bootsum = pd.DataFrame(), pd.DataFrame()
        boot_dfs.append(boot); boot_summary_dfs.append(bootsum)

        if panel in CLINICAL_PANELS:
            plot_axis_association_forest(assoc, panel, fig_dir / f"{clean_filename(panel)}_axis1_clinical_association_forest.png")
            plot_gradient_continuous_box_points(
                gpat, gsum, panel,
                fig_dir / f"{clean_filename(panel)}_axis1_continuous_gradient_box_point.png",
            )
            plot_gradient_ordinal_heatmap(
                gsum, panel,
                fig_dir / f"{clean_filename(panel)}_axis1_ordinal_gradient_heatmap.png",
            )

    scores_all = pd.concat(score_dfs, ignore_index=True)
    folds_all = pd.concat(fold_dfs, ignore_index=True)
    alignment_summary = pd.concat(align_dfs, ignore_index=True)
    assoc_all = pd.concat(assoc_dfs, ignore_index=True)
    fold_assoc_all = pd.concat(fold_assoc_dfs, ignore_index=True)
    signstab_all = pd.concat(signstab_dfs, ignore_index=True)
    gradient_patient_all = pd.concat(gradient_patient_dfs, ignore_index=True)
    gradient_summary_all = pd.concat(gradient_summary_dfs, ignore_index=True)
    if len(gradient_summary_all) and "trend_spearman_p" in gradient_summary_all.columns:
        uniq_grad = gradient_summary_all.drop_duplicates(["panel", "adjustment", "component", "variable"]).copy()
        uniq_grad["trend_spearman_fdr_global"] = fdr_bh(uniq_grad["trend_spearman_p"].values)
        uniq_grad["kruskal_fdr_global"] = fdr_bh(uniq_grad["kruskal_p"].values)
        gradient_summary_all = gradient_summary_all.merge(
            uniq_grad[["panel", "adjustment", "component", "variable", "trend_spearman_fdr_global", "kruskal_fdr_global"]],
            on=["panel", "adjustment", "component", "variable"],
            how="left",
        )
    boot_all = pd.concat(boot_dfs, ignore_index=True) if boot_dfs else pd.DataFrame()
    boot_summary_all = pd.concat(boot_summary_dfs, ignore_index=True) if boot_summary_dfs else pd.DataFrame()
    neg_ctrl_all = pd.concat(neg_ctrl_dfs, ignore_index=True) if neg_ctrl_dfs else pd.DataFrame()
    neg_summary = pd.DataFrame(neg_summary_rows)

    scores_all.to_csv(table_dir / "oof_cca_axis_scores_by_panel.csv", index=False, encoding="utf-8-sig")
    folds_all.to_csv(table_dir / "fold_level_cca_alignment_summary.csv", index=False, encoding="utf-8-sig")
    alignment_summary.to_csv(table_dir / "cca_panel_alignment_summary.csv", index=False, encoding="utf-8-sig")
    assoc_all.to_csv(table_dir / "cca_axis_clinical_associations.csv", index=False, encoding="utf-8-sig")
    fold_assoc_all.to_csv(table_dir / "fold_level_axis_clinical_associations.csv", index=False, encoding="utf-8-sig")
    signstab_all.to_csv(table_dir / "cross_validation_loading_sign_stability.csv", index=False, encoding="utf-8-sig")
    gradient_patient_all.to_csv(table_dir / "clinical_gradient_patient_values.csv", index=False, encoding="utf-8-sig")
    gradient_summary_all.to_csv(table_dir / "clinical_gradient_by_axis.csv", index=False, encoding="utf-8-sig")
    boot_all.to_csv(table_dir / "bootstrap_axis_loading_values.csv", index=False, encoding="utf-8-sig")
    boot_summary_all.to_csv(table_dir / "bootstrap_axis_loading_stability_summary.csv", index=False, encoding="utf-8-sig")
    neg_ctrl_all.to_csv(table_dir / "negative_controls.csv", index=False, encoding="utf-8-sig")
    neg_summary.to_csv(table_dir / "negative_control_summary.csv", index=False, encoding="utf-8-sig")

    plot_alignment_forest(alignment_summary, fig_dir / "domain_specific_cca_alignment_forest.png")
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
    plot_negative_controls(neg_ctrl_all, neg_summary, fig_dir / "negative_controls_all_clinical.png")

    # Repeated random-split robustness.
    # This is separate from bootstrap CIs: it tests whether results depend on one random 5-fold split.
    if not args.skip_repeated_random_split and args.n_random_split_repeats > 0:
        repeated_cca_values, repeated_cca_summary = run_repeated_cca_alignment(
            X, clinical, patient_ids, panels, args
        )
        repeated_cca_values.to_csv(table_dir / "repeated_cv_cca_alignment_values.csv", index=False, encoding="utf-8-sig")
        repeated_cca_summary.to_csv(table_dir / "repeated_cv_cca_alignment_summary.csv", index=False, encoding="utf-8-sig")
        plot_repeated_cca_alignment(
            repeated_cca_values,
            fig_dir / "repeated_cv_cca_axis1_alignment_by_split.png",
            component=1,
        )

    # Anti-circularity analyses.
    # 1) Leave-one-variable-out CCA: the evaluated variable is removed from Y during CCA fitting.
    lovo_summary, lovo_scores, lovo_gradient_patient, lovo_gradient_summary = run_leave_one_variable_out_cca(
        X, clinical, patient_ids, args
    )
    lovo_summary.to_csv(table_dir / "leave_one_variable_out_cca_summary.csv", index=False, encoding="utf-8-sig")
    lovo_scores.to_csv(table_dir / "leave_one_variable_out_cca_axis_scores.csv", index=False, encoding="utf-8-sig")
    lovo_gradient_patient.to_csv(table_dir / "leave_one_variable_out_gradient_patient_values.csv", index=False, encoding="utf-8-sig")
    lovo_gradient_summary.to_csv(table_dir / "leave_one_variable_out_gradient_summary.csv", index=False, encoding="utf-8-sig")
    plot_leave_one_variable_out_forest(
        lovo_summary,
        fig_dir / "leave_one_variable_out_axis1_heldout_variable_forest.png",
        component=1,
    )
    plot_leave_one_variable_out_gradient_heatmap(
        lovo_gradient_summary,
        fig_dir / "leave_one_variable_out_axis1_gradient_heatmap.png",
        component=1,
    )

    # 2) Cross-domain clinical characterization: evaluate variables not used in the source-domain CCA.
    cross_domain_summary, cross_domain_gradient_patient, cross_domain_gradient_summary = run_cross_domain_characterization(
        scores_all, clinical, args
    )
    cross_domain_summary.to_csv(table_dir / "cross_domain_characterization_summary.csv", index=False, encoding="utf-8-sig")
    cross_domain_gradient_patient.to_csv(table_dir / "cross_domain_gradient_patient_values.csv", index=False, encoding="utf-8-sig")
    cross_domain_gradient_summary.to_csv(table_dir / "cross_domain_gradient_summary.csv", index=False, encoding="utf-8-sig")
    plot_cross_domain_forest(
        cross_domain_summary,
        fig_dir / "cross_domain_axis1_clinical_characterization_forest.png",
        component=1,
    )
    plot_cross_domain_dot_heatmap(
        cross_domain_summary,
        fig_dir / "cross_domain_axis1_clinical_characterization_dot_heatmap.png",
        component=1,
    )

    # Endpoint validation and ROC plots.
    endpoint_summary, endpoint_pred, endpoint_info = run_endpoint_validation(X, clinical, patient_ids, args)
    endpoint_summary.to_csv(table_dir / "endpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
    endpoint_axis_comparison = build_endpoint_axis_feature_comparison(endpoint_summary)
    endpoint_axis_comparison.to_csv(table_dir / "endpoint_axis1_vs_axis12_comparison.csv", index=False, encoding="utf-8-sig")
    endpoint_pred.to_csv(table_dir / "endpoint_validation_predictions.csv", index=False, encoding="utf-8-sig")
    endpoint_info.to_csv(table_dir / "endpoints_used.csv", index=False, encoding="utf-8-sig")
    plot_endpoint_summary(endpoint_summary, fig_dir / "endpoint_validation_auroc_forest.png")
    plot_endpoint_roc_curves(endpoint_pred, endpoint_summary, fig_dir, args)

    if not args.skip_repeated_random_split and args.n_random_split_repeats > 0:
        repeated_endpoint_values, repeated_endpoint_summary = run_repeated_endpoint_validation(
            X, clinical, patient_ids, args
        )
        repeated_endpoint_values.to_csv(table_dir / "repeated_cv_endpoint_values.csv", index=False, encoding="utf-8-sig")
        repeated_endpoint_summary.to_csv(table_dir / "repeated_cv_endpoint_summary.csv", index=False, encoding="utf-8-sig")
        repeated_endpoint_axis_comparison = build_endpoint_axis_feature_comparison(repeated_endpoint_summary)
        repeated_endpoint_axis_comparison.to_csv(table_dir / "repeated_cv_endpoint_axis1_vs_axis12_comparison.csv", index=False, encoding="utf-8-sig")
        plot_repeated_endpoint_auroc(
            repeated_endpoint_values,
            fig_dir / "repeated_cv_endpoint_auroc_by_split.png",
        )

    # Confounder and LOPO controls.
    conf_summary, conf_assoc = run_confounder_controls(X, clinical, patient_ids, args)
    conf_summary.to_csv(table_dir / "confounder_adjusted_alignment_summary.csv", index=False, encoding="utf-8-sig")
    conf_assoc.to_csv(table_dir / "confounder_adjusted_axis_clinical_associations.csv", index=False, encoding="utf-8-sig")
    plot_confounder(conf_summary, alignment_summary, fig_dir / "confounder_adjusted_alignment.png")

    lopo = run_lopo_control(X, clinical, patient_ids, args)
    lopo.to_csv(table_dir / "leave_one_position_out_alignment_summary.csv", index=False, encoding="utf-8-sig")
    plot_lopo(lopo, alignment_summary, fig_dir / "leave_one_position_out_alignment.png")

    # Position contribution: full position-concat vs only-A/E/M/P/T.
    position_contribution = run_position_contribution_analysis(clinical, patient_ids, args)
    position_contribution.to_csv(table_dir / "position_contribution_alignment_summary.csv", index=False, encoding="utf-8-sig")
    plot_position_contribution(position_contribution, fig_dir / "position_contribution_alignment.png")

    # Model hyperparameter sensitivity: n_pca, n_components, and endpoint axis-feature options.
    hyper_align, hyper_endpoint = run_hyperparameter_sensitivity(X, clinical, patient_ids, args)
    hyper_align.to_csv(table_dir / "model_hyperparameter_sensitivity_alignment.csv", index=False, encoding="utf-8-sig")
    hyper_endpoint.to_csv(table_dir / "model_hyperparameter_sensitivity_endpoint.csv", index=False, encoding="utf-8-sig")
    plot_hyperparameter_alignment(hyper_align, fig_dir / "model_hyperparameter_sensitivity_alignment.png")
    plot_hyperparameter_endpoint(hyper_endpoint, fig_dir / "model_hyperparameter_sensitivity_endpoint_auroc.png")

    # Endpoint threshold sensitivity: EF, NT-proBNP, and NYHA thresholds.
    threshold_summary, threshold_pred, threshold_info = run_endpoint_threshold_sensitivity(X, clinical, patient_ids, args)
    threshold_summary.to_csv(table_dir / "endpoint_threshold_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    threshold_pred.to_csv(table_dir / "endpoint_threshold_sensitivity_predictions.csv", index=False, encoding="utf-8-sig")
    threshold_info.to_csv(table_dir / "endpoint_threshold_sensitivity_info.csv", index=False, encoding="utf-8-sig")
    plot_endpoint_threshold_sensitivity(threshold_summary, fig_dir / "endpoint_threshold_sensitivity_auroc.png")

    write_markdown(out_dir, config, panels, single_df, alignment_summary, endpoint_summary, neg_summary)
    log(f"Done. Outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
