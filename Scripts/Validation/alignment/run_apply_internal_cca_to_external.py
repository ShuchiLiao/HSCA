#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Apply the internally fitted BEATs acoustic-clinical CCA axis to an external cohort.

This script is the core alignment step for external validation. It follows the
logic of Scripts/Alignment/run_cca_analysis.py in the HSCA repository, but keeps
only the minimal external-validation workflow:

1. Read internal BEATs patient-level embeddings and internal clinical table.
2. Fit the CCA preprocessing pipeline on the internal cohort only.
3. Read external BEATs patient-level embeddings and cleaned external clinical table.
4. Transform the external cohort using the internally fitted pipeline only.
5. Report external acoustic-clinical axis correlation, bootstrap CI, one
   permutation negative control, and AUROC for fixed clinical endpoints.

No model selection, window-length selection, endpoint selection, or CCA refitting
is performed on the external cohort.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


PATIENT_ID_COL = "patient_id"

PANEL_NAME = "functional_impairment_hf_burden"
CLINICAL_VARS = ["EF_Teich", "NTproBNP", "NYHA"]
ENDPOINTS = ["EF_lt_40", "NYHA_ge_3", "NTproBNP_ge_900", "LVEDD_dilated"]

# Same clinical-burden orientation as Scripts/Alignment/run_cca_analysis.py:
# higher acoustic axis should mean heavier burden; EF is reversed.
BURDEN_DIRECTION: Dict[str, int] = {"EF_Teich": -1, "NTproBNP": +1, "NYHA": +1, "LVEDD_mm": +1}


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_patient_id(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return re.sub(r"\s+", "", s)


def infer_patient_col(df: pd.DataFrame, explicit: Optional[str] = None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Patient ID column not found: {explicit}")
        return explicit
    for c in ["patient_id", "patient", "pid", "ID", "id", "subject_id", "序号", "病人编码"]:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = str(c).lower()
        if "patient" in lc or lc in {"pid", "id"}:
            return c
    raise ValueError(f"Cannot infer patient ID column. Columns={list(df.columns)}")


def find_first_existing(base: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        p = base / name
        if p.exists():
            return p
    return None


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def safe_spearman(x, y) -> Tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 5 or np.nanstd(x[mask]) < 1e-12 or np.nanstd(y[mask]) < 1e-12:
        return np.nan, np.nan, n
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p), n


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


def bootstrap_ci_auc(y_true, y_score, n_boot: int, seed: int) -> Tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if len(y) < 10 or len(np.unique(y)) < 2 or n_boot <= 0:
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


# -----------------------------------------------------------------------------
# Data loading and alignment
# -----------------------------------------------------------------------------

def load_patient_embeddings(embedding_dir: Path, patient_embedding_npy: Optional[Path], patient_meta_csv: Optional[Path], patient_id_col: Optional[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    embedding_dir = Path(embedding_dir)
    emb_path = Path(patient_embedding_npy) if patient_embedding_npy else find_first_existing(embedding_dir, ["patient_embeddings.npy", "patient_embedding.npy", "patient_level_embeddings.npy"])
    meta_path = Path(patient_meta_csv) if patient_meta_csv else find_first_existing(embedding_dir, ["patient_meta.csv", "patient_metadata.csv", "patient_embeddings_meta.csv", "patient_order.csv", "patient_ids.csv"])
    if emb_path is None or not emb_path.exists():
        raise FileNotFoundError(f"Cannot find patient embedding .npy under {embedding_dir}")
    if meta_path is None or not meta_path.exists():
        raise FileNotFoundError(f"Cannot find patient meta CSV under {embedding_dir}")

    X = np.load(emb_path).astype(np.float32)
    meta = pd.read_csv(meta_path)
    meta.columns = [str(c).replace("\ufeff", "").strip() for c in meta.columns]
    if X.ndim != 2:
        raise ValueError(f"Expected 2D patient embeddings, got shape={X.shape}")
    if len(meta) != X.shape[0]:
        raise ValueError(f"Meta rows {len(meta)} != embedding rows {X.shape[0]}")

    pid_col = infer_patient_col(meta, patient_id_col)
    meta = meta.copy()
    meta[PATIENT_ID_COL] = meta[pid_col].map(normalize_patient_id)
    meta["embedding_row"] = np.arange(len(meta))
    meta = meta.dropna(subset=[PATIENT_ID_COL]).drop_duplicates(PATIENT_ID_COL, keep="first")
    log(f"Loaded embeddings: {emb_path}, shape={X.shape}")
    log(f"Loaded embedding meta: {meta_path}, patient column={pid_col}")
    return X, meta[[PATIENT_ID_COL, "embedding_row"]]


def load_clinical_table(path: Path, patient_id_col: Optional[str], required_panel_vars: Sequence[str]) -> pd.DataFrame:
    df = read_table(path)
    pid_col = infer_patient_col(df, patient_id_col)
    df = df.copy()
    df[PATIENT_ID_COL] = df[pid_col].map(normalize_patient_id)
    df = df.dropna(subset=[PATIENT_ID_COL]).drop_duplicates(PATIENT_ID_COL, keep="first")

    for c in required_panel_vars:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Optional endpoint-related columns are converted if present; absent columns are handled later.
    for c in ["EF_Teich", "NTproBNP", "NTproBNP_raw", "NYHA", "LVEDD_mm", "sex_male"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    log(f"Loaded clinical table: {path}, n={len(df)}")
    return df


def load_endpoint_table(path: Optional[Path], patient_id_col: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None or not Path(path).exists():
        return None
    df = read_table(Path(path))
    pid_col = infer_patient_col(df, patient_id_col)
    df = df.copy()
    df[PATIENT_ID_COL] = df[pid_col].map(normalize_patient_id)
    df = df.dropna(subset=[PATIENT_ID_COL]).drop_duplicates(PATIENT_ID_COL, keep="first")
    for c in ENDPOINTS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    log(f"Loaded external endpoint table: {path}, n={len(df)}")
    return df[[c for c in [PATIENT_ID_COL] + ENDPOINTS if c in df.columns]]


def align_embeddings_and_clinical(X: np.ndarray, meta: pd.DataFrame, clinical: pd.DataFrame, label: str) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    merged = meta.merge(clinical, on=PATIENT_ID_COL, how="inner")
    if len(merged) == 0:
        raise ValueError(f"No overlapping patient IDs between {label} embeddings and clinical table")
    rows = merged["embedding_row"].to_numpy(dtype=int)
    X_aligned = X[rows]
    finite = np.isfinite(X_aligned).all(axis=1)
    if not finite.all():
        log(f"{label}: dropping {(~finite).sum()} patients with non-finite embeddings")
        X_aligned = X_aligned[finite]
        merged = merged.loc[finite].reset_index(drop=True)
    clinical_aligned = merged.drop(columns=["embedding_row"]).reset_index(drop=True)
    patient_ids = clinical_aligned[PATIENT_ID_COL].tolist()
    log(f"Aligned {label} data: n={len(patient_ids)} patients")
    return X_aligned, clinical_aligned, patient_ids


# -----------------------------------------------------------------------------
# Internal CCA fit and external transform
# -----------------------------------------------------------------------------

def choose_n_pca(n_samples: int, n_features: int, requested: int) -> int:
    return int(max(1, min(requested, n_features, n_samples - 2)))


def as_score_matrix(z, n_rows: int) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    z = np.atleast_2d(z)
    if z.shape[0] != n_rows and z.shape[1] == n_rows:
        z = z.T
    if z.shape[0] != n_rows:
        raise ValueError(f"Cannot orient CCA score matrix, got shape={z.shape}, expected rows={n_rows}")
    return z


def orient_axis_by_internal_burden(acoustic_scores: np.ndarray, clinical: pd.DataFrame, used_vars: Sequence[str]) -> int:
    evidence = []
    x = acoustic_scores[:, 0]
    for v in used_vars:
        direction = BURDEN_DIRECTION.get(v, 0)
        if direction == 0 or v not in clinical.columns:
            continue
        rho, _, n = safe_spearman(x, clinical[v].to_numpy(float))
        if n >= 20 and np.isfinite(rho):
            evidence.append(direction * rho)
    return 1 if (np.nanmean(evidence) if evidence else 0) >= 0 else -1


def fit_internal_cca_pipeline(X_internal: np.ndarray, clinical_internal: pd.DataFrame, clinical_vars: Sequence[str], args) -> Tuple[Dict, np.ndarray, np.ndarray, pd.DataFrame]:
    min_nonmissing = min(args.min_nonmissing_clinical_vars, len(clinical_vars))
    valid = clinical_internal[list(clinical_vars)].notna().sum(axis=1).to_numpy() >= min_nonmissing
    X_fit = X_internal[valid]
    C_fit = clinical_internal.loc[valid].reset_index(drop=True)
    if len(C_fit) < args.min_internal_n:
        raise ValueError(f"Too few internal patients for CCA fit: n={len(C_fit)} < {args.min_internal_n}")

    y_imp = SimpleImputer(strategy="median")
    Y_imp = y_imp.fit_transform(C_fit[list(clinical_vars)])
    keep = np.nanstd(Y_imp, axis=0) > 1e-12
    used_vars = [v for v, k in zip(clinical_vars, keep) if k]
    if len(used_vars) < 2:
        raise ValueError(f"Too few non-constant clinical variables for CCA: used_vars={used_vars}")
    Y_imp = Y_imp[:, keep]

    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(np.asarray(X_fit, dtype=float))
    n_pca = choose_n_pca(len(X_fit), X_fit.shape[1], args.n_pca)
    pca = PCA(n_components=n_pca, random_state=args.seed)
    X_pca = pca.fit_transform(X_scaled)

    y_scaler = StandardScaler()
    Y_scaled = y_scaler.fit_transform(Y_imp)

    n_comp = int(min(1, X_pca.shape[1], Y_scaled.shape[1], len(X_fit) - 2))
    if n_comp < 1:
        raise ValueError("n_components became <1")

    cca = CCA(n_components=n_comp, max_iter=args.cca_max_iter, scale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cca.fit(X_pca, Y_scaled)
        x_score, y_score = cca.transform(X_pca, Y_scaled)

    x_score = as_score_matrix(x_score, len(X_fit))
    y_score = as_score_matrix(y_score, len(X_fit))
    axis_sign = orient_axis_by_internal_burden(x_score, C_fit, used_vars)
    x_score *= axis_sign
    y_score *= axis_sign

    model = {
        "panel": PANEL_NAME,
        "clinical_vars_requested": list(clinical_vars),
        "used_vars": used_vars,
        "keep_clinical_vars_mask": keep.tolist(),
        "clinical_imputer": y_imp,
        "x_scaler": x_scaler,
        "pca": pca,
        "y_scaler": y_scaler,
        "cca": cca,
        "n_pca": n_pca,
        "n_components": n_comp,
        "axis_sign": axis_sign,
        "seed": args.seed,
    }
    log(f"Fitted internal CCA: n={len(C_fit)}, n_pca={n_pca}, used_vars={used_vars}, axis_sign={axis_sign}")
    return model, x_score, y_score, C_fit


def transform_external_with_internal_model(X_external: np.ndarray, clinical_external: pd.DataFrame, model: Dict) -> Tuple[np.ndarray, np.ndarray]:
    clinical_vars = model["clinical_vars_requested"]
    for c in clinical_vars:
        if c not in clinical_external.columns:
            clinical_external[c] = np.nan
        clinical_external[c] = pd.to_numeric(clinical_external[c], errors="coerce")

    X_scaled = model["x_scaler"].transform(np.asarray(X_external, dtype=float))
    X_pca = model["pca"].transform(X_scaled)
    Y_imp_all = model["clinical_imputer"].transform(clinical_external[clinical_vars])
    Y_imp = Y_imp_all[:, np.asarray(model["keep_clinical_vars_mask"], dtype=bool)]
    Y_scaled = model["y_scaler"].transform(Y_imp)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_score, y_score = model["cca"].transform(X_pca, Y_scaled)
    x_score = as_score_matrix(x_score, len(X_external)) * int(model["axis_sign"])
    y_score = as_score_matrix(y_score, len(X_external)) * int(model["axis_sign"])
    return x_score, y_score


# -----------------------------------------------------------------------------
# External validation summaries
# -----------------------------------------------------------------------------

def summarize_external_alignment(score_df: pd.DataFrame, clinical_external: pd.DataFrame, args) -> pd.DataFrame:
    used_vars = [v for v in CLINICAL_VARS if v in clinical_external.columns]
    min_nonmissing = min(args.min_nonmissing_clinical_vars, len(used_vars))
    valid_panel = clinical_external[used_vars].notna().sum(axis=1).to_numpy() >= min_nonmissing

    x = score_df.loc[valid_panel, "cca_acoustic_axis1"].to_numpy(float)
    y = score_df.loc[valid_panel, "cca_clinical_axis1"].to_numpy(float)
    rho, p, n = safe_spearman(x, y)
    lo, hi = bootstrap_ci_spearman(x, y, args.n_bootstrap, args.seed + 101)
    return pd.DataFrame([{
        "panel": PANEL_NAME,
        "component": 1,
        "n_external_aligned": len(score_df),
        "n_external_panel_valid": n,
        "clinical_vars": ";".join(CLINICAL_VARS),
        "min_nonmissing_clinical_vars": min_nonmissing,
        "spearman_acoustic_vs_clinical_axis": rho,
        "spearman_p": p,
        "spearman_ci95_low": lo,
        "spearman_ci95_high": hi,
    }])


def permutation_negative_control(score_df: pd.DataFrame, clinical_external: pd.DataFrame, observed: float, args) -> pd.DataFrame:
    used_vars = [v for v in CLINICAL_VARS if v in clinical_external.columns]
    min_nonmissing = min(args.min_nonmissing_clinical_vars, len(used_vars))
    valid_panel = clinical_external[used_vars].notna().sum(axis=1).to_numpy() >= min_nonmissing
    x = score_df.loc[valid_panel, "cca_acoustic_axis1"].to_numpy(float)
    y = score_df.loc[valid_panel, "cca_clinical_axis1"].to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    rng = np.random.default_rng(args.seed + 444)
    rows = []
    for b in range(args.n_permutation):
        yp = rng.permutation(y)
        rho, p, n = safe_spearman(x, yp)
        rows.append({"iteration": b + 1, "control_type": "external_clinical_axis_permutation", "spearman": rho, "spearman_p": p, "n": n})
    null = pd.DataFrame(rows)
    vals = null["spearman"].dropna().to_numpy(float)
    empirical_p = float((1 + np.sum(np.abs(vals) >= abs(observed))) / (1 + len(vals))) if len(vals) and np.isfinite(observed) else np.nan
    null["observed_spearman"] = observed
    null["empirical_p_abs_ge_observed"] = empirical_p
    return null


def make_endpoint_from_clinical(clinical: pd.DataFrame, name: str) -> Tuple[pd.Series, str, str]:
    y = pd.Series(np.nan, index=clinical.index, dtype=float)
    if name == "EF_lt_40":
        if "EF_Teich" not in clinical.columns:
            return y, "EF_Teich", "missing EF_Teich"
        v = pd.to_numeric(clinical["EF_Teich"], errors="coerce")
        y.loc[v.notna()] = (v.loc[v.notna()] < 40).astype(float)
        return y, "EF_Teich", "EF_Teich < 40"
    if name == "NYHA_ge_3":
        if "NYHA" not in clinical.columns:
            return y, "NYHA", "missing NYHA"
        v = pd.to_numeric(clinical["NYHA"], errors="coerce")
        y.loc[v.notna()] = (v.loc[v.notna()] >= 3).astype(float)
        return y, "NYHA", "NYHA >= 3"
    if name == "NTproBNP_ge_900":
        if "NTproBNP_raw" in clinical.columns:
            v = pd.to_numeric(clinical["NTproBNP_raw"], errors="coerce")
            threshold = 900.0
            source = "NTproBNP_raw"
        elif "NTproBNP" in clinical.columns:
            v = pd.to_numeric(clinical["NTproBNP"], errors="coerce")
            threshold = np.log1p(900.0) if np.nanmax(v.to_numpy(float)) <= 20 else 900.0
            source = "NTproBNP"
        else:
            return y, "NTproBNP", "missing NTproBNP"
        y.loc[v.notna()] = (v.loc[v.notna()] >= threshold).astype(float)
        return y, source, f"{source} >= {threshold:.4g} (900 pg/mL; log1p threshold if transformed)"
    if name == "LVEDD_dilated":
        if "LVEDD_mm" not in clinical.columns:
            return y, "LVEDD_mm", "missing LVEDD_mm"
        v = pd.to_numeric(clinical["LVEDD_mm"], errors="coerce")
        if "sex_male" in clinical.columns:
            sex = pd.to_numeric(clinical["sex_male"], errors="coerce")
            valid = v.notna() & sex.notna()
            thr = pd.Series(np.where(sex >= 0.5, 58.0, 52.0), index=clinical.index)
            y.loc[valid] = (v.loc[valid] > thr.loc[valid]).astype(float)
            return y, "LVEDD_mm", "LVEDD >58 mm male or >52 mm female"
        y.loc[v.notna()] = (v.loc[v.notna()] > 55.0).astype(float)
        return y, "LVEDD_mm", "LVEDD >55 mm when sex is unavailable"
    raise ValueError(name)


def endpoint_series(clinical: pd.DataFrame, endpoint_table: Optional[pd.DataFrame], name: str) -> Tuple[pd.Series, str, str]:
    if endpoint_table is not None and name in endpoint_table.columns:
        merged = clinical[[PATIENT_ID_COL]].merge(endpoint_table[[PATIENT_ID_COL, name]], on=PATIENT_ID_COL, how="left")
        y = pd.to_numeric(merged[name], errors="coerce")
        return y, name, "precomputed endpoint label"
    return make_endpoint_from_clinical(clinical, name)


def evaluate_external_endpoints(score_df: pd.DataFrame, clinical_external: pd.DataFrame, endpoint_table: Optional[pd.DataFrame], args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    pred_rows = []
    score = score_df["cca_acoustic_axis1"].to_numpy(float)
    for i, endpoint in enumerate(ENDPOINTS):
        y_series, source, rule = endpoint_series(clinical_external, endpoint_table, endpoint)
        valid = y_series.notna() & np.isfinite(score)
        y = y_series.loc[valid].astype(int).to_numpy()
        s = score[valid.to_numpy()]
        n = int(len(y))
        n_pos = int(y.sum()) if n else 0
        n_neg = int(n - n_pos)
        row = {
            "endpoint": endpoint,
            "source_column": source,
            "rule": rule,
            "n": n,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "positive_rate": float(n_pos / n) if n else np.nan,
            "score": "cca_acoustic_axis1",
        }
        if n < args.min_endpoint_n:
            row.update({"status": "skipped_too_few_samples", "auroc": np.nan, "auroc_ci95_low": np.nan, "auroc_ci95_high": np.nan})
        elif len(np.unique(y)) < 2:
            row.update({"status": "skipped_single_class", "auroc": np.nan, "auroc_ci95_low": np.nan, "auroc_ci95_high": np.nan})
        elif min(n_pos, n_neg) < args.min_endpoint_class_n:
            row.update({"status": "skipped_too_few_positive_or_negative", "auroc": np.nan, "auroc_ci95_low": np.nan, "auroc_ci95_high": np.nan})
        else:
            auc = float(roc_auc_score(y, s))
            lo, hi = bootstrap_ci_auc(y, s, args.n_bootstrap, args.seed + 2000 + i)
            row.update({"status": "ok", "auroc": auc, "auroc_ci95_low": lo, "auroc_ci95_high": hi})
            tmp = pd.DataFrame({
                PATIENT_ID_COL: clinical_external.loc[valid, PATIENT_ID_COL].to_numpy(),
                "endpoint": endpoint,
                "y_true": y,
                "cca_acoustic_axis1": s,
            })
            pred_rows.append(tmp)
        summary_rows.append(row)
    pred_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    return pd.DataFrame(summary_rows), pred_df


def missingness_summary(clinical: pd.DataFrame, variables: Sequence[str]) -> pd.DataFrame:
    rows = []
    for v in variables:
        if v not in clinical.columns:
            rows.append({"variable": v, "available": False, "n": len(clinical), "n_missing": len(clinical), "missing_rate": 1.0})
            continue
        miss = clinical[v].isna()
        rows.append({"variable": v, "available": True, "n": len(clinical), "n_missing": int(miss.sum()), "missing_rate": float(miss.mean())})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Apply internally fitted BEATs CCA axis to an external validation cohort.")
    parser.add_argument("--internal-embedding-dir", type=Path, default=Path(r"D:\PycharmProjects\HSCA\Outputs\representation\Embeddings\beats\4_5_4_1"))
    parser.add_argument("--internal-patient-embeddings", type=Path, default=None)
    parser.add_argument("--internal-patient-meta", type=Path, default=None)
    parser.add_argument("--internal-clinical-table", type=Path, default=Path(r"D:\PycharmProjects\HSCA\Outputs\preprocessing\Data_clinic\4_5_4_1\clinical_clean.csv"))
    parser.add_argument("--external-embedding-dir", type=Path, default=Path(r"D:\PycharmProjects\HSCA\Outputs\validation\representation\beats"))
    parser.add_argument("--external-patient-embeddings", type=Path, default=None)
    parser.add_argument("--external-patient-meta", type=Path, default=None)
    parser.add_argument("--external-clinical-table", type=Path, default=Path(r"D:\PycharmProjects\HSCA\Outputs\validation\preprocessing\Data_clinic\clinical_clean.csv"))
    parser.add_argument("--external-endpoint-table", type=Path, default=Path(r"D:\PycharmProjects\HSCA\Outputs\validation\preprocessing\Data_clinic\external_endpoint_labels.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path(r"D:\PycharmProjects\HSCA\Outputs\validation\alignment"))
    parser.add_argument("--internal-patient-id-col", type=str, default=None)
    parser.add_argument("--external-patient-id-col", type=str, default=None)
    parser.add_argument("--n-pca", type=int, default=50)
    parser.add_argument("--cca-max-iter", type=int, default=1000)
    parser.add_argument("--min-nonmissing-clinical-vars", type=int, default=2)
    parser.add_argument("--min-internal-n", type=int, default=30)
    parser.add_argument("--min-endpoint-n", type=int, default=20)
    parser.add_argument("--min-endpoint-class-n", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-permutation", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    tables_dir = args.out_dir / "tables"
    models_dir = args.out_dir / "models"
    config_dir = args.out_dir / "config"
    for d in [tables_dir, models_dir, config_dir]:
        ensure_dir(d)

    X_int, meta_int = load_patient_embeddings(args.internal_embedding_dir, args.internal_patient_embeddings, args.internal_patient_meta, args.internal_patient_id_col)
    C_int_raw = load_clinical_table(args.internal_clinical_table, args.internal_patient_id_col, CLINICAL_VARS)
    X_int_aligned, C_int, int_pids = align_embeddings_and_clinical(X_int, meta_int, C_int_raw, "internal")

    model, x_int_score, y_int_score, C_int_fit = fit_internal_cca_pipeline(X_int_aligned, C_int, CLINICAL_VARS, args)
    joblib.dump(model, models_dir / "internal_cca_model.joblib")

    X_ext, meta_ext = load_patient_embeddings(args.external_embedding_dir, args.external_patient_embeddings, args.external_patient_meta, args.external_patient_id_col)
    C_ext_raw = load_clinical_table(args.external_clinical_table, args.external_patient_id_col, CLINICAL_VARS)
    endpoint_table = load_endpoint_table(args.external_endpoint_table, args.external_patient_id_col)
    X_ext_aligned, C_ext, ext_pids = align_embeddings_and_clinical(X_ext, meta_ext, C_ext_raw, "external")

    x_ext_score, y_ext_score = transform_external_with_internal_model(X_ext_aligned, C_ext, model)
    score_df = pd.DataFrame({PATIENT_ID_COL: ext_pids, "panel": PANEL_NAME, "cca_acoustic_axis1": x_ext_score[:, 0], "cca_clinical_axis1": y_ext_score[:, 0]})
    for v in CLINICAL_VARS + ["NTproBNP_raw", "LVEDD_mm", "sex_male"]:
        if v in C_ext.columns:
            score_df[v] = C_ext[v].to_numpy()

    alignment_summary = summarize_external_alignment(score_df, C_ext, args)
    observed = float(alignment_summary.loc[0, "spearman_acoustic_vs_clinical_axis"])
    null_df = permutation_negative_control(score_df, C_ext, observed, args)
    endpoint_summary, endpoint_pred = evaluate_external_endpoints(score_df, C_ext, endpoint_table, args)
    cohort_df = pd.DataFrame({PATIENT_ID_COL: ext_pids, "included_in_external_alignment": True})
    miss_df = missingness_summary(C_ext, CLINICAL_VARS + ["NTproBNP_raw", "LVEDD_mm", "sex_male"])

    score_df.to_csv(tables_dir / "external_axis_scores.csv", index=False, encoding="utf-8-sig")
    alignment_summary.to_csv(tables_dir / "external_main_alignment_summary.csv", index=False, encoding="utf-8-sig")
    endpoint_summary.to_csv(tables_dir / "external_endpoint_auroc_summary.csv", index=False, encoding="utf-8-sig")
    endpoint_pred.to_csv(tables_dir / "external_endpoint_predictions.csv", index=False, encoding="utf-8-sig")
    null_df.to_csv(tables_dir / "external_permutation_null.csv", index=False, encoding="utf-8-sig")
    cohort_df.to_csv(tables_dir / "external_analysis_cohort.csv", index=False, encoding="utf-8-sig")
    miss_df.to_csv(tables_dir / "external_alignment_missingness.csv", index=False, encoding="utf-8-sig")

    config = {
        "panel": PANEL_NAME,
        "clinical_vars": CLINICAL_VARS,
        "endpoints": ENDPOINTS,
        "internal_embedding_dir": str(args.internal_embedding_dir),
        "internal_clinical_table": str(args.internal_clinical_table),
        "external_embedding_dir": str(args.external_embedding_dir),
        "external_clinical_table": str(args.external_clinical_table),
        "external_endpoint_table": str(args.external_endpoint_table),
        "out_dir": str(args.out_dir),
        "n_pca": args.n_pca,
        "n_bootstrap": args.n_bootstrap,
        "n_permutation": args.n_permutation,
        "seed": args.seed,
        "used_vars": model["used_vars"],
        "internal_n_aligned": int(len(C_int)),
        "internal_n_fit": int(len(C_int_fit)),
        "external_n_aligned": int(len(C_ext)),
    }
    with open(config_dir / "external_alignment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    log("Done.")
    log(f"Saved axis scores: {tables_dir / 'external_axis_scores.csv'}")
    log(f"Saved main summary: {tables_dir / 'external_main_alignment_summary.csv'}")
    log(f"Saved endpoint AUROC summary: {tables_dir / 'external_endpoint_auroc_summary.csv'}")
    return score_df, alignment_summary, endpoint_summary


if __name__ == "__main__":
    main()
