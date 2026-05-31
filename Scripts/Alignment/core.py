"""Core data utilities for robust clinical alignment.

This module is responsible for:
1. Loading and validating the patient-level acoustic distance matrix.
2. Cleaning the clinical spreadsheet via the registry-driven processing layer.
3. Building patient-level technical covariates from window metadata.
4. Aligning all tables to the exact patient order used by the acoustic distance matrix.
5. Constructing mixed-type clinical distance matrices for function / structure /
   burden / all groups.
6. Saving a reusable prepared-data bundle for downstream analyses.

Design notes
------------
- The code is intentionally explicit and easy to audit.
- Clinical distance construction uses a readable Gower-like mixed-type scheme.
- The prepared-data object stores both clinical and technical covariates so that
  robustness analyses can be added later without rebuilding the data layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import pickle

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:  # pragma: no cover
    from .config import (
        PipelineConfig,
        build_default_config,
        ensure_output_dirs,
        log_done,
        log_info,
        log_warn,
        print_banner,
    )
    from .clinical_registry import (
        VariableSpec,
        build_variable_registry,
        get_distance_variables,
        registry_to_dataframe,
    )
    from .clinical_processing import (
        build_adjustment_strata,
        build_analysis_ready_clinical_table,
        build_technical_covariates_from_window_meta,
        merge_clinical_and_technical_covariates,
        normalize_patient_id,
    )
except ImportError:  # pragma: no cover
    from config import (
        PipelineConfig,
        build_default_config,
        ensure_output_dirs,
        log_done,
        log_info,
        log_warn,
        print_banner,
    )
    from clinical_registry import (
        VariableSpec,
        build_variable_registry,
        get_distance_variables,
        registry_to_dataframe,
    )
    from clinical_processing import (
        build_adjustment_strata,
        build_analysis_ready_clinical_table,
        build_technical_covariates_from_window_meta,
        merge_clinical_and_technical_covariates,
        normalize_patient_id,
    )


# =============================================================================
# Small helpers
# =============================================================================


def _ensure_square_distance_matrix(D: np.ndarray, name: str) -> np.ndarray:
    """Validate and sanitize a square distance matrix."""
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"{name} must be square, got shape={D.shape}")
    if np.any(~np.isfinite(D)):
        raise ValueError(f"{name} contains NaN or Inf values.")
    if np.any(D < -1e-10):
        raise ValueError(f"{name} contains negative values.")
    if not np.allclose(D, D.T, atol=1e-7):
        raise ValueError(f"{name} is not symmetric within tolerance.")
    np.fill_diagonal(D, 0.0)
    return D



def upper_triangle_vector(D: np.ndarray) -> np.ndarray:
    """Return the upper-triangle vector of a square matrix, excluding diagonal."""
    D = np.asarray(D)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape={D.shape}")
    return D[np.triu_indices_from(D, k=1)]



def _safe_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)



def _robust_continuous_scale(x: np.ndarray) -> float:
    """Return a robust denominator for continuous-variable distance."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 1.0
    q05, q95 = np.quantile(x, [0.05, 0.95])
    denom = float(q95 - q05)
    if denom <= 1e-12:
        denom = float(np.nanmax(x) - np.nanmin(x))
    if denom <= 1e-12:
        denom = 1.0
    return denom



def _max_ordinal_gap(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 1.0
    gap = float(np.nanmax(x) - np.nanmin(x))
    return gap if gap > 1e-12 else 1.0



def _detect_id_col(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Cannot find patient-id column. Candidates={list(candidates)} | columns={list(df.columns)}")


# =============================================================================
# Main data container
# =============================================================================


@dataclass
class ClinicalAlignmentData:
    """Prepared data object shared across the clinical alignment pipeline."""

    patient_order: List[str]
    acoustic_distance: np.ndarray
    clinical_raw: pd.DataFrame
    clinical_clean: pd.DataFrame
    technical_covariates: pd.DataFrame
    clinical_plus_technical: pd.DataFrame
    clinical_plus_technical_strata: pd.DataFrame
    missingness_df: pd.DataFrame
    registry_df: pd.DataFrame
    meta: Dict[str, Any]
    clinical_distance_mats: Dict[str, np.ndarray] = field(default_factory=dict)

    def save(self, out_dir: str | Path) -> None:
        """Save prepared artifacts to disk for later reuse."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "acoustic_distance.npy", self.acoustic_distance)
        pd.DataFrame({"patient_id": self.patient_order}).to_csv(out_dir / "patient_order.csv", index=False)
        self.clinical_raw.to_csv(out_dir / "aligned_clinical_raw.csv", index=False, encoding="utf-8-sig")
        self.clinical_clean.to_csv(out_dir / "aligned_clinical_clean.csv", index=False, encoding="utf-8-sig")
        self.technical_covariates.to_csv(out_dir / "technical_covariates.csv", index=False, encoding="utf-8-sig")
        self.clinical_plus_technical.to_csv(out_dir / "aligned_clinical_plus_technical.csv", index=False, encoding="utf-8-sig")
        self.clinical_plus_technical_strata.to_csv(out_dir / "aligned_clinical_plus_technical_strata.csv", index=False, encoding="utf-8-sig")
        self.missingness_df.to_csv(out_dir / "missingness_summary.csv", index=False, encoding="utf-8-sig")
        self.registry_df.to_csv(out_dir / "registry_snapshot.csv", index=False, encoding="utf-8-sig")
        for group_name, D in self.clinical_distance_mats.items():
            np.save(out_dir / f"clinical_distance_{group_name}.npy", D)
        with (out_dir / "prepared_meta.json").open("w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)
        with (out_dir / "prepared_data.pkl").open("wb") as f:
            pickle.dump(self, f)
        log_done(f"Saved prepared clinical alignment data to: {out_dir}")

    @classmethod
    def load(cls, prepared_dir: str | Path) -> "ClinicalAlignmentData":
        prepared_dir = Path(prepared_dir)
        pkl_path = prepared_dir / "prepared_data.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Prepared-data pickle not found: {pkl_path}")
        with pkl_path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, ClinicalAlignmentData):
            raise TypeError(f"Loaded object is not ClinicalAlignmentData: {type(obj)}")
        log_done(f"Loaded prepared clinical alignment data from: {pkl_path}")
        return obj


# =============================================================================
# Loaders and aligners
# =============================================================================


def load_acoustic_distance(
    acoustic_distance_npy: str | Path,
    patient_order_csv: str | Path,
) -> Tuple[np.ndarray, List[str]]:
    """Load and validate the acoustic distance matrix and patient order."""
    print_banner("Loading acoustic distance matrix")
    acoustic_distance_npy = Path(acoustic_distance_npy)
    patient_order_csv = Path(patient_order_csv)

    if not acoustic_distance_npy.exists():
        raise FileNotFoundError(f"Acoustic distance matrix not found: {acoustic_distance_npy}")
    if not patient_order_csv.exists():
        raise FileNotFoundError(f"patient_order.csv not found: {patient_order_csv}")

    log_info(f"acoustic_distance_npy : {acoustic_distance_npy}")
    log_info(f"patient_order_csv     : {patient_order_csv}")

    D = np.load(acoustic_distance_npy)
    D = _ensure_square_distance_matrix(D, name="acoustic_distance")

    patient_order_df = pd.read_csv(patient_order_csv)
    if "patient_id" not in patient_order_df.columns:
        raise ValueError(f"patient_order.csv must contain 'patient_id', got columns={list(patient_order_df.columns)}")
    patient_order = patient_order_df["patient_id"].astype(str).tolist()

    if len(patient_order) != D.shape[0]:
        raise ValueError(
            f"patient_order length != distance size: {len(patient_order)} vs {D.shape[0]}"
        )
    log_done(f"Loaded acoustic distance matrix | shape={D.shape}")
    return D, patient_order



def load_clinical_table(clinical_table_path: str | Path) -> pd.DataFrame:
    """Load the raw clinical spreadsheet (.xlsx / .xls / .csv)."""
    print_banner("Loading clinical table")
    clinical_table_path = Path(clinical_table_path)
    if not clinical_table_path.exists():
        raise FileNotFoundError(f"Clinical table not found: {clinical_table_path}")
    log_info(f"clinical_table_path : {clinical_table_path}")

    suffix = clinical_table_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(clinical_table_path)
    elif suffix == ".csv":
        df = pd.read_csv(clinical_table_path)
    else:
        raise ValueError(f"Unsupported clinical table format: {suffix}")

    log_done(f"Loaded clinical table | shape={df.shape}")
    return df



def align_table_to_patient_order(
    df: pd.DataFrame,
    patient_order: Sequence[str],
    id_col: str,
) -> pd.DataFrame:
    """Align a table to the exact patient order of the acoustic distance matrix."""
    if id_col not in df.columns:
        raise ValueError(f"align_table_to_patient_order: missing id_col={id_col}")

    out = df.copy()
    out[id_col] = normalize_patient_id(out[id_col])
    out = out.loc[out[id_col].notna()].copy()
    out = out.drop_duplicates(subset=[id_col], keep="first")

    order_df = pd.DataFrame({"patient_id": list(map(str, patient_order))})
    merged = order_df.merge(out, left_on="patient_id", right_on=id_col, how="left", validate="1:1")

    n_missing = int(merged[id_col].isna().sum()) if id_col in merged.columns else len(merged)
    if n_missing > 0:
        log_warn(f"{n_missing} patients in patient_order have no matching row in {id_col}-indexed table.")

    if id_col != "patient_id" and id_col in merged.columns:
        merged = merged.drop(columns=[id_col])
    return merged


# =============================================================================
# Clinical distance matrices
# =============================================================================


def compute_clinical_distance_matrix(
    df: pd.DataFrame,
    registry: Sequence[VariableSpec],
    group_name: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Build a Gower-like mixed-type clinical distance matrix for one group.

    Returns
    -------
    D : np.ndarray
        Clinical distance matrix, shape [N, N].
    vars_df : pd.DataFrame
        Table listing which variables were used and their effective scales.
    """
    print_banner(f"Computing clinical distance matrix | group={group_name}")
    var_specs = get_distance_variables(registry, group=group_name)
    if len(var_specs) == 0:
        raise ValueError(f"No distance variables found for group={group_name}")

    n = len(df)
    accum = np.zeros((n, n), dtype=np.float64)
    weight_sum = np.zeros((n, n), dtype=np.float64)
    var_rows: List[Dict[str, Any]] = []

    iterator = tqdm(var_specs, desc=f"clinical-distance-{group_name}", leave=False)
    for spec in iterator:
        col = spec.clean_name
        if col not in df.columns:
            log_warn(f"Distance variable missing from clinical table: {col}")
            continue
        s = df[col]
        if s.notna().sum() < 5:
            log_warn(f"Distance variable too sparse, skipped: {col}")
            continue

        if spec.var_type == "continuous":
            x = _safe_numeric(s)
            scale = _robust_continuous_scale(x)
            mask = np.isfinite(x)
            diff = np.abs(x[:, None] - x[None, :]) / scale
            valid = mask[:, None] & mask[None, :]
            contrib = np.where(valid, diff, 0.0)
            w = np.where(valid, 1.0, 0.0)
        elif spec.var_type == "ordinal":
            x = _safe_numeric(s)
            scale = _max_ordinal_gap(x)
            mask = np.isfinite(x)
            diff = np.abs(x[:, None] - x[None, :]) / scale
            valid = mask[:, None] & mask[None, :]
            contrib = np.where(valid, diff, 0.0)
            w = np.where(valid, 1.0, 0.0)
        elif spec.var_type in {"binary", "categorical"}:
            x = s.astype("object").to_numpy()
            mask = pd.notna(s).to_numpy()
            neq = (x[:, None] != x[None, :]).astype(np.float64)
            valid = mask[:, None] & mask[None, :]
            contrib = np.where(valid, neq, 0.0)
            w = np.where(valid, 1.0, 0.0)
            scale = 1.0
        else:
            log_warn(f"Unsupported distance variable type, skipped: {col} ({spec.var_type})")
            continue

        accum += contrib * w
        weight_sum += w
        var_rows.append(
            {
                "group": group_name,
                "clean_name": spec.clean_name,
                "raw_name": spec.raw_name,
                "var_type": spec.var_type,
                "weight": 1.0,
                "effective_scale": float(scale),
                "coverage": float(s.notna().mean()),
                "notes": spec.notes,
            }
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        D = np.divide(accum, weight_sum, out=np.zeros_like(accum), where=weight_sum > 0)
    D = _ensure_square_distance_matrix(D, name=f"clinical_distance_{group_name}")
    vars_df = pd.DataFrame(var_rows)
    log_done(f"Clinical distance built | group={group_name} | shape={D.shape} | n_vars={len(vars_df)}")
    return D, vars_df



def compute_single_variable_distance_matrix(
    df: pd.DataFrame,
    spec: VariableSpec,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build a patient-by-patient distance matrix for one single clinical variable.

    Continuous variables use robustly scaled absolute difference.
    Ordinal variables use normalized grade difference.
    Binary / categorical variables use mismatch distance (0/1).
    """
    col = spec.clean_name
    if col not in df.columns:
        raise ValueError(f"Single-variable distance variable missing from dataframe: {col}")
    s = df[col]
    n = len(df)

    if spec.var_type == "continuous":
        x = _safe_numeric(s)
        scale = _robust_continuous_scale(x)
        mask = np.isfinite(x)
        diff = np.abs(x[:, None] - x[None, :]) / scale
        valid = mask[:, None] & mask[None, :]
        D = np.where(valid, diff, 0.0)
    elif spec.var_type == "ordinal":
        x = _safe_numeric(s)
        scale = _max_ordinal_gap(x)
        mask = np.isfinite(x)
        diff = np.abs(x[:, None] - x[None, :]) / scale
        valid = mask[:, None] & mask[None, :]
        D = np.where(valid, diff, 0.0)
    elif spec.var_type in {"binary", "categorical"}:
        x = s.astype("object").to_numpy()
        mask = pd.notna(s).to_numpy()
        neq = (x[:, None] != x[None, :]).astype(np.float64)
        valid = mask[:, None] & mask[None, :]
        D = np.where(valid, neq, 0.0)
        scale = 1.0
    else:
        raise ValueError(f"Unsupported variable type for single-variable distance: {spec.var_type}")

    D = _ensure_square_distance_matrix(D, name=f"single_variable_distance_{col}")
    info = {
        "clean_name": spec.clean_name,
        "raw_name": spec.raw_name,
        "group": spec.group,
        "var_type": spec.var_type,
        "effective_scale": float(scale),
        "coverage": float(s.notna().mean()),
        "notes": spec.notes,
    }
    return D, info


def build_all_single_variable_distance_matrices(
    df: pd.DataFrame,
    registry: Sequence[VariableSpec],
    groups: Sequence[str] = ("function", "structure", "burden"),
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Build one clinical distance matrix per variable for selected clinical groups."""
    mats: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []
    allowed = set(groups)
    for spec in registry:
        if spec.group not in allowed:
            continue
        if spec.clean_name not in df.columns:
            continue
        try:
            D, info = compute_single_variable_distance_matrix(df, spec)
        except Exception as exc:
            log_warn(f"Skipping single-variable distance for {spec.clean_name}: {exc}")
            continue
        mats[spec.clean_name] = D
        rows.append(info)
    info_df = pd.DataFrame(rows).sort_values(["group", "clean_name"]).reset_index(drop=True)
    return mats, info_df




def build_all_clinical_distance_matrices(
    df: pd.DataFrame,
    registry: Sequence[VariableSpec],
    groups: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
    """Build all requested clinical distance matrices."""
    mats: Dict[str, np.ndarray] = {}
    vars_tables: Dict[str, pd.DataFrame] = {}
    for group in groups:
        D, vars_df = compute_clinical_distance_matrix(df, registry, group_name=group)
        mats[str(group)] = D
        vars_tables[str(group)] = vars_df
    return mats, vars_tables



def get_knn_from_distance(D: np.ndarray, k: int) -> np.ndarray:
    """Return top-k nearest-neighbor indices for each patient."""
    D = _ensure_square_distance_matrix(D, name="distance_for_knn")
    n = D.shape[0]
    if not (1 <= int(k) < n):
        raise ValueError(f"k must satisfy 1 <= k < n, got k={k}, n={n}")
    return np.argsort(D, axis=1)[:, 1 : int(k) + 1]


# =============================================================================
# One-stop builder
# =============================================================================


def prepare_clinical_alignment_data(cfg: PipelineConfig) -> ClinicalAlignmentData:
    """One-stop data builder used by downstream clinical-alignment analyses."""
    print_banner("Preparing clinical alignment data")

    D_acoustic, patient_order = load_acoustic_distance(
        cfg.paths.acoustic_distance_npy,
        cfg.paths.patient_order_csv,
    )
    raw_clinical_df = load_clinical_table(cfg.paths.clinical_table_path)
    registry = build_variable_registry()
    clinical_clean_df, missingness_df = build_analysis_ready_clinical_table(
        patient_info_df=raw_clinical_df,
        registry=registry,
        patient_id_candidates=cfg.analysis.patient_id_candidates,
    )
    technical_df = build_technical_covariates_from_window_meta(
        cfg.paths.window_meta_csv,
        patient_id_col="patient_id",
        position_col="position",
        window_idx_col="window_idx",
        window_seconds=4.0,
        stride_seconds=1.0,
    )
    clinical_plus_technical = merge_clinical_and_technical_covariates(clinical_clean_df, technical_df)
    clinical_plus_technical = build_adjustment_strata(clinical_plus_technical)

    raw_id_col = _detect_id_col(raw_clinical_df, cfg.analysis.patient_id_candidates)
    clinical_raw_aligned = align_table_to_patient_order(raw_clinical_df, patient_order, raw_id_col)
    clinical_clean_aligned = align_table_to_patient_order(clinical_clean_df, patient_order, "patient_id")
    technical_aligned = align_table_to_patient_order(technical_df, patient_order, "patient_id")
    cpt_aligned = align_table_to_patient_order(clinical_plus_technical, patient_order, "patient_id")

    clinical_distance_mats, _ = build_all_clinical_distance_matrices(
        df=cpt_aligned,
        registry=registry,
        groups=cfg.analysis.clinical_groups,
    )

    meta = {
        "n_patients": int(len(patient_order)),
        "clinical_groups": list(cfg.analysis.clinical_groups),
        "window_meta_csv": str(cfg.paths.window_meta_csv),
    }
    data = ClinicalAlignmentData(
        patient_order=list(map(str, patient_order)),
        acoustic_distance=D_acoustic,
        clinical_raw=clinical_raw_aligned,
        clinical_clean=clinical_clean_aligned,
        technical_covariates=technical_aligned,
        clinical_plus_technical=cpt_aligned,
        clinical_plus_technical_strata=cpt_aligned,
        missingness_df=missingness_df,
        registry_df=registry_to_dataframe(registry),
        meta=meta,
        clinical_distance_mats=clinical_distance_mats,
    )
    log_done("ClinicalAlignmentData object prepared successfully.")
    return data


# =============================================================================
# CLI
# =============================================================================


def _build_cli() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare aligned clinical data and clinical distance matrices for clinical alignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--acoustic-distance-npy", type=str, default=None)
    parser.add_argument("--patient-order-csv", type=str, default=None)
    parser.add_argument("--clinical-table-path", type=str, default=None)
    parser.add_argument("--window-meta-csv", type=str, default=None)
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser



def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()
    cfg = build_default_config()
    if args.acoustic_distance_npy:
        cfg.paths.acoustic_distance_npy = args.acoustic_distance_npy
    if args.patient_order_csv:
        cfg.paths.patient_order_csv = args.patient_order_csv
    if args.clinical_table_path:
        cfg.paths.clinical_table_path = args.clinical_table_path
    if args.window_meta_csv:
        cfg.paths.window_meta_csv = args.window_meta_csv
    if args.out_root:
        cfg.paths.output_root = args.out_root
    if args.seed is not None:
        cfg.analysis.random_seed = int(args.seed)

    dirs = ensure_output_dirs(cfg.paths.output_root)
    data = prepare_clinical_alignment_data(cfg)
    data.save(dirs["prepared"])
    for group_name, D in data.clinical_distance_mats.items():
        np.save(dirs["global_alignment"] / f"clinical_distance_{group_name}.npy", D)
    log_done("Core preparation finished.")


if __name__ == "__main__":  # pragma: no cover
    main()
