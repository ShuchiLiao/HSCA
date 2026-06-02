
"""Clinical table cleaning and registry-driven processing for clinical alignment.

This module is responsible for:
1. Detecting and normalizing patient identifiers.
2. Cleaning raw clinical columns into analysis-ready variables.
3. Applying the clinical registry consistently.
4. Building missingness / coverage summaries for later reporting.
5. Constructing technical covariates from window-level metadata so that
   nuisance-control analyses can be added later without reworking the data layer.

Design notes
------------
- Cleaning rules are explicit and readable.
- The implementation follows the user's earlier clinical-cleaning logic,
  adapted to the current distance-based clinical alignment task.
- Heavy statistical analyses are intentionally excluded from this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import re
import argparse

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from clinical_registry import (
        VariableSpec,
        build_variable_registry,
        registry_to_dataframe,
    )
import constants
# =============================================================================
# Console helpers
# =============================================================================


def print_banner(title: str) -> None:
    line = "=" * 96
    print(f"\n{line}\n{title}\n{line}")


def log_info(msg: str) -> None:
    print(f"[info] {msg}")


def log_warn(msg: str) -> None:
    print(f"[warn] {msg}")


def log_done(msg: str) -> None:
    print(f"[done] {msg}")


# =============================================================================
# ID helpers
# =============================================================================


_FLOAT_LIKE_RE = re.compile(r"^\s*[-+]?\d+(?:\.0+)?\s*$")


def detect_patient_id_col(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    """Detect the patient ID column from a candidate list."""
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Cannot find patient ID column. Candidates={list(candidates)} | available={list(df.columns)}"
    )


def normalize_patient_id(series: pd.Series) -> pd.Series:
    """Normalize patient IDs to comparable strings."""
    s = series.copy()

    def _norm_one(x: Any) -> Any:
        if pd.isna(x):
            return np.nan
        text = str(x).replace("\u3000", " ").strip()
        if text == "":
            return np.nan
        if _FLOAT_LIKE_RE.match(text):
            try:
                val = float(text)
                if float(val).is_integer():
                    return str(int(val))
            except Exception:
                pass
        return text

    return s.map(_norm_one).astype("object")


# =============================================================================
# Generic cleaners
# =============================================================================


_NUMERIC_TOKEN_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def coerce_numeric_series(series: pd.Series) -> pd.Series:
    """Coerce a messy object series into float."""
    s = series.copy()

    def _parse_one(x: Any) -> float:
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)

        text = str(x).replace("\u3000", " ").strip()
        if text == "":
            return np.nan

        text = (
            text.replace("，", ",")
            .replace("；", ";")
            .replace("≤", "")
            .replace("≥", "")
            .replace("<", "")
            .replace(">", "")
            .replace("~", "-")
        )
        text = re.sub(r"\s+", "", text)
        if text in {"-", "--", "—", "NA", "N/A", "nan", "None", "null"}:
            return np.nan

        text = text.replace(",", "")
        m = _NUMERIC_TOKEN_RE.search(text)
        if m is None:
            return np.nan
        try:
            return float(m.group(0))
        except Exception:
            return np.nan

    return s.map(_parse_one).astype(float)


_TRUE_SET = {"1", "是", "Y", "YES", "TRUE", "T", "阳性", "有"}
_FALSE_SET = {"0", "否", "N", "NO", "FALSE", "F", "阴性", "无"}


def clean_binary_01(series: pd.Series) -> pd.Series:
    """Normalize mixed binary representations into 0/1 floats."""
    s = series.copy()

    def _one(x: Any) -> float:
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)) and float(x) in (0.0, 1.0):
            return float(x)
        text = str(x).replace("\u3000", " ").strip().upper()
        if text == "":
            return np.nan
        if text in _TRUE_SET:
            return 1.0
        if text in _FALSE_SET:
            return 0.0
        num = coerce_numeric_series(pd.Series([text])).iloc[0]
        if pd.notna(num) and num in (0, 1):
            return float(num)
        return np.nan

    return s.map(_one).astype(float)


def clean_ordinal_default(series: pd.Series) -> pd.Series:
    """Default numeric ordinal cleaner for grades such as 0/1/2/3."""
    vals = coerce_numeric_series(series)
    vals[(vals < 0) | (vals > 10)] = np.nan
    return vals.astype(float)


def clean_nyha(series: pd.Series) -> pd.Series:
    """Normalize NYHA representations into ordered values 1/2/3/4."""
    s = series.copy()
    roman_map = {
        "Ⅰ": 1, "I": 1, "I级": 1, "Ⅰ级": 1, "1": 1, "1级": 1,
        "Ⅱ": 2, "II": 2, "II级": 2, "Ⅱ级": 2, "2": 2, "2级": 2,
        "Ⅲ": 3, "III": 3, "III级": 3, "Ⅲ级": 3, "3": 3, "3级": 3,
        "Ⅳ": 4, "IV": 4, "IV级": 4, "Ⅳ级": 4, "4": 4, "4级": 4,
    }

    def _one(x: Any) -> float:
        if pd.isna(x):
            return np.nan
        text = str(x).replace(" ", "").strip().upper()
        if text == "":
            return np.nan
        text = text.replace("NYHA", "").replace("心功能", "")
        for k, v in roman_map.items():
            if text == k.upper():
                return float(v)
        num = coerce_numeric_series(pd.Series([text])).iloc[0]
        if pd.notna(num) and num in {1, 2, 3, 4}:
            return float(num)
        return np.nan

    return s.map(_one).astype(float)


_PROCEDURE_TOKEN_MAP = {
    "CABG": "2_revascularization",
    "PCI": "2_revascularization",
    "冠脉造影": "4_diagnostic_cath_or_angio",
    "射频消融": "3_rhythm_or_device",
    "ICD": "3_rhythm_or_device",
    "除颤": "3_rhythm_or_device",
    "起搏器": "3_rhythm_or_device",
    "瓣膜": "1_structural_or_valve",
    "置换": "1_structural_or_valve",
    "封堵": "1_structural_or_valve",
}
_CODE_MAP = {
    0: "0_no_procedure",
    1: "1_structural_or_valve",
    2: "1_structural_or_valve",
    3: "3_rhythm_or_device",
    4: "3_rhythm_or_device",
    5: "3_rhythm_or_device",
    6: "4_diagnostic_cath_or_angio",
    7: "2_revascularization",
    8: "2_revascularization",
}


def clean_procedure_category(series: pd.Series) -> pd.Series:
    """Map raw procedure/treatment values into grouped categories."""
    s = series.copy()

    def _codes_from_text(text: str) -> List[int]:
        nums = re.findall(r"\d+", text)
        out: List[int] = []
        for n in nums:
            try:
                out.append(int(n))
            except Exception:
                pass
        return out

    def _one(x: Any) -> Any:
        if pd.isna(x):
            return np.nan
        text = str(x).replace("\u3000", " ").strip()
        if text == "":
            return np.nan
        upper = text.upper().replace("，", ",").replace("；", ",")

        matched: List[str] = []
        for token, cat in _PROCEDURE_TOKEN_MAP.items():
            if token.upper() in upper:
                matched.append(cat)
        for code in _codes_from_text(upper):
            if code in _CODE_MAP:
                matched.append(_CODE_MAP[code])

        matched = sorted(set(matched))
        if len(matched) == 0:
            if upper in {"0", "无", "NONE", "NO", "NO_PROCEDURE"}:
                return "0_no_procedure"
            return "5_other_mixed"
        if len(matched) == 1:
            return matched[0]
        return "5_other_mixed"

    return s.map(_one).astype("object")


def clean_text_series(series: pd.Series) -> pd.Series:
    """Trim free text; keep empty text as NaN."""
    return series.map(lambda x: np.nan if pd.isna(x) or str(x).strip() == "" else str(x).strip()).astype("object")


# =============================================================================
# Registry-driven clinical processing
# =============================================================================


def apply_variable_registry(
    raw_df: pd.DataFrame,
    registry: Sequence[VariableSpec],
    patient_id_candidates: Sequence[str],
) -> pd.DataFrame:
    """Apply the variable registry to generate an analysis-ready clinical table."""
    print_banner("Applying clinical variable registry")

    out = pd.DataFrame(index=raw_df.index)
    id_col = detect_patient_id_col(raw_df, patient_id_candidates)
    out["patient_id"] = normalize_patient_id(raw_df[id_col])

    iterator = tqdm(registry, desc="registry-apply", leave=False)
    for spec in iterator:
        if spec.group == "exclude":
            continue
        if spec.raw_name == "__technical__":
            # Technical covariates are added later from window metadata.
            continue

        if spec.raw_name not in raw_df.columns:
            out[spec.clean_name] = np.nan
            log_warn(f"Missing raw clinical column: {spec.raw_name}")
            continue

        src = raw_df[spec.raw_name]
        transform = str(spec.transform).lower()

        if transform == "none":
            if spec.var_type == "continuous":
                cleaned = coerce_numeric_series(src)
            elif spec.var_type == "binary":
                cleaned = clean_binary_01(src)
            elif spec.var_type == "ordinal":
                cleaned = clean_ordinal_default(src)
            elif spec.var_type == "text":
                cleaned = clean_text_series(src)
            else:
                cleaned = clean_text_series(src)
        elif transform == "log1p":
            cleaned = coerce_numeric_series(src)
            cleaned = cleaned.where(cleaned >= 0, np.nan)
            cleaned = np.log1p(cleaned).astype(float)
        elif transform == "binary_map":
            cleaned = clean_binary_01(src)
        elif transform == "nyha_map":
            cleaned = clean_nyha(src)
        elif transform == "ordinal_map":
            cleaned = clean_ordinal_default(src)
        elif transform == "procedure_map":
            cleaned = clean_procedure_category(src)
        elif transform == "text_group":
            cleaned = clean_text_series(src)
        else:
            cleaned = clean_text_series(src)

        out[spec.clean_name] = cleaned

    n_before = len(out)
    out = out.loc[out["patient_id"].notna()].copy()
    if len(out) < n_before:
        log_warn(f"Dropped {n_before - len(out)} rows due to missing patient_id after normalization.")

    if out["patient_id"].duplicated().any():
        dup_n = int(out["patient_id"].duplicated().sum())
        log_warn(f"Found duplicated patient_id rows: {dup_n}. Keeping first occurrence.")
        out = out.drop_duplicates(subset=["patient_id"], keep="first").copy()

    out = out.reset_index(drop=True)
    log_done(f"Clinical clean table built | shape={out.shape}")
    return out


def build_missingness_table(clean_df: pd.DataFrame, registry: Sequence[VariableSpec]) -> pd.DataFrame:
    """Build a variable-level missingness / coverage table."""
    rows: List[Dict[str, Any]] = []
    n_rows = int(len(clean_df))

    for spec in registry:
        if spec.group == "exclude":
            continue
        col = spec.clean_name
        n_non_missing = int(clean_df[col].notna().sum()) if col in clean_df.columns else 0
        coverage = float(n_non_missing / n_rows) if n_rows > 0 else np.nan
        rows.append(
            {
                "raw_name": spec.raw_name,
                "clean_name": spec.clean_name,
                "group": spec.group,
                "var_type": spec.var_type,
                "transform": spec.transform,
                "use_for_distance": bool(spec.use_for_distance),
                "use_for_neighbor": bool(spec.use_for_neighbor),
                "use_for_retrieval": bool(spec.use_for_retrieval),
                "use_for_adjustment": bool(spec.use_for_adjustment),
                "n_rows": n_rows,
                "n_non_missing": n_non_missing,
                "n_missing": int(n_rows - n_non_missing),
                "coverage": coverage,
                "notes": spec.notes,
            }
        )

    out = pd.DataFrame(rows).sort_values(["group", "clean_name"]).reset_index(drop=True)
    log_done(f"Missingness table built | rows={len(out)}")
    return out


# =============================================================================
# Technical covariates from window metadata
# =============================================================================


def build_technical_covariates_from_window_meta(
    window_meta_csv: str | Path,
    patient_id_col: str = "patient_id",
    position_col: str = "position",
    window_idx_col: str = "window_idx",
    window_seconds: float = 4.0,
    stride_seconds: float = 1.0,
) -> pd.DataFrame:
    """Build patient-level technical covariates from window metadata.

    Expected columns in the metadata table
    --------------------------------------
    patient_id
    position
    window_idx

    Notes
    -----
    Duration is estimated from fixed windowing:
        duration = window_seconds + (n_windows - 1) * stride_seconds
    for n_windows >= 1, otherwise 0.
    """
    window_meta_csv = Path(window_meta_csv)
    if not window_meta_csv.exists():
        raise FileNotFoundError(f"window_meta.csv not found: {window_meta_csv}")

    print_banner("Building technical covariates from window metadata")
    log_info(f"window_meta_csv : {window_meta_csv}")

    meta = pd.read_csv(window_meta_csv)
    for col in [patient_id_col, position_col, window_idx_col]:
        if col not in meta.columns:
            raise ValueError(f"window meta missing required column: {col}")

    meta = meta[[patient_id_col, position_col, window_idx_col]].copy()
    meta[patient_id_col] = normalize_patient_id(meta[patient_id_col])
    meta[position_col] = meta[position_col].astype(str).str.strip().str.upper()
    meta = meta.loc[meta[patient_id_col].notna()].copy()

    count_df = (
        meta.groupby([patient_id_col, position_col], observed=False)
        .size()
        .reset_index(name="n_windows")
    )

    positions = ["A", "E", "M", "P", "T"]
    patient_ids = sorted(count_df[patient_id_col].dropna().astype(str).unique().tolist())
    out = pd.DataFrame({"patient_id": patient_ids})

    for pos in positions:
        sub = count_df.loc[count_df[position_col] == pos, [patient_id_col, "n_windows"]].rename(columns={"n_windows": f"n_windows_{pos}"})
        out = out.merge(sub, on="patient_id", how="left")

    for pos in positions:
        n_col = f"n_windows_{pos}"
        d_col = f"duration_{pos}"
        out[n_col] = pd.to_numeric(out[n_col], errors="coerce").fillna(0).astype(int)
        out[d_col] = np.where(
            out[n_col] > 0,
            float(window_seconds) + np.maximum(out[n_col] - 1, 0) * float(stride_seconds),
            0.0,
        )

    n_cols = [f"n_windows_{p}" for p in positions]
    d_cols = [f"duration_{p}" for p in positions]
    out["n_windows_total"] = out[n_cols].sum(axis=1).astype(int)
    out["duration_total"] = out[d_cols].sum(axis=1).astype(float)

    log_done(f"Technical covariates built | shape={out.shape}")
    return out


def merge_clinical_and_technical_covariates(clinical_clean_df: pd.DataFrame, technical_df: pd.DataFrame) -> pd.DataFrame:
    """Merge cleaned clinical variables with patient-level technical covariates, keeping only patients with valid windows."""
    print_banner("Merging clinical and technical covariates")
    if "patient_id" not in clinical_clean_df.columns:
        raise ValueError("clinical_clean_df must contain patient_id")
    if "patient_id" not in technical_df.columns:
        raise ValueError("technical_df must contain patient_id")

    left = clinical_clean_df.copy()
    right = technical_df.copy()
    left["patient_id"] = normalize_patient_id(left["patient_id"])
    right["patient_id"] = normalize_patient_id(right["patient_id"])

    n_left = len(left)
    n_right = len(right)
    merged = left.merge(right, on="patient_id", how="inner", validate="1:1")
    log_info(f"Clinical patients: {n_left} | window-meta patients: {n_right} | matched patients: {len(merged)}")
    log_done(f"Merged clinical + technical table | shape={merged.shape}")
    return merged


def build_adjustment_strata(
    df: pd.DataFrame,
    age_col: str = "age_years",
    sex_col: str = "sex_male",
    duration_col: str = "duration_total",
    n_windows_col: str = "n_windows_total",
) -> pd.DataFrame:
    """Build coarse strata used in later robustness analyses."""
    out = df.copy()

    def _qbin(series: pd.Series, q: int, name: str) -> pd.Series:
        x = pd.to_numeric(series, errors="coerce")
        valid = x.notna().sum()
        if valid < max(10, q):
            return pd.Series([np.nan] * len(x), index=x.index, name=name)
        try:
            return pd.qcut(x, q=q, labels=False, duplicates="drop").astype("float")
        except Exception:
            return pd.Series([np.nan] * len(x), index=x.index, name=name)

    if age_col in out.columns:
        out["age_bin"] = _qbin(out[age_col], q=5, name="age_bin")
    if duration_col in out.columns:
        out["duration_bin"] = _qbin(out[duration_col], q=5, name="duration_bin")
    if n_windows_col in out.columns:
        out["n_windows_bin"] = _qbin(out[n_windows_col], q=5, name="n_windows_bin")
    if sex_col in out.columns:
        out["sex_group"] = pd.to_numeric(out[sex_col], errors="coerce")

    return out


# =============================================================================
# One-stop builder
# =============================================================================


def build_analysis_ready_clinical_table(
    patient_info_df: pd.DataFrame,
    registry: Sequence[VariableSpec],
    patient_id_candidates: Sequence[str],
    window_meta_csv: str | Path | None = None,
    window_seconds: float = 4.0,
    stride_seconds: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the cleaned clinical table and variable-level missingness summary."""
    print_banner("Building analysis-ready clinical table")
    clinical_clean_df = apply_variable_registry(patient_info_df, registry, patient_id_candidates=patient_id_candidates)

    if window_meta_csv is not None:
        technical_df = build_technical_covariates_from_window_meta(
            window_meta_csv=window_meta_csv,
            patient_id_col="patient_id",
            position_col="position",
            window_idx_col="window_idx",
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
        )
        clinical_clean_df = merge_clinical_and_technical_covariates(clinical_clean_df, technical_df)
        clinical_clean_df = build_adjustment_strata(clinical_clean_df)

    missingness_df = build_missingness_table(clinical_clean_df, registry)
    return clinical_clean_df, missingness_df


def main(argv = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run clinical processing
    Python:
        from Scripts.Alignment.clinical_processing import main
        clinical_clean_df, missingness_df = main([])

    Command line:
        python -m Scripts.Alignment.clinical_processing
    """
    parser = argparse.ArgumentParser(description="Build analysis-ready clinical table for clinical alignment.")
    parser.add_argument("--patient-info", type=Path, default=Path("Data/patient_info.xlsx"))
    parser.add_argument("--window-meta", type=Path, default=Path("Outputs/preprocessing/Data_windows/window_index.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("Outputs/preprocessing/Data_clinic"))
    parser.add_argument("--patient-id-candidates", nargs="+", default=["编码", "patient_id", "患者编号", "ID"])
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--no-window-meta", action="store_true")

    args = parser.parse_args(argv)

    patient_info_path = args.patient_info
    window_meta_path = args.window_meta
    outdir = args.outdir

    print_banner("Clinical processing runner")
    log_info(f"patient_info_path  : {patient_info_path}")
    log_info(f"window_meta_path   : {window_meta_path}")
    log_info(f"outdir             : {outdir}")

    if not patient_info_path.exists():
        raise FileNotFoundError(f"patient_info.xlsx not found: {patient_info_path}")

    outdir.mkdir(parents=True, exist_ok=True)

    patient_info_df = pd.read_excel(patient_info_path)
    registry = build_variable_registry()

    use_window_meta = (not args.no_window_meta) and window_meta_path.exists()
    if args.no_window_meta:
        log_warn("Window metadata disabled by --no-window-meta.")
    elif not window_meta_path.exists():
        log_warn(f"Window metadata not found. Technical covariates will be skipped: {window_meta_path}")

    clinical_clean_df, missingness_df = build_analysis_ready_clinical_table(
        patient_info_df=patient_info_df,
        registry=registry,
        patient_id_candidates=args.patient_id_candidates,
        window_meta_csv=window_meta_path if use_window_meta else None,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    )

    registry_df = registry_to_dataframe(registry)

    clinical_clean_path = outdir / "clinical_clean.csv"
    missingness_path = outdir / "clinical_missingness.csv"
    registry_path = outdir / "clinical_variable_registry.csv"

    clinical_clean_df.to_csv(clinical_clean_path, index=False, encoding="utf-8-sig")
    missingness_df.to_csv(missingness_path, index=False, encoding="utf-8-sig")
    registry_df.to_csv(registry_path, index=False, encoding="utf-8-sig")

    log_done(f"Saved clinical clean table      : {clinical_clean_path}")
    log_done(f"Saved missingness table         : {missingness_path}")
    log_done(f"Saved clinical variable registry: {registry_path}")

    return clinical_clean_df, missingness_df


if __name__ == "__main__":
    min_windows_per_position = constants.MIN_WINDOWN_PER_POSTISIONS
    patient_pass_min_positions = constants.PATIENT_PASS_MIN_POSTISIONS
    window_sec = constants.WINDOW_SEC
    stride_sec = constants.STRIDE_SEC
    patient_info = r"D:\PycharmProjects\HSCA\Data\patient_info.xlsx"
    outdir = constants.OUTPUT_FOLDER/"preprocessing"/"Data_clinic"/f"{min_windows_per_position}_{patient_pass_min_positions}_{window_sec}_{stride_sec}"
    window_meta = (constants.OUTPUT_FOLDER/"preprocessing"/"Data_windows"/
                   f"windows_{min_windows_per_position}_{patient_pass_min_positions}_{window_sec}_{stride_sec}"/"window_index.csv")
    print(window_meta)

    main_args = ["--patient-info", str(patient_info), "--window-meta", str(window_meta), "--outdir", str(outdir), "--window-seconds", str(window_sec), "--stride-seconds", str(stride_sec)]
    main(main_args)
