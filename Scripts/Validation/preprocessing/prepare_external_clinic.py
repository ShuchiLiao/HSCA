#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare external clinical table for the heart-sound external validation cohort.

This script is the external-cohort counterpart of Scripts/Preprocessing/clinical_processing.py.
It does not introduce a new clinical-processing pipeline. Instead, it:

1. Reads the external clinical spreadsheet.
2. Maps external column names to the raw column names expected by clinical_registry.py.
3. Calls the registry-driven cleaning functions from clinical_processing.py.
4. Saves the cleaned clinical table, missingness table, registry table, and endpoint labels.

Default input
-------------
D:/TongJiPCG/同济心音外部验证/中山一院听诊队列.xlsx

Default output
--------------
Outputs/validation/preprocessing/Data_clinic/

Notes
-----
- Patient ID is the "序号" column in the external spreadsheet.
- Patient name is kept as patient_name for checking and merging with valid_recordings.csv.
- NTproBNP in clinical_clean.csv follows the internal registry behavior, i.e. log1p-transformed
  under the clean name "NTproBNP". The raw numeric value is additionally retained as
  "NTproBNP_raw" for endpoint construction.
- Missing external variables are not forced. The registry-driven processor will fill them as NaN,
  consistent with the internal processing behavior.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Paths and imports from the existing HSCA preprocessing module
# =============================================================================

def _resolve_repo_root() -> Path:
    """Resolve repository root when this file is placed under Scripts/Validation/preprocessing."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "Scripts" / "Preprocessing").exists():
            return parent
    return Path.cwd()


REPO_ROOT = _resolve_repo_root()
PREPROCESSING_DIR = REPO_ROOT / "Scripts" / "Preprocessing"
for p in [REPO_ROOT, PREPROCESSING_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from Scripts.Preprocessing.clinical_registry import build_variable_registry, registry_to_dataframe
    from Scripts.Preprocessing.clinical_processing import (
        build_analysis_ready_clinical_table,
        coerce_numeric_series,
        normalize_patient_id,
    )
except Exception as exc:  # pragma: no cover - import failure should be explicit for users
    raise ImportError(
        "Cannot import clinical_registry.py / clinical_processing.py from Scripts/Preprocessing. "
        "Please place this file under Scripts/Validation/preprocessing inside the HSCA repository."
    ) from exc


# =============================================================================
# Console helpers, matching the style of the existing preprocessing scripts
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
# External-column mapping
# =============================================================================

# Only maps columns needed by the existing registry and later external validation.
# The keys are the raw column names used by clinical_registry.py.
# The values are possible names in the external Zhongshan cohort spreadsheet.
EXTERNAL_TO_REGISTRY_ALIASES: Dict[str, Sequence[str]] = {
    # Patient identifier / demographics
    "编码": ["序号", "编码", "patient_id", "患者编号", "ID"],
    "年龄（岁）": ["基线年龄", "年龄（岁）", "年龄", "age_years"],
    "性别（男=1，女=0）": ["性别  男1", "性别 男1", "性别（男=1，女=0）", "性别", "sex_male"],

    # Functional / HF burden panel
    "NYHA分级": ["NYHA分级(1=Ⅰ,2=Ⅱ,3=Ⅲ,4=Ⅳ)", "NYHA分级", "NYHA", "NYHA_class"],
    "EF值(Teich法）": ["EF值(Teich法）", "EF值(Teich法)", "EF_Teich", "LVEF_Teich"],
    "B型钠尿肽前体(NT-proBNP)（pg/ml, <300)": [
        "NT-proBNP(pg/mL)", "NT-proBNP(pg/ml)", "NT-proBNP", "NTproBNP", "NT_pro_BNP"
    ],
    "心率（次/分）": ["心率(次/分)", "心率（次/分）", "心率", "heart_rate"],

    # Structural remodeling
    "左房（mm)": ["左房（mm)", "左房(mm)", "LA_mm"],
    "左心室（舒张末）(mm)": ["左心室（舒张末）(mm)", "左心室(舒张末)(mm)", "LVEDD", "LVEDD_mm"],
    "室间隔（mm）": ["室间隔（mm）", "室间隔(mm)", "IVS_mm"],
    "左室后壁（mm)": ["左室后壁（mm)", "左室后壁(mm)", "LVPW_mm"],

    # Valve grades. The external spreadsheet may omit '=' before 中度/重度 in the column names.
    "主动脉瓣狭窄（0=无，1=轻度，2=中度，3=重度）": [
        "主动脉瓣狭窄（0=无，1=轻度，2中度，3=重度）",
        "主动脉瓣狭窄（0=无，1=轻度，2=中度，3=重度）",
        "AS_grade",
    ],
    "主动脉瓣关闭不全（0=无，1=轻度，2=中度，3=重度）": [
        "主动脉瓣关闭不全（0=无，1=轻度，2中度，3=重度）",
        "主动脉瓣关闭不全（0=无，1=轻度，2=中度，3=重度）",
        "AR_grade",
    ],
    "二尖瓣狭窄（0=无，1=轻度，2=中度，3=重度）": [
        "二尖瓣狭窄（0=无，1=轻度，2中度，3=重度）",
        "二尖瓣狭窄（0=无，1=轻度，2=中度，3=重度）",
        "MS_grade",
    ],
    "二尖瓣关闭不全（0=无，1=轻度，2=中度，3=重度）": [
        "二尖瓣关闭不全（0=无，1=轻度，2中度，3=重度）",
        "二尖瓣关闭不全（0=无，1=轻度，2=中度，3=重度）",
        "MR_grade",
    ],
    "三尖瓣关闭不全（0=无，1=轻度，2=中度，3=重度）": [
        "三尖瓣关闭不全（0=无，1=轻度，2中度，3=重度）",
        "三尖瓣关闭不全（0=无，1=轻度，2=中度，3=重度）",
        "TR_grade",
    ],
    "肺动脉瓣关闭不全（0=无，1=轻度，2=中度，3=重度）": [
        "肺动脉瓣关闭不全（0=无，1=轻度，2中度，3=重度）",
        "肺动脉瓣关闭不全（0=无，1=轻度，2=中度，3=重度）",
        "PR_grade",
    ],
}

PATIENT_NAME_ALIASES = ["姓名", "患者姓名", "病人姓名", "name", "patient_name"]
NT_PRO_BNP_ALIASES = list(EXTERNAL_TO_REGISTRY_ALIASES["B型钠尿肽前体(NT-proBNP)（pg/ml, <300)"])


# =============================================================================
# Generic helpers
# =============================================================================

def _clean_col_name(col: Any) -> str:
    """Normalize spreadsheet column names for matching only; do not use as output names."""
    text = str(col).replace("\ufeff", "").replace("\u3000", " ").strip()
    text = re.sub(r"\s+", "", text)
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("，", ",").replace("：", ":")
    return text


def _build_col_lookup(columns: Iterable[Any]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for col in columns:
        key = _clean_col_name(col)
        if key not in lookup:
            lookup[key] = str(col)
    return lookup


def _find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Find the first matching column using both exact and whitespace-insensitive matching."""
    for c in candidates:
        if c in df.columns:
            return c
    lookup = _build_col_lookup(df.columns)
    for c in candidates:
        key = _clean_col_name(c)
        if key in lookup:
            return lookup[key]
    return None


def read_external_clinical_table(path: Path) -> pd.DataFrame:
    """Read Excel/CSV clinical table. Uses openpyxl explicitly for xlsx-like files."""
    if not path.exists():
        # Common typo: .xslx instead of .xlsx, and vice versa.
        alternatives = []
        if path.suffix.lower() == ".xlsx":
            alternatives.append(path.with_suffix(".xslx"))
        if path.suffix.lower() == ".xslx":
            alternatives.append(path.with_suffix(".xlsx"))
        for alt in alternatives:
            if alt.exists():
                log_warn(f"Clinical table not found at {path}; using {alt} instead.")
                path = alt
                break
    if not path.exists():
        raise FileNotFoundError(f"External clinical table not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xslx"}:
        return pd.read_excel(path, engine="openpyxl")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported clinical table format: {path.suffix}")


def map_external_columns_to_registry(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create a registry-compatible raw DataFrame.

    Returns
    -------
    mapped_df:
        Raw table with columns renamed/copied to the names expected by clinical_registry.py.
    mapping_df:
        Column-level mapping record.
    patient_map_df:
        patient_id + patient_name mapping for manual checking.
    """
    print_banner("Mapping external clinical columns to internal registry names")

    mapped = raw_df.copy()
    mapped.columns = [str(c).replace("\ufeff", "").strip() for c in mapped.columns]

    mapping_rows: List[Dict[str, Any]] = []
    for registry_raw_name, candidates in EXTERNAL_TO_REGISTRY_ALIASES.items():
        src_col = _find_col(mapped, candidates)
        if src_col is not None:
            mapped[registry_raw_name] = mapped[src_col]
            status = "mapped"
        else:
            mapped[registry_raw_name] = np.nan
            status = "missing"
        mapping_rows.append(
            {
                "registry_raw_name": registry_raw_name,
                "external_source_column": src_col if src_col is not None else "",
                "status": status,
                "candidate_columns": " | ".join(candidates),
            }
        )

    id_col = _find_col(mapped, EXTERNAL_TO_REGISTRY_ALIASES["编码"])
    if id_col is None:
        raise ValueError("Cannot find patient ID column. Expected one of: 序号 / 编码 / patient_id / 患者编号 / ID")

    name_col = _find_col(mapped, PATIENT_NAME_ALIASES)
    patient_map_df = pd.DataFrame(
        {
            "patient_id": normalize_patient_id(mapped[id_col]),
            "patient_name": mapped[name_col].astype("object") if name_col is not None else np.nan,
            "source_id_col": id_col,
            "source_name_col": name_col if name_col is not None else "",
        }
    )
    patient_map_df = patient_map_df.loc[patient_map_df["patient_id"].notna()].drop_duplicates("patient_id", keep="first")

    # Retain raw NT-proBNP separately because the registry transforms NTproBNP with log1p.
    nt_raw_col = _find_col(mapped, NT_PRO_BNP_ALIASES)
    if nt_raw_col is not None:
        mapped["__NTproBNP_raw__"] = coerce_numeric_series(mapped[nt_raw_col])
    else:
        mapped["__NTproBNP_raw__"] = np.nan

    log_done(f"Mapped registry columns: {sum(r['status'] == 'mapped' for r in mapping_rows)} mapped, "
             f"{sum(r['status'] == 'missing' for r in mapping_rows)} missing")
    return mapped, pd.DataFrame(mapping_rows), patient_map_df


def add_patient_name_and_raw_values(
    clinical_clean_df: pd.DataFrame,
    mapped_raw_df: pd.DataFrame,
    patient_map_df: pd.DataFrame,
    patient_id_candidates: Sequence[str],
) -> pd.DataFrame:
    """Add patient_name and raw NTproBNP while keeping registry-cleaned variables unchanged."""
    out = clinical_clean_df.copy()
    if "patient_id" not in out.columns:
        raise ValueError("clinical_clean_df must contain patient_id")

    id_col = _find_col(mapped_raw_df, list(patient_id_candidates))
    if id_col is None:
        id_col = _find_col(mapped_raw_df, EXTERNAL_TO_REGISTRY_ALIASES["编码"])
    if id_col is None:
        return out

    raw_extra = pd.DataFrame(
        {
            "patient_id": normalize_patient_id(mapped_raw_df[id_col]),
            "NTproBNP_raw": mapped_raw_df.get("__NTproBNP_raw__", np.nan),
        }
    )
    raw_extra = raw_extra.loc[raw_extra["patient_id"].notna()].drop_duplicates("patient_id", keep="first")

    name_extra = patient_map_df[["patient_id", "patient_name"]].copy() if "patient_name" in patient_map_df.columns else pd.DataFrame()
    if not name_extra.empty:
        out = out.merge(name_extra, on="patient_id", how="left")
    out = out.merge(raw_extra, on="patient_id", how="left")

    # Put identifiers first for easier manual inspection.
    front = [c for c in ["patient_id", "patient_name"] if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


def build_endpoint_labels(clean_df: pd.DataFrame, lvedd_threshold: float = 55.0) -> pd.DataFrame:
    """Build the four external-validation endpoint labels used later in alignment."""
    out = clean_df[["patient_id"]].copy()

    if "EF_Teich" in clean_df.columns:
        ef = pd.to_numeric(clean_df["EF_Teich"], errors="coerce")
        out["EF_lt_40"] = np.where(ef.notna(), (ef < 40).astype(float), np.nan)
    else:
        out["EF_lt_40"] = np.nan

    if "NYHA" in clean_df.columns:
        nyha = pd.to_numeric(clean_df["NYHA"], errors="coerce")
        out["NYHA_ge_3"] = np.where(nyha.notna(), (nyha >= 3).astype(float), np.nan)
    else:
        out["NYHA_ge_3"] = np.nan

    if "NTproBNP_raw" in clean_df.columns:
        nt_raw = pd.to_numeric(clean_df["NTproBNP_raw"], errors="coerce")
        out["NTproBNP_ge_900"] = np.where(nt_raw.notna(), (nt_raw >= 900).astype(float), np.nan)
    else:
        out["NTproBNP_ge_900"] = np.nan

    if "LVEDD_mm" in clean_df.columns:
        lvedd = pd.to_numeric(clean_df["LVEDD_mm"], errors="coerce")
        out["LVEDD_dilated"] = np.where(lvedd.notna(), (lvedd >= float(lvedd_threshold)).astype(float), np.nan)
    else:
        out["LVEDD_dilated"] = np.nan

    return out


def summarize_clinical_table(clean_df: pd.DataFrame, endpoint_df: pd.DataFrame) -> pd.DataFrame:
    """Small QC summary for quick inspection."""
    rows: List[Dict[str, Any]] = []
    rows.append({"item": "n_patients_clean", "value": int(len(clean_df))})
    for col in ["EF_Teich", "NTproBNP", "NTproBNP_raw", "NYHA", "LVEDD_mm"]:
        if col in clean_df.columns:
            rows.append({"item": f"{col}_non_missing", "value": int(pd.to_numeric(clean_df[col], errors="coerce").notna().sum())})
    for col in ["EF_lt_40", "NYHA_ge_3", "NTproBNP_ge_900", "LVEDD_dilated"]:
        if col in endpoint_df.columns:
            y = pd.to_numeric(endpoint_df[col], errors="coerce")
            rows.append({"item": f"{col}_available", "value": int(y.notna().sum())})
            rows.append({"item": f"{col}_positive", "value": int((y == 1).sum())})
    return pd.DataFrame(rows)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# =============================================================================
# Main runner
# =============================================================================

def main(argv=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run external clinical preprocessing.

    Command line:
        python Scripts/Validation/preprocessing/prepare_external_clinic.py

    PyCharm / Python:
        from Scripts.Validation.preprocessing.prepare_external_clinic import main
        clinical_clean_df, missingness_df = main([])
    """
    parser = argparse.ArgumentParser(description="Prepare external clinical table for validation.")
    parser.add_argument("--clinical-table", type=Path, default=Path(r"D:\TongJiPCG\同济心音外部验证\中山一院听诊队列.xlsx"))
    parser.add_argument("--window-meta", type=Path, default=Path(r"D:\PycharmProjects\HSCA\Outputs\validation\preprocessing\Data_windows\windows_4_5_4_1\window_index.csv"))
    parser.add_argument("--outdir", type=Path, default=Path(r"D:\PycharmProjects\HSCA\Outputs\validation\preprocessing\Data_clinic"))
    parser.add_argument("--patient-id-candidates", nargs="+", default=["编码", "序号", "patient_id", "患者编号", "ID"])
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--lvedd-threshold", type=float, default=55.0)
    parser.add_argument("--no-window-meta", action="store_true")
    args = parser.parse_args(argv)

    print_banner("External clinical preprocessing runner")
    log_info(f"clinical_table : {args.clinical_table}")
    log_info(f"window_meta    : {args.window_meta}")
    log_info(f"outdir         : {args.outdir}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    raw_df = read_external_clinical_table(args.clinical_table)
    raw_df.columns = [str(c).replace("\ufeff", "").strip() for c in raw_df.columns]
    log_done(f"Loaded external clinical table | shape={raw_df.shape}")

    mapped_df, mapping_df, patient_map_df = map_external_columns_to_registry(raw_df)

    registry = build_variable_registry()
    use_window_meta = (not args.no_window_meta) and args.window_meta.exists()
    if args.no_window_meta:
        log_warn("Window metadata disabled by --no-window-meta.")
    elif not args.window_meta.exists():
        log_warn(f"Window metadata not found. Technical covariates will be skipped: {args.window_meta}")

    clinical_clean_df, missingness_df = build_analysis_ready_clinical_table(
        patient_info_df=mapped_df,
        registry=registry,
        patient_id_candidates=args.patient_id_candidates,
        window_meta_csv=args.window_meta if use_window_meta else None,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    )

    clinical_clean_df = add_patient_name_and_raw_values(
        clinical_clean_df=clinical_clean_df,
        mapped_raw_df=mapped_df,
        patient_map_df=patient_map_df,
        patient_id_candidates=args.patient_id_candidates,
    )
    endpoint_df = build_endpoint_labels(clinical_clean_df, lvedd_threshold=args.lvedd_threshold)
    qc_summary_df = summarize_clinical_table(clinical_clean_df, endpoint_df)
    registry_df = registry_to_dataframe(registry)

    clinical_clean_path = args.outdir / "clinical_clean.csv"
    missingness_path = args.outdir / "clinical_missingness.csv"
    registry_path = args.outdir / "clinical_variable_registry.csv"
    mapped_raw_path = args.outdir / "external_clinical_raw_mapped.csv"
    mapping_path = args.outdir / "external_column_mapping.csv"
    patient_map_path = args.outdir / "external_patient_id_name_map.csv"
    endpoint_path = args.outdir / "external_endpoint_labels.csv"
    qc_path = args.outdir / "external_clinical_qc_summary.csv"
    config_path = args.outdir / "external_clinical_preprocessing_config.json"

    clinical_clean_df.to_csv(clinical_clean_path, index=False, encoding="utf-8-sig")
    missingness_df.to_csv(missingness_path, index=False, encoding="utf-8-sig")
    registry_df.to_csv(registry_path, index=False, encoding="utf-8-sig")
    mapped_df.to_csv(mapped_raw_path, index=False, encoding="utf-8-sig")
    mapping_df.to_csv(mapping_path, index=False, encoding="utf-8-sig")
    patient_map_df.to_csv(patient_map_path, index=False, encoding="utf-8-sig")
    endpoint_df.to_csv(endpoint_path, index=False, encoding="utf-8-sig")
    qc_summary_df.to_csv(qc_path, index=False, encoding="utf-8-sig")
    save_json(
        {
            "clinical_table": str(args.clinical_table),
            "window_meta": str(args.window_meta),
            "use_window_meta": bool(use_window_meta),
            "outdir": str(args.outdir),
            "patient_id_candidates": list(args.patient_id_candidates),
            "window_seconds": float(args.window_seconds),
            "stride_seconds": float(args.stride_seconds),
            "lvedd_threshold": float(args.lvedd_threshold),
        },
        config_path,
    )

    log_done(f"Saved clinical clean table       : {clinical_clean_path}")
    log_done(f"Saved missingness table          : {missingness_path}")
    log_done(f"Saved clinical variable registry : {registry_path}")
    log_done(f"Saved endpoint labels            : {endpoint_path}")
    log_done(f"Saved column mapping             : {mapping_path}")
    log_done(f"Saved QC summary                 : {qc_path}")

    return clinical_clean_df, missingness_df


if __name__ == "__main__":
    main()
