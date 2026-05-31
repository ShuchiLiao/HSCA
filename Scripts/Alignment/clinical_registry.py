
"""Clinical variable registry for distance-based clinical alignment.

This module defines the clinical variables used in:
1. Clinical distance-matrix construction.
2. Neighborhood consistency analysis.
3. Retrieval-based clinical validation.
4. Supplementary robustness analyses:
   - confounder adjustment
   - baseline comparison
   - discovery / validation
   - retrieval-confidence analysis

Design notes
------------
- Variables are declared once here, then reused by downstream processing.
- Each variable explicitly states whether it participates in distance,
  neighborhood, retrieval, and adjustment analyses.
- The registry remains concise and focused on clinically interpretable
  variables for the current manuscript.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence
import pandas as pd


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
# Registry dataclass
# =============================================================================


@dataclass
class VariableSpec:
    """One clinical variable definition.

    Parameters
    ----------
    raw_name:
        Original column name in the clinical spreadsheet.
    clean_name:
        Internal standardized name used in code.
    group:
        High-level clinical group used in the paper, e.g. function / structure / burden.
    var_type:
        continuous / binary / ordinal / categorical / text / exclude.
    transform:
        Cleaning / transform route, e.g. none / log1p / nyha_map / binary_map.
    use_for_distance:
        Whether the variable participates in clinical distance-matrix construction.
    use_for_neighbor:
        Whether the variable participates in neighborhood consistency analysis.
    use_for_retrieval:
        Whether the variable can be used as a retrieval validation target.
    use_for_adjustment:
        Whether the variable can be used as a covariate in confounder adjustment.
    weight:
        Variable weight in the clinical distance matrix.
    notes:
        Human-readable note for later reporting.
    """

    raw_name: str
    clean_name: str
    group: str
    var_type: str
    transform: str
    use_for_distance: bool
    use_for_neighbor: bool
    use_for_retrieval: bool
    use_for_adjustment: bool
    weight: float = 1.0
    notes: str = ""


# =============================================================================
# Registry builder
# =============================================================================


def build_variable_registry() -> List[VariableSpec]:
    """Build the clinical variable registry for the manuscript."""
    reg: List[VariableSpec] = []

    # -------------------------------------------------------------------------
    # Demographic / adjustment variables
    # -------------------------------------------------------------------------
    reg.extend(
        [
            VariableSpec("年龄（岁）", "age_years", "demographic", "continuous", "none", False, False, True, True, 1.0, "Age in years."),
            VariableSpec("性别（男=1，女=0）", "sex_male", "demographic", "binary", "binary_map", False, False, True, True, 1.0, "1=male, 0=female."),
        ]
    )

    # -------------------------------------------------------------------------
    # Function group
    # -------------------------------------------------------------------------
    reg.extend(
        [
            VariableSpec("NYHA分级", "NYHA", "function", "ordinal", "nyha_map", True, True, True, False, 1.0, "NYHA class."),
            VariableSpec("EF值(Teich法）", "EF_Teich", "function", "continuous", "none", True, True, True, False, 1.0, "Teichholz ejection fraction."),
            VariableSpec("B型钠尿肽前体(NT-proBNP)（pg/ml, <300)", "NTproBNP", "function", "continuous", "log1p", True, True, True, False, 1.0, "NT-proBNP."),
            VariableSpec("高敏肌钙蛋白T (TnT-T)（ng/ml, 0-0.014)", "hsTnT", "function", "continuous", "log1p", True, True, True, False, 1.0, "Supportive myocardial injury marker."),
            VariableSpec("呼吸困难/气促（1=是，0=否）", "symptom_dyspnea", "function", "binary", "binary_map", True, True, True, False, 1.0, "Dyspnea symptom."),
            VariableSpec("心率（次/分）", "heart_rate", "function", "continuous", "none", True, True, True, False, 1.0, "Heart rate."),
        ]
    )

    # -------------------------------------------------------------------------
    # Structure group
    # -------------------------------------------------------------------------
    reg.extend(
        [
            VariableSpec("左房（mm)", "LA_mm", "structure", "continuous", "none", True, True, True, False, 1.0, "Left atrial diameter."),
            VariableSpec("左心室（舒张末）(mm)", "LVEDD_mm", "structure", "continuous", "none", True, True, True, False, 1.0, "LVEDD."),
            VariableSpec("室间隔（mm）", "IVS_mm", "structure", "continuous", "none", True, True, True, False, 1.0, "IVS thickness."),
            VariableSpec("左室后壁（mm)", "LVPW_mm", "structure", "continuous", "none", True, True, True, False, 1.0, "LVPW thickness."),
            VariableSpec("主动脉瓣关闭不全（0=无，1=轻度，2=中度，3=重度）", "AR_grade", "structure", "ordinal", "ordinal_map", True, True, True, False, 1.0, "Aortic regurgitation."),
            VariableSpec("主动脉瓣狭窄（0=无，1=轻度，2=中度，3=重度）", "AS_grade", "structure", "ordinal", "ordinal_map", True, True, True, False, 1.0, "Aortic stenosis."),
            VariableSpec("二尖瓣狭窄（0=无，1=轻度，2=中度，3=重度）", "MS_grade", "structure", "ordinal", "ordinal_map", True, True, True, False, 1.0, "Mitral stenosis."),
            VariableSpec("二尖瓣关闭不全（0=无，1=轻度，2=中度，3=重度）", "MR_grade", "structure", "ordinal", "ordinal_map", True, True, True, False, 1.0, "Mitral regurgitation."),
            VariableSpec("三尖瓣关闭不全（0=无，1=轻度，2=中度，3=重度）", "TR_grade", "structure", "ordinal", "ordinal_map", True, True, True, False, 1.0, "Tricuspid regurgitation."),
            VariableSpec("肺动脉瓣关闭不全（0=无，1=轻度，2=中度，3=重度）", "PR_grade", "structure", "ordinal", "ordinal_map", True, True, True, False, 1.0, "Pulmonary regurgitation."),
        ]
    )

    # -------------------------------------------------------------------------
    # Burden group
    # -------------------------------------------------------------------------
    reg.extend(
        [
            VariableSpec("胸闷/痛（1=是，0=否）", "symptom_chest_discomfort", "burden", "binary", "binary_map", True, True, True, False, 1.0, "Chest discomfort."),
            VariableSpec("心慌/悸（1=是，0=否）", "symptom_palpitations", "burden", "binary", "binary_map", True, True, True, False, 1.0, "Palpitations."),
            VariableSpec("乏力（1=是，0=否）", "symptom_fatigue", "burden", "binary", "binary_map", True, True, True, False, 1.0, "Fatigue."),
            VariableSpec("头晕（1=是，0=否）", "symptom_dizziness", "burden", "binary", "binary_map", True, True, True, False, 1.0, "Dizziness."),
            VariableSpec("黑曚/晕厥（1=是，0=否）", "symptom_syncope", "burden", "binary", "binary_map", True, True, True, False, 1.0, "Syncope."),
            VariableSpec("CRP (mg/l, 0-10)", "CRP", "burden", "continuous", "log1p", True, True, True, False, 1.0, "Inflammatory burden."),
            VariableSpec("D-二聚体 (mg/l FEU, 0-0.55)", "D_dimer", "burden", "continuous", "log1p", True, True, True, False, 1.0, "Thrombotic burden."),
            VariableSpec("血红蛋白Hb (g/l, 130-175)", "Hb", "burden", "continuous", "none", True, True, True, False, 1.0, "Anemia-related burden."),
        ]
    )

    # -------------------------------------------------------------------------
    # Technical covariates constructed from window metadata
    # -------------------------------------------------------------------------
    reg.extend(
        [
            VariableSpec("__technical__", "n_windows_total", "technical", "continuous", "none", False, False, False, True, 1.0, "Total number of windows."),
            VariableSpec("__technical__", "duration_total", "technical", "continuous", "none", False, False, False, True, 1.0, "Total recording duration proxy."),
        ]
    )

    # -------------------------------------------------------------------------
    # Excluded / text-only variables retained for completeness
    # -------------------------------------------------------------------------
    reg.extend(
        [
            VariableSpec("编码", "patient_id_raw", "exclude", "exclude", "none", False, False, False, False, 0.0, "Primary identifier, not analyzed directly."),
            VariableSpec("其他症状", "other_symptoms_text", "exclude", "text", "text_group", False, False, False, False, 0.0, "Free-text symptom description."),
            VariableSpec("入院诊断", "admission_diagnosis_text", "exclude", "text", "text_group", False, False, False, False, 0.0, "Admission diagnosis text."),
            VariableSpec("出院诊断", "discharge_diagnosis_text", "exclude", "text", "text_group", False, False, False, False, 0.0, "Discharge diagnosis text."),
        ]
    )
    return reg


# =============================================================================
# Convenience helpers
# =============================================================================


def registry_to_dataframe(registry: Sequence[VariableSpec]) -> pd.DataFrame:
    """Convert the registry to a DataFrame for saving and inspection."""
    return pd.DataFrame([asdict(v) for v in registry])


def get_distance_variables(registry: Sequence[VariableSpec], group: str = "all") -> List[VariableSpec]:
    """Return variables used to build clinical distance matrices."""
    if group == "all":
        allowed = {"function", "structure", "burden"}
        return [v for v in registry if v.use_for_distance and v.group in allowed]
    return [v for v in registry if v.use_for_distance and v.group == group]


def get_neighbor_variables(registry: Sequence[VariableSpec], group: str = "all") -> List[VariableSpec]:
    """Return variables used in neighborhood consistency analysis."""
    if group == "all":
        allowed = {"function", "structure", "burden"}
        return [v for v in registry if v.use_for_neighbor and v.group in allowed]
    return [v for v in registry if v.use_for_neighbor and v.group == group]


def get_retrieval_targets(registry: Sequence[VariableSpec], group: str = "all") -> List[VariableSpec]:
    """Return variables eligible for retrieval-based validation."""
    if group == "all":
        allowed = {"function", "structure", "burden"}
        return [v for v in registry if v.use_for_retrieval and v.group in allowed]
    return [v for v in registry if v.use_for_retrieval and v.group == group]


def get_adjustment_variables(registry: Sequence[VariableSpec]) -> List[VariableSpec]:
    """Return variables that may be used as confounder-adjustment covariates."""
    return [v for v in registry if v.use_for_adjustment]


__all__ = [
    "VariableSpec",
    "build_variable_registry",
    "registry_to_dataframe",
    "get_distance_variables",
    "get_neighbor_variables",
    "get_retrieval_targets",
    "get_adjustment_variables",
]
