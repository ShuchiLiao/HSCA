from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover
    from .config import PipelineConfig, PlotConfig, build_default_config, log_done, log_info, log_warn, print_banner
    from .clinical_registry import build_variable_registry, VariableSpec
    from .core import ClinicalAlignmentData
    from .analysis import run_position_leave_one_out_contribution
except ImportError:  # pragma: no cover
    from config import PipelineConfig, PlotConfig, build_default_config, log_done, log_info, log_warn, print_banner
    from clinical_registry import build_variable_registry, VariableSpec
    from core import ClinicalAlignmentData
    from analysis import run_position_leave_one_out_contribution


# =============================================================================
# helpers
# =============================================================================


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_csv(path: str | Path, desc: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")
    return pd.read_csv(path)


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# =============================================================================
# Variable-level statistical consolidation
# =============================================================================


def build_variablewise_consolidation_tables(
    global_alignment_dir: str | Path,
    neighborhood_dir: str | Path,
    out_dir: str | Path,
    alpha: float = 0.05,
    top_n: int = 12,
) -> Dict[str, pd.DataFrame]:
    """Create concise variable-level main tables from existing outputs."""
    print_banner("Building variable-level statistical consolidation tables")
    global_alignment_dir = Path(global_alignment_dir)
    neighborhood_dir = Path(neighborhood_dir)
    out_dir = _ensure_dir(out_dir)

    var_global = _load_csv(global_alignment_dir / "variablewise_global_alignment_summary.csv", "variablewise global alignment summary")
    required_global = [
        "variable", "group", "var_type", "coverage", "spearman_r", "perm_p_spearman",
        "perm_q_spearman_overall", "perm_q_spearman_within_group",
    ]
    missing = [c for c in required_global if c not in var_global.columns]
    if missing:
        raise ValueError(f"variablewise_global_alignment_summary.csv missing columns: {missing}")

    global_main = (
        var_global.loc[:, required_global]
        .rename(columns={
            "perm_p_spearman": "global_perm_p_spearman",
            "perm_q_spearman_overall": "global_perm_q_spearman_overall",
            "perm_q_spearman_within_group": "global_perm_q_spearman_within_group",
        })
        .copy()
    )
    global_main["global_fdr_sig_overall"] = global_main["global_perm_q_spearman_overall"] < float(alpha)
    global_main["global_fdr_sig_within_group"] = global_main["global_perm_q_spearman_within_group"] < float(alpha)
    global_main = global_main.sort_values(["spearman_r", "coverage"], ascending=[False, False]).reset_index(drop=True)
    global_main.to_csv(out_dir / "variablewise_global_main_table.csv", index=False, encoding="utf-8-sig")
    global_main.head(int(top_n)).to_csv(out_dir / "variablewise_global_top_table.csv", index=False, encoding="utf-8-sig")

    nb_var = _load_csv(neighborhood_dir / "neighborhood_variable_summary.csv", "neighborhood variable summary")
    required_nb = [
        "variable", "group", "var_type", "k", "improvement_ratio", "perm_p",
        "perm_q_overall_all_k", "perm_q_overall_within_k", "perm_q_within_group_within_k",
    ]
    missing = [c for c in required_nb if c not in nb_var.columns]
    if missing:
        raise ValueError(f"neighborhood_variable_summary.csv missing columns: {missing}")

    nb_best = nb_var.sort_values(["variable", "improvement_ratio"], ascending=[True, False]).drop_duplicates(subset=["variable"], keep="first")
    nb_best = nb_best.loc[:, required_nb].rename(columns={
        "perm_p": "neighborhood_perm_p",
        "perm_q_overall_all_k": "neighborhood_perm_q_overall_all_k",
        "perm_q_overall_within_k": "neighborhood_perm_q_overall_within_k",
        "perm_q_within_group_within_k": "neighborhood_perm_q_within_group_within_k",
        "k": "best_k",
    }).copy()
    nb_best["neighborhood_fdr_sig_within_k"] = nb_best["neighborhood_perm_q_overall_within_k"] < float(alpha)
    nb_best = nb_best.sort_values(["improvement_ratio"], ascending=False).reset_index(drop=True)
    nb_best.to_csv(out_dir / "variablewise_neighborhood_bestk_table.csv", index=False, encoding="utf-8-sig")
    nb_best.head(max(int(top_n), 18)).to_csv(out_dir / "variablewise_neighborhood_top_table.csv", index=False, encoding="utf-8-sig")

    merged = global_main.merge(
        nb_best[[
            "variable", "best_k", "improvement_ratio", "neighborhood_perm_p",
            "neighborhood_perm_q_overall_all_k", "neighborhood_perm_q_overall_within_k",
            "neighborhood_perm_q_within_group_within_k", "neighborhood_fdr_sig_within_k",
        ]],
        on="variable",
        how="left",
    )
    merged = merged.sort_values(["spearman_r", "improvement_ratio"], ascending=[False, False]).reset_index(drop=True)
    merged.to_csv(out_dir / "variablewise_consolidated_main_table.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "alpha": float(alpha),
        "n_global_variables": int(len(global_main)),
        "n_neighborhood_variables": int(len(nb_best)),
        "n_global_fdr_sig_overall": int(global_main["global_fdr_sig_overall"].sum()),
        "n_neighborhood_fdr_sig_within_k": int(nb_best["neighborhood_fdr_sig_within_k"].sum()),
    }
    with (out_dir / "variablewise_consolidation_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log_done(f"Saved variable-level consolidation tables to: {out_dir}")
    return {
        "global_main": global_main,
        "neighborhood_best": nb_best,
        "consolidated": merged,
    }


# =============================================================================
# Retrieval case explanations
# =============================================================================


def _choose_targets_from_retrieval_summary(summary_df: pd.DataFrame, n_cont: int = 6, n_cat: int = 4) -> List[Tuple[str, int, str]]:
    picks: List[Tuple[str, int, str]] = []
    cont = summary_df.loc[summary_df["target_type"].isin(["continuous", "ordinal"]) & summary_df["spearman"].notna()].copy()
    if len(cont) > 0:
        best_cont = cont.sort_values(["target", "spearman"], ascending=[True, False]).drop_duplicates(subset=["target"], keep="first")
        best_cont = best_cont.sort_values("spearman", ascending=False).head(int(n_cont))
        picks.extend([(str(r["target"]), int(r["k"]), str(r["target_type"])) for _, r in best_cont.iterrows()])

    cat = summary_df.loc[summary_df["target_type"].isin(["binary", "categorical"]) & summary_df["balanced_accuracy"].notna()].copy()
    if len(cat) > 0:
        best_cat = cat.sort_values(["target", "balanced_accuracy"], ascending=[True, False]).drop_duplicates(subset=["target"], keep="first")
        best_cat = best_cat.sort_values("balanced_accuracy", ascending=False).head(int(n_cat))
        picks.extend([(str(r["target"]), int(r["k"]), str(r["target_type"])) for _, r in best_cat.iterrows()])
    return picks


def build_retrieval_case_explanations(
    retrieval_dir: str | Path,
    out_dir: str | Path,
    n_success: int = 3,
    n_failure: int = 3,
    n_cont_targets: int = 6,
    n_cat_targets: int = 4,
    max_neighbor_rank: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Select representative success/failure cases for retrieval-based explanation."""
    print_banner("Building retrieval case explanations")
    retrieval_dir = Path(retrieval_dir)
    out_dir = _ensure_dir(out_dir)

    summary_df = _load_csv(retrieval_dir / "retrieval_summary.csv", "retrieval summary")
    pred_df = _load_csv(retrieval_dir / "retrieval_predictions.csv", "retrieval predictions")
    neighbor_df = _load_csv(retrieval_dir / "retrieval_neighbor_details.csv", "retrieval neighbor details")

    picks = _choose_targets_from_retrieval_summary(summary_df, n_cont=n_cont_targets, n_cat=n_cat_targets)
    if len(picks) == 0:
        raise ValueError("No suitable retrieval targets found for case explanation.")

    case_rows: List[Dict[str, Any]] = []
    selected_keys: List[Tuple[str, str, int]] = []

    for target, best_k, target_type in picks:
        sub = pred_df.loc[(pred_df["target"] == target) & (pred_df["k"] == int(best_k))].copy()
        if len(sub) == 0:
            continue
        sub["true_num"] = _safe_numeric(sub["true_value"])
        sub["pred_num"] = _safe_numeric(sub["predicted_value"])
        if target_type in {"continuous", "ordinal"}:
            sub = sub.loc[sub["true_num"].notna() & sub["pred_num"].notna()].copy()
            if len(sub) == 0:
                continue
            sub["abs_error"] = (sub["true_num"] - sub["pred_num"]).abs()
            success = sub.sort_values(["abs_error", "mean_topk_distance"], ascending=[True, True]).head(int(n_success)).copy()
            failure = sub.sort_values(["abs_error", "mean_topk_distance"], ascending=[False, True]).head(int(n_failure)).copy()
            for rank, (_, row) in enumerate(success.iterrows(), start=1):
                case_rows.append({
                    "target": target,
                    "target_type": target_type,
                    "k": int(best_k),
                    "case_type": "success",
                    "case_rank": int(rank),
                    "patient_id": row["patient_id"],
                    "true_value": row["true_value"],
                    "predicted_value": row["predicted_value"],
                    "predicted_probability": row.get("predicted_probability", np.nan),
                    "top1_neighbor_id": row.get("top1_neighbor_id", np.nan),
                    "mean_topk_distance": row.get("mean_topk_distance", np.nan),
                    "abs_error": row["abs_error"],
                    "is_correct": np.nan,
                })
                selected_keys.append((str(row["patient_id"]), target, int(best_k)))
            for rank, (_, row) in enumerate(failure.iterrows(), start=1):
                case_rows.append({
                    "target": target,
                    "target_type": target_type,
                    "k": int(best_k),
                    "case_type": "failure",
                    "case_rank": int(rank),
                    "patient_id": row["patient_id"],
                    "true_value": row["true_value"],
                    "predicted_value": row["predicted_value"],
                    "predicted_probability": row.get("predicted_probability", np.nan),
                    "top1_neighbor_id": row.get("top1_neighbor_id", np.nan),
                    "mean_topk_distance": row.get("mean_topk_distance", np.nan),
                    "abs_error": row["abs_error"],
                    "is_correct": np.nan,
                })
                selected_keys.append((str(row["patient_id"]), target, int(best_k)))
        else:
            sub = sub.loc[sub["true_value"].notna() & sub["predicted_value"].notna()].copy()
            if len(sub) == 0:
                continue
            sub["is_correct"] = sub["true_value"].astype(str) == sub["predicted_value"].astype(str)
            success = sub.loc[sub["is_correct"]].sort_values(["mean_topk_distance"], ascending=[True]).head(int(n_success)).copy()
            failure = sub.loc[~sub["is_correct"]].sort_values(["mean_topk_distance"], ascending=[True]).head(int(n_failure)).copy()
            for rank, (_, row) in enumerate(success.iterrows(), start=1):
                case_rows.append({
                    "target": target,
                    "target_type": target_type,
                    "k": int(best_k),
                    "case_type": "success",
                    "case_rank": int(rank),
                    "patient_id": row["patient_id"],
                    "true_value": row["true_value"],
                    "predicted_value": row["predicted_value"],
                    "predicted_probability": row.get("predicted_probability", np.nan),
                    "top1_neighbor_id": row.get("top1_neighbor_id", np.nan),
                    "mean_topk_distance": row.get("mean_topk_distance", np.nan),
                    "abs_error": np.nan,
                    "is_correct": True,
                })
                selected_keys.append((str(row["patient_id"]), target, int(best_k)))
            for rank, (_, row) in enumerate(failure.iterrows(), start=1):
                case_rows.append({
                    "target": target,
                    "target_type": target_type,
                    "k": int(best_k),
                    "case_type": "failure",
                    "case_rank": int(rank),
                    "patient_id": row["patient_id"],
                    "true_value": row["true_value"],
                    "predicted_value": row["predicted_value"],
                    "predicted_probability": row.get("predicted_probability", np.nan),
                    "top1_neighbor_id": row.get("top1_neighbor_id", np.nan),
                    "mean_topk_distance": row.get("mean_topk_distance", np.nan),
                    "abs_error": np.nan,
                    "is_correct": False,
                })
                selected_keys.append((str(row["patient_id"]), target, int(best_k)))

    case_df = pd.DataFrame(case_rows)
    if len(case_df) == 0:
        raise ValueError("No retrieval case examples could be selected.")

    key_df = pd.DataFrame(selected_keys, columns=["query_patient_id", "target", "k"]).drop_duplicates()
    neighbor_keep = neighbor_df.merge(key_df, on=["query_patient_id", "target", "k"], how="inner")
    if max_neighbor_rank is not None and "neighbor_rank" in neighbor_keep.columns:
        neighbor_keep = neighbor_keep.loc[neighbor_keep["neighbor_rank"] <= int(max_neighbor_rank)].copy()

    case_df.to_csv(out_dir / "retrieval_case_examples_summary.csv", index=False, encoding="utf-8-sig")
    neighbor_keep.to_csv(out_dir / "retrieval_case_examples_neighbors.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "n_targets_selected": int(case_df[["target", "k"]].drop_duplicates().shape[0]),
        "n_cases": int(len(case_df)),
        "n_neighbor_rows": int(len(neighbor_keep)),
    }
    with (out_dir / "retrieval_case_examples_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log_done(f"Saved retrieval case explanations to: {out_dir}")
    return {"cases": case_df, "neighbors": neighbor_keep}


# =============================================================================
# Orchestration
# =============================================================================


def run_interpretability_suite(
    data: ClinicalAlignmentData,
    registry: Sequence[VariableSpec],
    cfg: PipelineConfig,
    out_root: str | Path,
) -> Dict[str, Any]:
    """Run interpretability / statistical consolidation analyses."""
    print_banner("Running interpretability / statistical consolidation suite")
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    dirs = {
        "root": out_root,
        "variablewise": _ensure_dir(out_root / "variablewise_consolidation"),
        "position_contribution": _ensure_dir(out_root / "position_contribution"),
        "retrieval_cases": _ensure_dir(out_root / "retrieval_case_examples"),
    }

    tables = build_variablewise_consolidation_tables(
        global_alignment_dir=Path(cfg.paths.output_root) / "global_alignment",
        neighborhood_dir=Path(cfg.paths.output_root) / "neighborhood",
        out_dir=dirs["variablewise"],
        alpha=float(cfg.analysis.fdr_alpha),
        top_n=int(cfg.analysis.variablewise_top_n),
    )

    pos_global, pos_nb, pos_rt = run_position_leave_one_out_contribution(
        data=data,
        registry=registry,
        position_distance_dir=cfg.paths.position_distance_dir,
        positions=cfg.analysis.positions,
        groups=cfg.analysis.clinical_groups,
        knn_list=cfg.analysis.knn_list,
        retrieval_kernel=cfg.analysis.retrieval_kernel,
        retrieval_sigma=cfg.analysis.retrieval_sigma,
        out_dir=dirs["position_contribution"],
        plot_cfg=cfg.plot,
        n_perm=0,
        n_jobs=cfg.analysis.n_jobs,
        random_seed=cfg.analysis.random_seed,
        file_pattern=cfg.analysis.position_distance_pattern,
        aggregation=cfg.analysis.position_aggregation,
        weights=cfg.analysis.position_weights,
        reference_mode=cfg.analysis.position_contribution_reference_mode,
    )

    cases = build_retrieval_case_explanations(
        retrieval_dir=Path(cfg.paths.output_root) / "retrieval",
        out_dir=dirs["retrieval_cases"],
        n_success=3,
        n_failure=3,
        n_cont_targets=int(cfg.analysis.retrieval_top_n_continuous),
        n_cat_targets=min(int(cfg.analysis.retrieval_top_n_categorical), 4),
        max_neighbor_rank=max(cfg.analysis.knn_list) if len(cfg.analysis.knn_list) > 0 else None,
    )

    manifest = {
        "variablewise_rows": int(len(tables["consolidated"])),
        "position_global_rows": int(len(pos_global)),
        "position_neighborhood_rows": int(len(pos_nb)),
        "position_retrieval_rows": int(len(pos_rt)),
        "retrieval_case_rows": int(len(cases["cases"])),
        "retrieval_case_neighbor_rows": int(len(cases["neighbors"])),
    }
    with (dirs["root"] / "interpretability_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log_done(f"Interpretability suite finished | outputs saved under: {dirs['root']}")
    return {
        "dirs": dirs,
        "variablewise": tables,
        "position_global": pos_global,
        "position_neighborhood": pos_nb,
        "position_retrieval": pos_rt,
        "retrieval_cases": cases,
    }


# =============================================================================
# CLI
# =============================================================================


def _build_cli() -> Any:
    import argparse
    parser = argparse.ArgumentParser(description="Interpretability and statistical consolidation utilities for clinical alignment.")
    parser.add_argument("--prepared-dir", type=str, required=True)
    parser.add_argument("--out-root", type=str, default="clinical_alignment/outputs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--position-distance-dir", type=str, default=None)
    return parser


def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()
    cfg = build_default_config()
    cfg.paths.output_root = str(args.out_root)
    if args.position_distance_dir:
        cfg.paths.position_distance_dir = str(args.position_distance_dir)
    if args.seed is not None:
        cfg.analysis.random_seed = int(args.seed)
    data = ClinicalAlignmentData.load(args.prepared_dir)
    registry = build_variable_registry()
    run_interpretability_suite(data=data, registry=registry, cfg=cfg, out_root=Path(cfg.paths.output_root) / "interpretability")


if __name__ == "__main__":  # pragma: no cover
    main()
