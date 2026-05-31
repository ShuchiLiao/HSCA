"""Configuration utilities for clinical alignment.

This module centralizes:
1. Core input / output paths.
2. Main-analysis defaults for global alignment, neighborhood consistency,
   and retrieval-based validation.
3. Robustness / supplementary-analysis defaults:
   - confounder adjustment
   - patient-level bootstrap / subsampling confidence intervals
   - discovery / validation splits
   - baseline-space comparison
4. Publication-ready plotting style shared across the package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns


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
# Dataclass configuration
# =============================================================================


@dataclass
class PathConfig:
    acoustic_distance_npy: str = "Phenotype_discovery/outputs/distance/patient_distance_matrix.npy"
    patient_order_csv: str = "Phenotype_discovery/outputs/distance/patient_order.csv"
    clinical_table_path: str = "Data/patient_info.xlsx"
    window_meta_csv: str = "Representation_learning/embeddings_4_1/beats/window_meta.csv"

    beats_patient_embedding_npy: str = "Representation_learning/embeddings_4_1/beats/patient_embeddings.npy"
    beats_patient_embedding_meta_csv: str = "Representation_learning/embeddings_4_1/beats/patient_meta.csv"
    beats_position_embedding_npy: str = "Representation_learning/embeddings_4_1/beats/position_embeddings.npy"
    beats_position_embedding_meta_csv: str = "Representation_learning/embeddings_4_1/beats/position_meta.csv"
    ead_patient_embedding_npy: str = "Representation_learning/embeddings_4_1/ead/patient_embeddings.npy"
    ead_patient_embedding_meta_csv: str = "Representation_learning/embeddings_4_1/ead/patient_meta.csv"

    ead_pointcloud_distance_npy: str = "Phenotype_discovery/outputs_ead/distance/patient_distance_matrix.npy"
    ead_pointcloud_patient_order_csv: str = "Phenotype_discovery/outputs_ead/distance/patient_order.csv"

    output_root: str = "clinical_alignment/outputs"
    position_distance_dir: str = "Phenotype_discovery/outputs/distance"


@dataclass
class AnalysisConfig:
    patient_id_candidates: Sequence[str] = field(default_factory=lambda: ("patient_id", "编码", "PID", "pid"))
    clinical_groups: List[str] = field(default_factory=lambda: ["function", "structure", "burden", "all"])

    global_permutations: int = 100
    variablewise_top_n: int = 12

    knn_list: List[int] = field(default_factory=lambda: [5, 10, 20])
    random_baseline_repeats: int = 100

    retrieval_kernel: str = "rbf"
    retrieval_sigma: str = "median"
    retrieval_top_n_continuous: int = 12
    retrieval_top_n_categorical: int = 8

    adjust_covariates: Tuple[str, ...] = ("age_years", "sex_male")
    technical_covariates: Tuple[str, ...] = ("n_windows_total", "duration_total")

    bootstrap_repeats: int = 100
    bootstrap_ci: float = 0.95
    bootstrap_subsample_fraction: float = 0.8

    discovery_fraction: float = 0.70
    split_repeats: int = 10

    baseline_spaces: Tuple[str, ...] = ("pointcloud_ot", "beats_meanpool", "ead_pointcloud_ot", "ead_meanpool")

    # Statistical consolidation / interpretability helpers.
    fdr_alpha: float = 0.05
    fdr_method: str = "bh"
    positions: Tuple[str, ...] = ("A", "E", "M", "P", "T")
    position_distance_pattern: str = "position_distance_{position}.npy"
    position_aggregation: str = "mean"
    position_weights: Dict[str, float] = field(default_factory=dict)
    position_contribution_reference_mode: str = "aggregate_all_positions"
    save_retrieval_neighbor_details: bool = True

    random_seed: int = 42
    n_jobs: int = min(4, os.cpu_count() or 1)


@dataclass
class PlotConfig:
    dpi: int = 300
    font_family: str = "Arial"
    base_font_size: int = 20
    title_font_size: int = 24
    label_font_size: int = 20
    tick_font_size: int = 18
    legend_font_size: int = 16
    context: str = "talk"
    style: str = "white"
    palette: List[str] = field(
        default_factory=lambda: [
            "#158F8C",
            "#2F6DB3",
            "#E38B2C",
            "#7B5AA6",
            "#5FB0B7",
            "#F1B44C",
            "#C75C5C",
        ]
    )
    axis_grid_alpha: float = 0.22
    axis_grid_linewidth: float = 0.9
    heatmap_grid_linewidth: float = 0.8
    heatmap_grid_color: str = "#F3F3F3"
    heatmap_annot: bool = True
    global_scatter_max_points: int = 60000
    neighborhood_heatmap_top_n: int = 18
    bootstrap_retrieval_top_n: int = 12
    retrieval_heatmap_top_n_continuous: int = 12
    retrieval_heatmap_top_n_categorical: int = 8


@dataclass
class PipelineConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)


# =============================================================================
# Config helpers
# =============================================================================


def build_default_config() -> PipelineConfig:
    return PipelineConfig()


def config_to_dict(cfg: PipelineConfig) -> Dict[str, Any]:
    return asdict(cfg)


def save_config(cfg: PipelineConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config_to_dict(cfg), f, indent=2, ensure_ascii=False)
    log_done(f"Saved config JSON to: {path}")


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = PipelineConfig(
        paths=PathConfig(**data.get("paths", {})),
        analysis=AnalysisConfig(**data.get("analysis", {})),
        plot=PlotConfig(**data.get("plot", {})),
    )
    log_done(f"Loaded config JSON from: {path}")
    return cfg


def ensure_output_dirs(output_root: str | Path) -> Dict[str, Path]:
    root = Path(output_root)
    dirs = {
        "root": root,
        "prepared": root / "prepared",
        "global_alignment": root / "global_alignment",
        "neighborhood": root / "neighborhood",
        "retrieval": root / "retrieval",
        "robustness_root": root / "robustness",
        "robust_adjusted": root / "robustness" / "adjusted",
        "robust_bootstrap": root / "robustness" / "bootstrap",
        "robust_baselines": root / "robustness" / "baselines",
        "robust_discovery_validation": root / "robustness" / "discovery_validation",
        "final": root / "final",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    log_done(f"Ensured clinical alignment output tree under: {root.resolve()}")
    return dirs


# =============================================================================
# Plot style helpers
# =============================================================================


def set_publication_plot_style(plot_cfg: PlotConfig | None = None) -> None:
    if plot_cfg is None:
        plot_cfg = PlotConfig()

    sns.set_theme(context=plot_cfg.context, style=plot_cfg.style, palette=plot_cfg.palette)
    plt.rcParams.update(
        {
            "figure.dpi": plot_cfg.dpi,
            "savefig.dpi": plot_cfg.dpi,
            "font.family": plot_cfg.font_family,
            "font.size": plot_cfg.base_font_size,
            "axes.titlesize": plot_cfg.title_font_size,
            "axes.labelsize": plot_cfg.label_font_size,
            "xtick.labelsize": plot_cfg.tick_font_size,
            "ytick.labelsize": plot_cfg.tick_font_size,
            "legend.fontsize": plot_cfg.legend_font_size,
            "axes.grid": False,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    log_done("Applied shared publication-ready plotting style.")


__all__ = [
    "PathConfig",
    "AnalysisConfig",
    "PlotConfig",
    "PipelineConfig",
    "build_default_config",
    "config_to_dict",
    "save_config",
    "load_config",
    "ensure_output_dirs",
    "set_publication_plot_style",
    "print_banner",
    "log_info",
    "log_warn",
    "log_done",
]
