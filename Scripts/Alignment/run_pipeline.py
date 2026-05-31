
"""Unified runner for robust clinical alignment and interpretability."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from config import (
    PipelineConfig,
    build_default_config,
    ensure_output_dirs,
    save_config,
    set_publication_plot_style,
    print_banner,
    log_warn,
    log_done,
)
from clinical_registry import build_variable_registry
from core import ClinicalAlignmentData, prepare_clinical_alignment_data
from baselines import prepare_baseline_distance_matrices, compare_distance_spaces, save_baseline_artifacts
from analysis import (
    run_global_distance_alignment,
    run_variablewise_global_alignment,
    run_neighborhood_consistency,
    run_retrieval_validation,
    run_adjusted_global_alignment,
    bootstrap_global_alignment,
    bootstrap_neighborhood_consistency,
    bootstrap_retrieval_validation,
    load_saved_baseline_distance_matrices,
    run_baseline_comparison,
    run_discovery_validation_alignment,
    run_positionwise_global_alignment_suite,
)
from interpretability import run_interpretability_suite


def _ensure_file_exists(path: str | Path, desc: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")
    return path


def _save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    log_done(f'Saved JSON to: {path}')


def _apply_runtime_args_to_config(cfg: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    for name in ['acoustic_distance_npy','patient_order_csv','clinical_table_path','window_meta_csv','beats_patient_embedding_npy','beats_patient_embedding_meta_csv','beats_position_embedding_npy','beats_position_embedding_meta_csv','ead_patient_embedding_npy','ead_patient_embedding_meta_csv','ead_pointcloud_distance_npy','ead_pointcloud_patient_order_csv','out_root','position_distance_dir']:
        pass
    if getattr(args, 'acoustic_distance_npy', None): cfg.paths.acoustic_distance_npy = str(args.acoustic_distance_npy)
    if getattr(args, 'patient_order_csv', None): cfg.paths.patient_order_csv = str(args.patient_order_csv)
    if getattr(args, 'clinical_table_path', None): cfg.paths.clinical_table_path = str(args.clinical_table_path)
    if getattr(args, 'window_meta_csv', None): cfg.paths.window_meta_csv = str(args.window_meta_csv)
    if getattr(args, 'beats_patient_embedding_npy', None): cfg.paths.beats_patient_embedding_npy = str(args.beats_patient_embedding_npy)
    if getattr(args, 'beats_patient_embedding_meta_csv', None): cfg.paths.beats_patient_embedding_meta_csv = str(args.beats_patient_embedding_meta_csv)
    if getattr(args, 'beats_position_embedding_npy', None): cfg.paths.beats_position_embedding_npy = str(args.beats_position_embedding_npy)
    if getattr(args, 'beats_position_embedding_meta_csv', None): cfg.paths.beats_position_embedding_meta_csv = str(args.beats_position_embedding_meta_csv)
    if getattr(args, 'ead_patient_embedding_npy', None): cfg.paths.ead_patient_embedding_npy = str(args.ead_patient_embedding_npy)
    if getattr(args, 'ead_patient_embedding_meta_csv', None): cfg.paths.ead_patient_embedding_meta_csv = str(args.ead_patient_embedding_meta_csv)
    if getattr(args, 'ead_pointcloud_distance_npy', None): cfg.paths.ead_pointcloud_distance_npy = str(args.ead_pointcloud_distance_npy)
    if getattr(args, 'ead_pointcloud_patient_order_csv', None): cfg.paths.ead_pointcloud_patient_order_csv = str(args.ead_pointcloud_patient_order_csv)
    if getattr(args, 'out_root', None): cfg.paths.output_root = str(args.out_root)
    if getattr(args, 'position_distance_dir', None): cfg.paths.position_distance_dir = str(args.position_distance_dir)
    if getattr(args, 'global_permutations', None) is not None: cfg.analysis.global_permutations = int(args.global_permutations)
    if getattr(args, 'knn_list', None): cfg.analysis.knn_list = [int(x) for x in args.knn_list]
    if getattr(args, 'random_baseline_repeats', None) is not None: cfg.analysis.random_baseline_repeats = int(args.random_baseline_repeats)
    if getattr(args, 'retrieval_kernel', None) is not None: cfg.analysis.retrieval_kernel = str(args.retrieval_kernel)
    if getattr(args, 'retrieval_sigma', None) is not None: cfg.analysis.retrieval_sigma = str(args.retrieval_sigma)
    if getattr(args, 'bootstrap_repeats', None) is not None: cfg.analysis.bootstrap_repeats = int(args.bootstrap_repeats)
    if getattr(args, 'bootstrap_ci', None) is not None: cfg.analysis.bootstrap_ci = float(args.bootstrap_ci)
    if getattr(args, 'discovery_fraction', None) is not None: cfg.analysis.discovery_fraction = float(args.discovery_fraction)
    if getattr(args, 'split_repeats', None) is not None: cfg.analysis.split_repeats = int(args.split_repeats)
    if getattr(args, 'n_jobs', None) is not None: cfg.analysis.n_jobs = int(args.n_jobs)
    if getattr(args, 'seed', None) is not None: cfg.analysis.random_seed = int(args.seed)
    return cfg


def _load_prepared_data(prepared_dir: str | Path) -> ClinicalAlignmentData:
    return ClinicalAlignmentData.load(prepared_dir)


def _build_and_save_baselines(data: ClinicalAlignmentData, cfg: PipelineConfig, dirs: Dict[str, Path]) -> Dict[str, Any]:
    baseline_mats = prepare_baseline_distance_matrices(data, cfg)
    comparison_df = compare_distance_spaces(baseline_mats)
    save_baseline_artifacts(baseline_mats, comparison_df, dirs['robust_baselines'], plot_cfg=cfg.plot)
    return baseline_mats


def _csv_has_columns(path: str | Path, required_cols: list[str]) -> bool:
    path = Path(path)
    if not path.exists(): return False
    try:
        cols = set(__import__('pandas').read_csv(path, nrows=0).columns.tolist())
    except Exception:
        return False
    return all(col in cols for col in required_cols)


def cmd_prepare(cfg: PipelineConfig, args: argparse.Namespace) -> Dict[str, Path]:
    print_banner('STEP 1 / PREPARE: build prepared clinical alignment data')
    dirs = ensure_output_dirs(cfg.paths.output_root)
    save_config(cfg, dirs['root'] / 'pipeline_config.json')
    set_publication_plot_style(cfg.plot)
    _ensure_file_exists(cfg.paths.acoustic_distance_npy, 'acoustic distance npy')
    _ensure_file_exists(cfg.paths.patient_order_csv, 'patient order csv')
    _ensure_file_exists(cfg.paths.clinical_table_path, 'clinical table')
    _ensure_file_exists(cfg.paths.window_meta_csv, 'window metadata csv')
    data = prepare_clinical_alignment_data(cfg)
    data.save(dirs['prepared'])
    import numpy as np
    for group_name, D in data.clinical_distance_mats.items():
        np.save(dirs['global_alignment'] / f'clinical_distance_{group_name}.npy', D)
    return dirs


def cmd_baselines(cfg: PipelineConfig, args: argparse.Namespace) -> Dict[str, Path]:
    print_banner('STEP 2 / BASELINES: build baseline distance spaces')
    dirs = ensure_output_dirs(cfg.paths.output_root)
    save_config(cfg, dirs['root'] / 'pipeline_config.json')
    set_publication_plot_style(cfg.plot)
    if Path(cfg.paths.beats_position_embedding_npy).exists() and Path(cfg.paths.beats_position_embedding_meta_csv).exists():
        _ensure_file_exists(cfg.paths.beats_position_embedding_npy, 'BEATs position embedding npy')
        _ensure_file_exists(cfg.paths.beats_position_embedding_meta_csv, 'BEATs position embedding meta csv')
    else:
        _ensure_file_exists(cfg.paths.beats_patient_embedding_npy, 'BEATs patient embedding npy')
        _ensure_file_exists(cfg.paths.beats_patient_embedding_meta_csv, 'BEATs patient embedding meta csv')
    _ensure_file_exists(cfg.paths.ead_patient_embedding_npy, 'EAD patient embedding npy')
    _ensure_file_exists(cfg.paths.ead_patient_embedding_meta_csv, 'EAD patient embedding meta csv')
    _ensure_file_exists(cfg.paths.ead_pointcloud_distance_npy, 'EAD point-cloud distance npy')
    _ensure_file_exists(cfg.paths.ead_pointcloud_patient_order_csv, 'EAD point-cloud patient order csv')
    prepared_dir = Path(args.prepared_dir) if getattr(args, 'prepared_dir', None) else dirs['prepared']
    data = _load_prepared_data(prepared_dir)
    _build_and_save_baselines(data, cfg, dirs)
    return dirs


def cmd_global_align(cfg: PipelineConfig, args: argparse.Namespace) -> Dict[str, Path]:
    print_banner('STEP 3 / GLOBAL-ALIGN: global acoustic-clinical distance alignment')
    dirs = ensure_output_dirs(cfg.paths.output_root)
    save_config(cfg, dirs['root'] / 'pipeline_config.json')
    set_publication_plot_style(cfg.plot)
    data = _load_prepared_data(Path(args.prepared_dir) if getattr(args, 'prepared_dir', None) else dirs['prepared'])
    registry = build_variable_registry()
    run_global_distance_alignment(data=data, registry=registry, groups=cfg.analysis.clinical_groups, n_perm=cfg.analysis.global_permutations, n_jobs=cfg.analysis.n_jobs, random_seed=cfg.analysis.random_seed, out_dir=dirs['global_alignment'], plot_cfg=cfg.plot)
    run_variablewise_global_alignment(data=data, registry=registry, groups=cfg.analysis.clinical_groups, n_perm=cfg.analysis.global_permutations, n_jobs=cfg.analysis.n_jobs, random_seed=cfg.analysis.random_seed, out_dir=dirs['global_alignment'], plot_cfg=cfg.plot, top_n=cfg.analysis.variablewise_top_n)

    position_distance_dir = Path(cfg.paths.position_distance_dir)
    if position_distance_dir.exists():
        run_positionwise_global_alignment_suite(
            data=data,
            registry=registry,
            groups=cfg.analysis.clinical_groups,
            position_distance_dir=position_distance_dir,
            positions=cfg.analysis.positions,
            file_pattern=cfg.analysis.position_distance_pattern,
            n_perm=cfg.analysis.global_permutations,
            n_jobs=cfg.analysis.n_jobs,
            random_seed=cfg.analysis.random_seed,
            out_dir=dirs['global_alignment'] / 'position_level',
            plot_cfg=cfg.plot,
            top_n=cfg.analysis.variablewise_top_n,
        )
    else:
        log_warn(f'Position distance directory not found, skipping position-wise alignment: {position_distance_dir}')
    return dirs


def cmd_neighborhood(cfg: PipelineConfig, args: argparse.Namespace) -> Dict[str, Path]:
    print_banner('STEP 4 / NEIGHBORHOOD: local neighborhood consistency')
    dirs = ensure_output_dirs(cfg.paths.output_root)
    save_config(cfg, dirs['root'] / 'pipeline_config.json')
    set_publication_plot_style(cfg.plot)
    data = _load_prepared_data(Path(args.prepared_dir) if getattr(args, 'prepared_dir', None) else dirs['prepared'])
    registry = build_variable_registry()
    run_neighborhood_consistency(data=data, registry=registry, groups=cfg.analysis.clinical_groups, knn_list=cfg.analysis.knn_list, random_repeats=cfg.analysis.random_baseline_repeats, random_seed=cfg.analysis.random_seed, out_dir=dirs['neighborhood'], plot_cfg=cfg.plot)
    return dirs


def cmd_retrieval(cfg: PipelineConfig, args: argparse.Namespace) -> Dict[str, Path]:
    print_banner('STEP 5 / RETRIEVAL: patient-level retrieval validation')
    dirs = ensure_output_dirs(cfg.paths.output_root)
    save_config(cfg, dirs['root'] / 'pipeline_config.json')
    set_publication_plot_style(cfg.plot)
    data = _load_prepared_data(Path(args.prepared_dir) if getattr(args, 'prepared_dir', None) else dirs['prepared'])
    registry = build_variable_registry()
    run_retrieval_validation(data=data, registry=registry, knn_list=cfg.analysis.knn_list, retrieval_kernel=cfg.analysis.retrieval_kernel, retrieval_sigma=cfg.analysis.retrieval_sigma, out_dir=dirs['retrieval'], plot_cfg=cfg.plot)
    return dirs


def cmd_robustness(cfg: PipelineConfig, args: argparse.Namespace) -> Dict[str, Path]:
    print_banner('STEP 6 / ROBUSTNESS: supplementary validation')
    dirs = ensure_output_dirs(cfg.paths.output_root)
    save_config(cfg, dirs['root'] / 'pipeline_config.json')
    set_publication_plot_style(cfg.plot)
    prepared_dir = Path(args.prepared_dir) if getattr(args, 'prepared_dir', None) else dirs['prepared']
    data = _load_prepared_data(prepared_dir)
    registry = build_variable_registry()
    adjusted_summary = run_adjusted_global_alignment(data=data, registry=registry, groups=cfg.analysis.clinical_groups, adjust_covariates=cfg.analysis.adjust_covariates, technical_covariates=cfg.analysis.technical_covariates, n_perm=cfg.analysis.global_permutations, n_jobs=cfg.analysis.n_jobs, random_seed=cfg.analysis.random_seed, out_dir=dirs['robust_adjusted'], plot_cfg=cfg.plot)
    boot_global = bootstrap_global_alignment(data=data, registry=registry, groups=cfg.analysis.clinical_groups, repeats=cfg.analysis.bootstrap_repeats, ci=cfg.analysis.bootstrap_ci, n_jobs=cfg.analysis.n_jobs, seed=cfg.analysis.random_seed, out_dir=dirs['robust_bootstrap'] / 'global_alignment', plot_cfg=cfg.plot)
    boot_neighborhood = bootstrap_neighborhood_consistency(data=data, registry=registry, groups=cfg.analysis.clinical_groups, knn_list=cfg.analysis.knn_list, random_repeats=cfg.analysis.random_baseline_repeats, repeats=cfg.analysis.bootstrap_repeats, ci=cfg.analysis.bootstrap_ci, n_jobs=cfg.analysis.n_jobs, seed=cfg.analysis.random_seed, out_dir=dirs['robust_bootstrap'] / 'neighborhood', plot_cfg=cfg.plot, subsample_fraction=cfg.analysis.bootstrap_subsample_fraction)
    boot_retrieval = bootstrap_retrieval_validation(data=data, registry=registry, knn_list=cfg.analysis.knn_list, kernel=cfg.analysis.retrieval_kernel, sigma=cfg.analysis.retrieval_sigma, repeats=cfg.analysis.bootstrap_repeats, ci=cfg.analysis.bootstrap_ci, n_jobs=cfg.analysis.n_jobs, seed=cfg.analysis.random_seed, out_dir=dirs['robust_bootstrap'] / 'retrieval', plot_cfg=cfg.plot, subsample_fraction=cfg.analysis.bootstrap_subsample_fraction)
    baseline_dir = dirs['robust_baselines']
    baseline_mats = load_saved_baseline_distance_matrices(baseline_dir) if (baseline_dir/'baseline_summary.json').exists() else _build_and_save_baselines(data, cfg, dirs)
    baseline_comparison = run_baseline_comparison(data=data, registry=registry, baseline_mats=baseline_mats, groups=cfg.analysis.clinical_groups, knn_list=cfg.analysis.knn_list, retrieval_kernel=cfg.analysis.retrieval_kernel, retrieval_sigma=cfg.analysis.retrieval_sigma, out_dir=baseline_dir, plot_cfg=cfg.plot)
    discovery_val = run_discovery_validation_alignment(data=data, registry=registry, groups=cfg.analysis.clinical_groups, knn_list=cfg.analysis.knn_list, retrieval_kernel=cfg.analysis.retrieval_kernel, retrieval_sigma=cfg.analysis.retrieval_sigma, discovery_fraction=cfg.analysis.discovery_fraction, split_repeats=cfg.analysis.split_repeats, random_seed=cfg.analysis.random_seed, out_dir=dirs['robust_discovery_validation'], plot_cfg=cfg.plot)
    manifest = {'adjusted_rows': int(len(adjusted_summary)), 'bootstrap_global_rows': int(len(boot_global)), 'bootstrap_neighborhood_rows': int(len(boot_neighborhood)), 'bootstrap_retrieval_rows': int(len(boot_retrieval)), 'baseline_rows': int(len(baseline_comparison)), 'discovery_validation_rows': int(len(discovery_val))}
    _save_json(manifest, dirs['robustness_root'] / 'robustness_manifest.json')
    return dirs


def cmd_interpretability(cfg: PipelineConfig, args: argparse.Namespace) -> Dict[str, Path]:
    print_banner('STEP 7 / INTERPRETABILITY: statistical consolidation and explanation')
    dirs = ensure_output_dirs(cfg.paths.output_root)
    dirs['interpretability'] = dirs['root'] / 'interpretability'
    dirs['interpretability'].mkdir(parents=True, exist_ok=True)
    save_config(cfg, dirs['root'] / 'pipeline_config.json')
    set_publication_plot_style(cfg.plot)
    prepared_dir = Path(args.prepared_dir) if getattr(args, 'prepared_dir', None) else dirs['prepared']
    data = _load_prepared_data(prepared_dir)
    registry = build_variable_registry()
    var_path = dirs['global_alignment'] / 'variablewise_global_alignment_summary.csv'
    if not _csv_has_columns(var_path, ['perm_q_spearman_overall','perm_q_spearman_within_group']):
        log_warn('Variable-wise global summary missing FDR columns; refreshing global-align outputs.')
        cmd_global_align(cfg, args)
    nb_var_path = dirs['neighborhood'] / 'neighborhood_variable_summary.csv'
    if not _csv_has_columns(nb_var_path, ['perm_q_overall_within_k','perm_q_within_group_within_k']):
        log_warn('Neighborhood variable summary missing FDR columns; refreshing neighborhood outputs.')
        cmd_neighborhood(cfg, args)
    retrieval_neighbor_path = dirs['retrieval'] / 'retrieval_neighbor_details.csv'
    if not retrieval_neighbor_path.exists():
        log_warn('Retrieval neighbor details not found; refreshing retrieval outputs.')
        cmd_retrieval(cfg, args)
    run_interpretability_suite(data=data, registry=registry, cfg=cfg, out_root=dirs['interpretability'])
    _save_json({'status':'completed'}, dirs['interpretability'] / 'interpretability_run_manifest.json')
    return dirs


def cmd_all(cfg: PipelineConfig, args: argparse.Namespace) -> Dict[str, Path]:
    print_banner('RUNNING FULL CLINICAL ALIGNMENT PIPELINE')
    dirs = cmd_prepare(cfg, args)
    dirs = cmd_baselines(cfg, args)
    dirs = cmd_global_align(cfg, args)
    dirs = cmd_neighborhood(cfg, args)
    dirs = cmd_retrieval(cfg, args)
    dirs = cmd_robustness(cfg, args)
    dirs = cmd_interpretability(cfg, args)
    _save_json({'status':'completed'}, dirs['final'] / 'run_manifest.json')
    log_done('Full clinical alignment pipeline finished successfully.')
    return dirs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Unified runner for robust clinical alignment and interpretability.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument('--acoustic-distance-npy', type=str, default=None)
    parent.add_argument('--patient-order-csv', type=str, default=None)
    parent.add_argument('--clinical-table-path', type=str, default=None)
    parent.add_argument('--window-meta-csv', type=str, default=None)
    parent.add_argument('--beats-patient-embedding-npy', type=str, default=None)
    parent.add_argument('--beats-patient-embedding-meta-csv', type=str, default=None)
    parent.add_argument('--beats-position-embedding-npy', type=str, default=None)
    parent.add_argument('--beats-position-embedding-meta-csv', type=str, default=None)
    parent.add_argument('--ead-patient-embedding-npy', type=str, default=None)
    parent.add_argument('--ead-patient-embedding-meta-csv', type=str, default=None)
    parent.add_argument('--ead-pointcloud-distance-npy', type=str, default=None)
    parent.add_argument('--ead-pointcloud-patient-order-csv', type=str, default=None)
    parent.add_argument('--out-root', type=str, default=str(THIS_DIR / 'outputs'))
    parent.add_argument('--position-distance-dir', type=str, default=None)
    parent.add_argument('--n-jobs', type=int, default=None)
    parent.add_argument('--seed', type=int, default=None)
    main_parent = argparse.ArgumentParser(add_help=False)
    main_parent.add_argument('--global-permutations', type=int, default=None)
    main_parent.add_argument('--knn-list', nargs='+', type=int, default=None)
    main_parent.add_argument('--random-baseline-repeats', type=int, default=None)
    main_parent.add_argument('--retrieval-kernel', type=str, default=None, choices=['rbf','inverse_distance'])
    main_parent.add_argument('--retrieval-sigma', type=str, default=None)
    robust_parent = argparse.ArgumentParser(add_help=False)
    robust_parent.add_argument('--bootstrap-repeats', type=int, default=None)
    robust_parent.add_argument('--bootstrap-ci', type=float, default=None)
    robust_parent.add_argument('--discovery-fraction', type=float, default=None)
    robust_parent.add_argument('--split-repeats', type=int, default=None)
    subparsers.add_parser('prepare', parents=[parent], help='Build the prepared clinical-alignment data bundle')
    p_base = subparsers.add_parser('baselines', parents=[parent], help='Build and compare baseline distance spaces'); p_base.add_argument('--prepared-dir', type=str, default=None)
    p_global = subparsers.add_parser('global-align', parents=[parent, main_parent], help='Run global acoustic-clinical distance alignment'); p_global.add_argument('--prepared-dir', type=str, default=None)
    p_nb = subparsers.add_parser('neighborhood', parents=[parent, main_parent], help='Run neighborhood consistency analysis'); p_nb.add_argument('--prepared-dir', type=str, default=None)
    p_ret = subparsers.add_parser('retrieval', parents=[parent, main_parent], help='Run retrieval-based validation'); p_ret.add_argument('--prepared-dir', type=str, default=None)
    p_rob = subparsers.add_parser('robustness', parents=[parent, main_parent, robust_parent], help='Run all supplementary robustness analyses'); p_rob.add_argument('--prepared-dir', type=str, default=None)
    p_int = subparsers.add_parser('interpretability', parents=[parent, main_parent], help='Run statistical consolidation and interpretability analyses'); p_int.add_argument('--prepared-dir', type=str, default=None)
    p_all = subparsers.add_parser('all', parents=[parent, main_parent, robust_parent], help='Run the full main + robustness + interpretability pipeline'); p_all.add_argument('--prepared-dir', type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser(); args = parser.parse_args()
    try:
        cfg = build_default_config(); cfg = _apply_runtime_args_to_config(cfg, args)
        if args.command == 'prepare': cmd_prepare(cfg, args)
        elif args.command == 'baselines': cmd_baselines(cfg, args)
        elif args.command == 'global-align': cmd_global_align(cfg, args)
        elif args.command == 'neighborhood': cmd_neighborhood(cfg, args)
        elif args.command == 'retrieval': cmd_retrieval(cfg, args)
        elif args.command == 'robustness': cmd_robustness(cfg, args)
        elif args.command == 'interpretability': cmd_interpretability(cfg, args)
        elif args.command == 'all': cmd_all(cfg, args)
        else: raise ValueError(f'Unsupported command: {args.command}')
    except Exception as exc:
        print('\n[error] Clinical alignment pipeline failed.')
        print(f'[error] {type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
