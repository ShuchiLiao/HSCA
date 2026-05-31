from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:  # local package-style imports
    from config import PlotConfig, build_default_config, set_publication_plot_style, log_done, log_warn, log_info, print_banner
except Exception:
    from dataclasses import dataclass, field
    @dataclass
    class PlotConfig:
        dpi: int = 300
        font_family: str = 'Arial'
        base_font_size: int = 20
        title_font_size: int = 24
        label_font_size: int = 20
        tick_font_size: int = 18
        legend_font_size: int = 16
        context: str = 'talk'
        style: str = 'white'
        palette: List[str] = field(default_factory=lambda: [
            '#158F8C', '#2F6DB3', '#E38B2C', '#7B5AA6', '#5FB0B7', '#F1B44C', '#C75C5C'
        ])
        axis_grid_alpha: float = 0.22
        axis_grid_linewidth: float = 0.9
        heatmap_grid_linewidth: float = 0.8
        heatmap_grid_color: str = '#F3F3F3'
        heatmap_annot: bool = True
        global_scatter_max_points: int = 60000
        neighborhood_heatmap_top_n: int = 18
        bootstrap_retrieval_top_n: int = 12
        retrieval_heatmap_top_n_continuous: int = 12
        retrieval_heatmap_top_n_categorical: int = 8
    def build_default_config():
        class X: pass
        x = X()
        x.plot = PlotConfig()
        return x
    def set_publication_plot_style(plot_cfg=None):
        if plot_cfg is None:
            plot_cfg = PlotConfig()
        sns.set_theme(context=plot_cfg.context, style=plot_cfg.style, palette=plot_cfg.palette)
        plt.rcParams.update({
            'figure.dpi': plot_cfg.dpi,
            'savefig.dpi': plot_cfg.dpi,
            'font.family': plot_cfg.font_family,
            'font.size': plot_cfg.base_font_size,
            'axes.titlesize': plot_cfg.title_font_size,
            'axes.labelsize': plot_cfg.label_font_size,
            'xtick.labelsize': plot_cfg.tick_font_size,
            'ytick.labelsize': plot_cfg.tick_font_size,
            'legend.fontsize': plot_cfg.legend_font_size,
            'axes.grid': False,
            'axes.spines.top': True,
            'axes.spines.right': True,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'legend.frameon': False,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        })
    def log_done(msg): print(f'[done] {msg}')
    def log_warn(msg): print(f'[warn] {msg}')
    def log_info(msg): print(f'[info] {msg}')
    def print_banner(title):
        line='='*96
        print(f'\n{line}\n{title}\n{line}')


def _cfg_attr(plot_cfg: PlotConfig, name: str, default):
    return getattr(plot_cfg, name, default)


def _apply_axes_style(ax: plt.Axes, plot_cfg: Optional[PlotConfig] = None, show_grid: bool = False, grid_axis: str = 'both') -> None:
    plot_cfg = PlotConfig() if plot_cfg is None else plot_cfg
    if show_grid:
        ax.grid(True, axis=grid_axis, alpha=_cfg_attr(plot_cfg, 'axis_grid_alpha', 0.22), linewidth=_cfg_attr(plot_cfg, 'axis_grid_linewidth', 0.9))
    else:
        ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.4)


def _save_fig(fig: plt.Figure, out_path: str | Path, dpi: int = 300) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    log_done(f'Saved figure to: {out_path}')


def _ensure_square_distance_matrix(D: np.ndarray, name: str = 'distance_matrix') -> np.ndarray:
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f'{name} must be square, got shape={D.shape}')
    if np.any(~np.isfinite(D)):
        raise ValueError(f'{name} contains NaN or Inf.')
    if not np.allclose(D, D.T, atol=1e-7):
        raise ValueError(f'{name} is not symmetric within tolerance.')
    np.fill_diagonal(D, 0.0)
    return D


def upper_triangle_vector(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D)
    return D[np.triu_indices_from(D, k=1)]


def _sample_for_scatter(x: np.ndarray, y: np.ndarray, max_points: int = 50000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_points, replace=False)
    return x[idx], y[idx]


def _binned_trend(x: np.ndarray, y: np.ndarray, n_bins: int = 24) -> pd.DataFrame:
    if len(x) < max(10, n_bins):
        return pd.DataFrame(columns=['x_mid', 'y_med', 'y_lo', 'y_hi'])
    bins = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), n_bins + 1)
    rows: List[Dict[str, float]] = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (x >= left) & (x < right if right < bins[-1] else x <= right)
        if np.sum(mask) < 10:
            continue
        ys = y[mask]
        rows.append({
            'x_mid': float((left + right) / 2.0),
            'y_med': float(np.nanmedian(ys)),
            'y_lo': float(np.nanquantile(ys, 0.25)),
            'y_hi': float(np.nanquantile(ys, 0.75)),
        })
    return pd.DataFrame(rows)


def _metric_display_name(metric: str) -> str:
    return {
        'pearson_r': 'Pearson r',
        'spearman_r': 'Spearman r',
        'improvement_ratio': 'Improvement ratio',
        'balanced_accuracy': 'Balanced accuracy',
        'spearman': 'Spearman',
    }.get(metric, metric)


# ---- plotting funcs extracted/adapted from analysis.py and baselines.py ----

def plot_global_alignment_scatter(acoustic_distance: np.ndarray, clinical_distance: np.ndarray, group_name: str, out_path: str | Path, plot_cfg: PlotConfig, seed: int = 42) -> None:
    set_publication_plot_style(plot_cfg)
    x = upper_triangle_vector(_ensure_square_distance_matrix(acoustic_distance))
    y = upper_triangle_vector(_ensure_square_distance_matrix(clinical_distance))
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    x_s, y_s = _sample_for_scatter(x, y, max_points=_cfg_attr(plot_cfg, 'global_scatter_max_points', 60000), seed=seed)
    fig, ax = plt.subplots(figsize=(9.6, 8.0))
    cmap = sns.blend_palette(['#F1B44C', '#158F8C', '#2F6DB3', '#7B5AA6'], as_cmap=True)
    hb = ax.hexbin(x_s, y_s, gridsize=46, cmap=cmap, mincnt=1)
    trend = _binned_trend(x_s, y_s, n_bins=24)
    if len(trend) > 0:
        ax.fill_between(trend['x_mid'], trend['y_lo'], trend['y_hi'], color='#C75C5C', alpha=0.18, linewidth=0)
        ax.plot(trend['x_mid'], trend['y_med'], color='#C75C5C', linewidth=2.5, label='Binned median trend')
        ax.legend(loc='upper right')
    cbar = fig.colorbar(hb, ax=ax, shrink=0.82)
    cbar.set_label('Pair count')
    ax.set_title(f'Acoustic vs Clinical Distance | {group_name}')
    ax.set_xlabel('Acoustic distance')
    ax.set_ylabel('Clinical distance')
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False)
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_global_alignment_summary(summary_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig) -> None:
    if len(summary_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = summary_df[['group', 'pearson_r', 'spearman_r']].copy()
    plot_df = plot_df.sort_values('spearman_r', ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(plot_df), dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, max(5.5, 1.35 * len(plot_df))))
    for y, (_, row) in zip(y_pos, plot_df.iterrows()):
        ax.plot([row['pearson_r'], row['spearman_r']], [y, y], color='#B7B7B7', linewidth=2.0, zorder=1)
    ax.scatter(plot_df['pearson_r'], y_pos, s=140, color='#158F8C', label='Pearson r', zorder=3)
    ax.scatter(plot_df['spearman_r'], y_pos, s=140, color='#2F6DB3', label='Spearman r', zorder=3)
    ax.axvline(0.0, color='#7F7F7F', linestyle='--', linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['group'].tolist())
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Clinical group')
    ax.set_title('Global acoustic-clinical distance alignment')
    ax.legend(loc='best')
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='x')
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_global_alignment_forest(summary_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig) -> None:
    if len(summary_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = summary_df.copy()
    plot_df['label'] = plot_df['group'].astype(str) + ' | ' + plot_df['metric'].map(_metric_display_name)
    plot_df = plot_df.sort_values(['group', 'metric']).reset_index(drop=True)
    y = np.arange(len(plot_df))[::-1]
    fig, ax = plt.subplots(figsize=(11, max(6.5, 0.7 * len(plot_df))))
    colors = {'pearson_r': '#158F8C', 'spearman_r': '#2F6DB3', 'improvement_ratio': '#E38B2C', 'balanced_accuracy': '#7B5AA6', 'spearman': '#C75C5C'}
    for yi, (_, row) in zip(y, plot_df.iterrows()):
        ax.plot([row['ci_lower'], row['ci_upper']], [yi, yi], color=colors.get(str(row['metric']), '#4C4C4C'), linewidth=2.6)
        ax.scatter(row['estimate_mean'], yi, s=110, color=colors.get(str(row['metric']), '#4C4C4C'), zorder=3)
    ax.axvline(0.0, linestyle='--', color='#7F7F7F', linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['label'].tolist())
    ax.set_xlabel('Estimate (mean with bootstrap CI)')
    ax.set_ylabel('')
    ax.set_title('Bootstrap robustness | global alignment')
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='x')
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_bootstrap_global_distribution(raw_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig) -> None:
    if len(raw_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = raw_df.melt(id_vars=['group', 'repeat'], value_vars=['pearson_r', 'spearman_r'], var_name='metric', value_name='value')
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    sns.violinplot(data=plot_df, x='group', y='value', hue='metric', inner=None, cut=0, linewidth=0, ax=ax)
    sns.boxplot(data=plot_df, x='group', y='value', hue='metric', showcaps=True, boxprops={'facecolor': 'none', 'edgecolor': 'black'}, whiskerprops={'linewidth': 1.0}, medianprops={'color': 'black'}, showfliers=False, width=0.28, ax=ax)
    sample_df = plot_df.sample(min(len(plot_df), 1200), random_state=42) if len(plot_df) > 0 else plot_df
    sns.stripplot(data=sample_df, x='group', y='value', hue='metric', dodge=True, alpha=0.18, size=3.0, color='black', ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ['Pearson r', 'Spearman r'], loc='best')
    ax.set_title('Bootstrap robustness | global alignment distribution')
    ax.set_xlabel('Clinical group')
    ax.set_ylabel('Bootstrap estimate')
    ax.axhline(0.0, linestyle='--', color='#7F7F7F', linewidth=1.2)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='y')
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_variablewise_global_alignment_topn(summary_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig, top_n: int = 12) -> None:
    if len(summary_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = summary_df.sort_values('spearman_r', ascending=False).head(int(top_n)).copy()
    plot_df = plot_df.sort_values('spearman_r', ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11.5, max(6.8, 0.55 * len(plot_df))))
    ax.hlines(y=plot_df['variable'], xmin=0.0, xmax=plot_df['spearman_r'], color='#B7B7B7', linewidth=2.0)
    sns.scatterplot(data=plot_df, x='spearman_r', y='variable', hue='group', s=130, ax=ax, zorder=3)
    ax.axvline(0.0, linestyle='--', color='#7F7F7F', linewidth=1.2)
    ax.set_title(f'Variable-wise global alignment | top {int(top_n)}')
    ax.set_xlabel('Spearman correlation')
    ax.set_ylabel('Clinical variable')
    ax.legend(loc='best')
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='x')
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_variablewise_global_alignment_all(summary_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig) -> None:
    if len(summary_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = summary_df.copy()
    plot_df['group_rank'] = plot_df['group'].map({'function': 0, 'structure': 1, 'burden': 2}).fillna(99)
    plot_df = plot_df.sort_values(['group_rank', 'spearman_r'], ascending=[True, True]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12.8, max(8.0, 0.4 * len(plot_df))))
    sns.scatterplot(data=plot_df, x='spearman_r', y='variable', hue='group', size='coverage', sizes=(60, 180), ax=ax, zorder=3)
    ax.axvline(0.0, linestyle='--', color='#7F7F7F', linewidth=1.2)
    ax.set_title('Variable-wise global alignment | all variables')
    ax.set_xlabel('Spearman correlation')
    ax.set_ylabel('Clinical variable')
    ax.legend(loc='best')
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='x')
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_neighborhood_summary(summary_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig) -> None:
    if len(summary_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = summary_df.sort_values(['group', 'k']).copy()
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    sns.lineplot(data=plot_df, x='k', y='improvement_ratio', hue='group', marker='o', linewidth=2.6, markersize=9, ax=ax)
    ax.axhline(1.0, linestyle='--', color='#7F7F7F', linewidth=1.2)
    ax.set_title('Neighborhood clinical consistency')
    ax.set_xlabel('Top-k acoustic neighbors')
    ax.set_ylabel('Improvement ratio vs random')
    ax.legend(loc='best')
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='y')
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_bootstrap_neighborhood_summary(summary_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig) -> None:
    if len(summary_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    fig, ax = plt.subplots(figsize=(10.8, 7.3))
    for group, sub in summary_df.groupby('group', dropna=False):
        sub = sub.sort_values('k')
        ax.plot(sub['k'], sub['estimate_mean'], marker='o', linewidth=2.4, markersize=8, label=str(group))
        ax.fill_between(sub['k'], sub['ci_lower'], sub['ci_upper'], alpha=0.18)
    ax.axhline(1.0, linestyle='--', color='#7F7F7F', linewidth=1.2)
    ax.set_title('Bootstrap robustness | neighborhood consistency')
    ax.set_xlabel('Top-k acoustic neighbors')
    ax.set_ylabel('Improvement ratio')
    ax.legend(loc='best')
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='y')
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_variable_level_neighbor_heatmap(variable_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig) -> None:
    if len(variable_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    ranked = (variable_df.groupby('variable', dropna=False)['improvement_ratio'].mean().sort_values(ascending=False).head(int(_cfg_attr(plot_cfg, 'neighborhood_heatmap_top_n', 18))).index.tolist())
    plot_df = variable_df.loc[variable_df['variable'].isin(ranked)].copy()
    pivot = plot_df.pivot_table(index='variable', columns='k', values='improvement_ratio', aggfunc='mean')
    pivot = pivot.reindex(ranked)
    fig, ax = plt.subplots(figsize=(10.5, max(6.5, 0.48 * len(pivot))))
    cmap = sns.blend_palette(['#F1B44C', '#158F8C', '#2F6DB3', '#7B5AA6'], as_cmap=True)
    sns.heatmap(pivot, cmap=cmap, ax=ax, linewidths=_cfg_attr(plot_cfg, 'heatmap_grid_linewidth', 0.8), linecolor=_cfg_attr(plot_cfg, 'heatmap_grid_color', '#F3F3F3'), annot=_cfg_attr(plot_cfg, 'heatmap_annot', True), fmt='.2f', cbar_kws={'label': 'Improvement ratio'})
    ax.set_title(f'Variable-level neighbor consistency | top {len(pivot)}')
    ax.set_xlabel('Top-k acoustic neighbors')
    ax.set_ylabel('Variable')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False)
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def _plot_metric_heatmap(df: pd.DataFrame, value_col: str, title: str, out_path: str | Path, plot_cfg: PlotConfig, top_n: int) -> None:
    if len(df) == 0 or value_col not in df.columns:
        return
    ranked = df.groupby('target', dropna=False)[value_col].max().sort_values(ascending=False).head(int(top_n)).index.tolist()
    plot_df = df.loc[df['target'].isin(ranked)].copy()
    pivot = plot_df.pivot_table(index='target', columns='k', values=value_col, aggfunc='mean')
    pivot = pivot.reindex(ranked)
    fig, ax = plt.subplots(figsize=(9.8, max(6.0, 0.5 * len(pivot))))
    cmap = sns.blend_palette(['#F7FBFF', '#9ECAE1', '#3182BD', '#08519C'], as_cmap=True)
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap=cmap, linewidths=_cfg_attr(plot_cfg, 'heatmap_grid_linewidth', 0.8), linecolor=_cfg_attr(plot_cfg, 'heatmap_grid_color', '#F3F3F3'), cbar_kws={'label': _metric_display_name(value_col)}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Top-k acoustic neighbors')
    ax.set_ylabel('')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False)
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_retrieval_summary(summary_df: pd.DataFrame, out_dir: str | Path, plot_cfg: PlotConfig, stem_prefix: str = 'retrieval') -> List[Path]:
    created = []
    if len(summary_df) == 0:
        return created
    set_publication_plot_style(plot_cfg)
    out_dir = Path(out_dir)
    cont_df = summary_df.loc[summary_df['target_type'].isin(['continuous', 'ordinal'])].copy()
    if len(cont_df) > 0 and 'spearman' in cont_df.columns:
        fp = out_dir / f'{stem_prefix}_continuous_heatmap.png'
        _plot_metric_heatmap(cont_df, 'spearman', f"{stem_prefix.replace('_', ' ').title()} | continuous / ordinal targets", fp, plot_cfg, _cfg_attr(plot_cfg, 'bootstrap_retrieval_top_n', 12) if stem_prefix.startswith('bootstrap') else _cfg_attr(plot_cfg, 'retrieval_heatmap_top_n_continuous', 12))
        created.append(fp)
    cls_df = summary_df.loc[summary_df['target_type'].isin(['binary', 'categorical'])].copy()
    if len(cls_df) > 0 and 'balanced_accuracy' in cls_df.columns:
        fp = out_dir / f'{stem_prefix}_categorical_heatmap.png'
        _plot_metric_heatmap(cls_df, 'balanced_accuracy', f"{stem_prefix.replace('_', ' ').title()} | categorical targets", fp, plot_cfg, _cfg_attr(plot_cfg, 'retrieval_heatmap_top_n_categorical', 8))
        created.append(fp)
    return created


def plot_baseline_comparison(baseline_df: pd.DataFrame, out_dir: str | Path, plot_cfg: PlotConfig) -> List[Path]:
    created = []
    if len(baseline_df) == 0:
        return created
    set_publication_plot_style(plot_cfg)
    plot_df = baseline_df.loc[baseline_df['analysis'] == 'global_alignment'].copy()
    if len(plot_df) == 0:
        return created
    main_df = plot_df.loc[plot_df['group'] != 'all'].copy()
    out_dir = Path(out_dir)
    if len(main_df) > 0:
        fig, ax = plt.subplots(figsize=(11, 7.0))
        sns.barplot(data=main_df, x='group', y='spearman_r', hue='space', ax=ax)
        ax.set_title('Baseline-space comparison | global alignment')
        ax.set_xlabel('Clinical group')
        ax.set_ylabel('Spearman correlation')
        ax.legend(loc='best')
        _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='y')
        fp = out_dir / 'baseline_comparison_global_alignment.png'
        _save_fig(fig, fp, dpi=plot_cfg.dpi)
        created.append(fp)
    all_df = plot_df.loc[plot_df['group'] == 'all'].copy()
    if len(all_df) > 0:
        fig, ax = plt.subplots(figsize=(9.0, 6.7))
        sns.barplot(data=all_df, x='group', y='spearman_r', hue='space', ax=ax)
        ax.set_title('Baseline-space comparison | integrated clinical distance')
        ax.set_xlabel('Clinical group')
        ax.set_ylabel('Spearman correlation')
        ax.legend(loc='best')
        _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='y')
        fp = out_dir / 'baseline_comparison_global_alignment_all.png'
        _save_fig(fig, fp, dpi=plot_cfg.dpi)
        created.append(fp)
    return created


def plot_distance_space_correlation_heatmap(comparison_df: pd.DataFrame, metric: str, out_path: str | Path, plot_cfg: Optional[PlotConfig] = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    set_publication_plot_style(plot_cfg)
    plot_cfg = PlotConfig() if plot_cfg is None else plot_cfg
    pivot = comparison_df.pivot(index='space_i', columns='space_j', values=metric)
    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    cmap = sns.blend_palette(['#F7FBFF', '#9ECAE1', '#3182BD', '#08519C'], as_cmap=True)
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap=cmap, vmin=-1, vmax=1, square=True, linewidths=_cfg_attr(plot_cfg, 'heatmap_grid_linewidth', 0.8), linecolor=_cfg_attr(plot_cfg, 'heatmap_grid_color', '#F3F3F3'), cbar_kws={'shrink': 0.85, 'label': metric}, ax=ax)
    ax.set_title(f'Distance-space comparison ({metric})')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False)
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_discovery_validation_selection_frequency(summary_df: pd.DataFrame, out_dir: str | Path, plot_cfg: PlotConfig) -> List[Path]:
    created = []
    if len(summary_df) == 0:
        return created
    set_publication_plot_style(plot_cfg)
    disc = summary_df.loc[summary_df['subset'] == 'discovery'].copy()
    if len(disc) == 0:
        return created
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    freq = disc['best_group_from_discovery'].value_counts().rename_axis('group').reset_index(name='count')
    ax.bar(freq['group'].astype(str), freq['count'].to_numpy())
    ax.set_title('Discovery-validation | best-group selection frequency')
    ax.set_xlabel('Best group selected in discovery')
    ax.set_ylabel('Count across splits')
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='y')
    fp = out_dir / 'discovery_validation_best_group_frequency.png'
    _save_fig(fig, fp, dpi=plot_cfg.dpi)
    created.append(fp)
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    kfreq = disc['best_k_from_discovery'].value_counts().rename_axis('k').reset_index(name='count')
    ax.bar(kfreq['k'].astype(str), kfreq['count'].to_numpy())
    ax.set_title('Discovery-validation | best-k selection frequency')
    ax.set_xlabel('Best k selected in discovery')
    ax.set_ylabel('Count across splits')
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='y')
    fp = out_dir / 'discovery_validation_best_k_frequency.png'
    _save_fig(fig, fp, dpi=plot_cfg.dpi)
    created.append(fp)
    return created


def plot_discovery_validation_paired(summary_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig, metric_col: str, title: str, ylabel: str) -> None:
    if len(summary_df) == 0 or metric_col not in summary_df.columns:
        return
    set_publication_plot_style(plot_cfg)
    pivot = summary_df.pivot(index='split_id', columns='subset', values=metric_col)
    if not {'discovery', 'validation'}.issubset(set(pivot.columns)):
        return
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    for _, row in pivot.iterrows():
        ax.plot([0, 1], [row['discovery'], row['validation']], color='#B7B7B7', linewidth=1.8, zorder=1)
        ax.scatter([0, 1], [row['discovery'], row['validation']], s=95, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['discovery', 'validation'])
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='y')
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


# ---- render orchestration ----



def _locate_file(root: Path, relative_path: str) -> Path | None:
    candidate = root / relative_path
    if candidate.exists():
        return candidate
    matches = list(root.rglob(Path(relative_path).name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer exact basename match under expected final parent name when possible.
        parent_name = Path(relative_path).parent.name
        for m in matches:
            if parent_name and m.parent.name == parent_name:
                return m
        return matches[0]
    return None


def _safe_read_csv_rel(root: Path, relative_path: str) -> Optional[pd.DataFrame]:
    path = _locate_file(root, relative_path)
    if path is None:
        log_warn(f'Missing csv, skip: {relative_path}')
        return None
    log_info(f'Using csv: {path}')
    return pd.read_csv(path)


def _safe_load_npy_rel(root: Path, relative_path: str) -> Optional[np.ndarray]:
    path = _locate_file(root, relative_path)
    if path is None:
        log_warn(f'Missing npy, skip: {relative_path}')
        return None
    log_info(f'Using npy: {path}')
    return np.load(path)


def render_global(root: Path, fig_dir: Path, plot_cfg: PlotConfig, created: List[str]) -> None:
    print_banner('Rendering global-alignment figures')
    D_ac = _safe_load_npy_rel(root, 'prepared/acoustic_distance.npy')
    if D_ac is not None:
        for group in ['all', 'burden', 'function', 'structure']:
            D_clin = _safe_load_npy_rel(root, f'global_alignment/clinical_distance_{group}.npy')
            if D_clin is not None:
                fp = fig_dir / f'global_acoustic_vs_clinical_distance_{group}.png'
                plot_global_alignment_scatter(D_ac, D_clin, group, fp, plot_cfg)
                created.append(str(fp))
    summary = _safe_read_csv_rel(root, 'global_alignment/global_alignment_summary.csv')
    if summary is not None:
        main_df = summary.loc[summary['group'] != 'all'].copy()
        if len(main_df) > 0:
            fp = fig_dir / 'global_alignment_summary.png'
            plot_global_alignment_summary(main_df, fp, plot_cfg)
            created.append(str(fp))
        all_df = summary.loc[summary['group'] == 'all'].copy()
        if len(all_df) > 0:
            fp = fig_dir / 'global_alignment_summary_all.png'
            plot_global_alignment_summary(all_df, fp, plot_cfg)
            created.append(str(fp))
    vw = _safe_read_csv_rel(root, 'global_alignment/variablewise_global_alignment_summary.csv')
    if vw is not None:
        fp = fig_dir / 'variablewise_global_alignment_all.png'
        plot_variablewise_global_alignment_all(vw, fp, plot_cfg)
        created.append(str(fp))
        fp = fig_dir / 'variablewise_global_alignment_topn.png'
        plot_variablewise_global_alignment_topn(vw, fp, plot_cfg, top_n=12)
        created.append(str(fp))


def render_neighborhood(root: Path, fig_dir: Path, plot_cfg: PlotConfig, created: List[str]) -> None:
    print_banner('Rendering neighborhood figures')
    group_df = _safe_read_csv_rel(root, 'neighborhood/neighborhood_group_summary.csv')
    if group_df is not None:
        main_df = group_df.loc[group_df['group'] != 'all'].copy()
        if len(main_df) > 0:
            fp = fig_dir / 'neighborhood_group_summary.png'
            plot_neighborhood_summary(main_df, fp, plot_cfg)
            created.append(str(fp))
        all_df = group_df.loc[group_df['group'] == 'all'].copy()
        if len(all_df) > 0:
            fp = fig_dir / 'neighborhood_group_summary_all.png'
            plot_neighborhood_summary(all_df, fp, plot_cfg)
            created.append(str(fp))
    var_df = _safe_read_csv_rel(root, 'neighborhood/neighborhood_variable_summary.csv')
    if var_df is not None:
        fp = fig_dir / 'variable_level_neighbor_heatmap.png'
        plot_variable_level_neighbor_heatmap(var_df, fp, plot_cfg)
        created.append(str(fp))


def render_retrieval(root: Path, fig_dir: Path, plot_cfg: PlotConfig, created: List[str]) -> None:
    print_banner('Rendering retrieval figures')
    summary = _safe_read_csv_rel(root, 'retrieval/retrieval_summary.csv')
    if summary is not None:
        for fp in plot_retrieval_summary(summary, fig_dir, plot_cfg, stem_prefix='retrieval'):
            created.append(str(fp))


def render_adjusted(root: Path, fig_dir: Path, plot_cfg: PlotConfig, created: List[str]) -> None:
    print_banner('Rendering adjusted-robustness figures')
    summary = _safe_read_csv_rel(root, 'robustness/adjusted/adjusted_global_alignment_summary.csv')
    if summary is not None:
        main_df = summary.loc[summary['group'] != 'all'].copy()
        if len(main_df) > 0:
            fp = fig_dir / 'adjusted_global_alignment_summary.png'
            plot_global_alignment_summary(main_df, fp, plot_cfg)
            created.append(str(fp))
        all_df = summary.loc[summary['group'] == 'all'].copy()
        if len(all_df) > 0:
            fp = fig_dir / 'adjusted_global_alignment_summary_all.png'
            plot_global_alignment_summary(all_df, fp, plot_cfg)
            created.append(str(fp))


def render_bootstrap(root: Path, fig_dir: Path, plot_cfg: PlotConfig, created: List[str]) -> None:
    print_banner('Rendering bootstrap-robustness figures')
    g_raw = _safe_read_csv_rel(root, 'robustness/bootstrap/global_alignment/bootstrap_global_alignment_raw.csv')
    g_sum = _safe_read_csv_rel(root, 'robustness/bootstrap/global_alignment/bootstrap_global_alignment_summary.csv')
    if g_sum is not None:
        fp = fig_dir / 'bootstrap_global_alignment_forest.png'
        plot_global_alignment_forest(g_sum, fp, plot_cfg)
        created.append(str(fp))
    if g_raw is not None:
        fp = fig_dir / 'bootstrap_global_alignment_distribution.png'
        plot_bootstrap_global_distribution(g_raw, fp, plot_cfg)
        created.append(str(fp))
    n_sum = _safe_read_csv_rel(root, 'robustness/bootstrap/neighborhood/bootstrap_neighborhood_summary.csv')
    if n_sum is not None:
        fp = fig_dir / 'bootstrap_neighborhood_summary.png'
        plot_bootstrap_neighborhood_summary(n_sum, fp, plot_cfg)
        created.append(str(fp))
    r_sum = _safe_read_csv_rel(root, 'robustness/bootstrap/retrieval/bootstrap_retrieval_summary.csv')
    if r_sum is not None:
        cont = r_sum.loc[r_sum['metric'] == 'spearman'].rename(columns={'estimate_mean': 'spearman'})
        cls = r_sum.loc[r_sum['metric'] == 'balanced_accuracy'].rename(columns={'estimate_mean': 'balanced_accuracy'})
        combo = pd.concat([cont, cls], ignore_index=True, sort=False)
        for fp in plot_retrieval_summary(combo, fig_dir, plot_cfg, stem_prefix='bootstrap_retrieval'):
            created.append(str(fp))


def render_baselines(root: Path, fig_dir: Path, plot_cfg: PlotConfig, created: List[str]) -> None:
    print_banner('Rendering baseline-comparison figures')
    base = _safe_read_csv_rel(root, 'robustness/baselines/baseline_comparison_summary.csv')
    if base is not None:
        for fp in plot_baseline_comparison(base, fig_dir, plot_cfg):
            created.append(str(fp))
    comp = _safe_read_csv_rel(root, 'robustness/baselines/distance_space_comparison.csv')
    if comp is not None:
        fp = fig_dir / 'distance_space_pearson_heatmap.png'
        plot_distance_space_correlation_heatmap(comp, 'pearson_r', fp, plot_cfg)
        created.append(str(fp))
        fp = fig_dir / 'distance_space_spearman_heatmap.png'
        plot_distance_space_correlation_heatmap(comp, 'spearman_r', fp, plot_cfg)
        created.append(str(fp))


def render_discovery_validation(root: Path, fig_dir: Path, plot_cfg: PlotConfig, created: List[str]) -> None:
    print_banner('Rendering discovery-validation figures')
    summary = _safe_read_csv_rel(root, 'robustness/discovery_validation/discovery_validation_summary.csv')
    if summary is None:
        return
    for fp in plot_discovery_validation_selection_frequency(summary, fig_dir, plot_cfg):
        created.append(str(fp))
    fp = fig_dir / 'discovery_validation_global_spearman.png'
    plot_discovery_validation_paired(summary, fp, plot_cfg, 'global_spearman_best_group', 'Discovery-validation | global alignment', 'Spearman correlation')
    if fp.exists(): created.append(str(fp))
    fp = fig_dir / 'discovery_validation_neighborhood_improvement.png'
    plot_discovery_validation_paired(summary, fp, plot_cfg, 'neighborhood_improvement_best_k', 'Discovery-validation | neighborhood consistency', 'Improvement ratio')
    if fp.exists(): created.append(str(fp))
    fp = fig_dir / 'discovery_validation_retrieval_mean_spearman.png'
    plot_discovery_validation_paired(summary, fp, plot_cfg, 'retrieval_mean_spearman', 'Discovery-validation | retrieval auxiliary score', 'Mean retrieval Spearman')
    if fp.exists(): created.append(str(fp))




# ---- composite figure helpers ----

def _ax_global_alignment_bar(summary_df: pd.DataFrame, ax: plt.Axes, plot_cfg: PlotConfig, ylabel: str = 'Correlation') -> None:
    plot_df = summary_df.copy()
    plot_df = _apply_group_labels(plot_df)
    plot_df = plot_df.loc[plot_df['group'] != 'Integrated'].copy()
    order = [g for g in GROUP_ORDER if g in plot_df['group'].unique().tolist()]
    long_df = plot_df.melt(id_vars='group', value_vars=['pearson_r', 'spearman_r'], var_name='metric', value_name='value')
    long_df['metric'] = long_df['metric'].map({'pearson_r': 'Pearson r', 'spearman_r': 'Spearman r'})
    sns.barplot(data=long_df, x='group', y='value', hue='metric', order=order,
                palette=[METRIC_COLORS['Pearson r'], METRIC_COLORS['Spearman r']], ax=ax)
    ax.axhline(0.0, color='#7F7F7F', linestyle='--', linewidth=1.0)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='y')


def _ax_global_alignment_forest(summary_df: pd.DataFrame, ax: plt.Axes, plot_cfg: PlotConfig) -> None:
    plot_df = summary_df.copy()
    plot_df = plot_df.loc[plot_df['group'] != 'all'].copy()
    plot_df = _apply_group_labels(plot_df)
    plot_df['label'] = plot_df['group'].astype(str) + ' | ' + plot_df['metric'].map(_metric_display_name)
    group_rank = {'Functional': 0, 'Structural': 1, 'Burden': 2}
    metric_rank = {'pearson_r': 0, 'spearman_r': 1}
    plot_df['_gr'] = plot_df['group'].map(group_rank).fillna(99)
    plot_df['_mr'] = plot_df['metric'].map(metric_rank).fillna(99)
    plot_df = plot_df.sort_values(['_gr', '_mr']).reset_index(drop=True)
    y = np.arange(len(plot_df))[::-1]
    colors = {'pearson_r': '#158F8C', 'spearman_r': '#2F6DB3'}
    for yi, (_, row) in zip(y, plot_df.iterrows()):
        c = colors.get(str(row['metric']), '#4C4C4C')
        ax.plot([row['ci_lower'], row['ci_upper']], [yi, yi], color=c, linewidth=2.2)
        ax.scatter(row['estimate_mean'], yi, s=90, color=c, zorder=3)
    ax.axvline(0.0, linestyle='--', color='#7F7F7F', linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['label'].tolist())
    ax.set_xlabel('Estimate (mean with 95% CI)')
    ax.set_ylabel('')
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='x')


def _ax_variablewise_fdr_lollipop(df: pd.DataFrame, ax: plt.Axes, plot_cfg: PlotConfig, top_n: int = 10) -> None:
    plot_df = df.copy()
    plot_df = plot_df.sort_values('spearman_r', ascending=False).head(int(top_n)).copy()
    plot_df = _apply_group_labels(plot_df)
    plot_df = plot_df.sort_values('spearman_r', ascending=False).reset_index(drop=True)
    y_pos = np.arange(len(plot_df), dtype=float)
    for y, (_, row) in zip(y_pos, plot_df.iterrows()):
        ax.hlines(y, 0.0, row['spearman_r'], color='#C7C7C7', linewidth=2.0, zorder=1)
        marker = 'o' if bool(row.get('global_fdr_sig_overall', True)) else 'X'
        ax.scatter(row['spearman_r'], y, s=110, color=GROUP_COLORS.get(row['group'], '#4C4C4C'), marker=marker, zorder=3)
    ax.axvline(0.0, linestyle='--', color='#7F7F7F', linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['variable'].tolist())
    for tick, (_, row) in zip(ax.get_yticklabels(), plot_df.iterrows()):
        tick.set_color(GROUP_COLORS.get(row['group'], '#333333'))
    ax.invert_yaxis()
    ax.set_xlabel('Spearman correlation')
    ax.set_ylabel('')
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='x')


def _ax_variable_neighbor_bubble(retrieval_df: pd.DataFrame, neighbor_df: pd.DataFrame, fdr_df: Optional[pd.DataFrame], ax: plt.Axes, plot_cfg: PlotConfig, top_n: int = 10) -> None:
    cont = retrieval_df.loc[retrieval_df['target_type'].isin(['continuous', 'ordinal'])].copy()
    if len(cont) == 0:
        ax.axis('off')
        return
    target_order = cont.groupby('target', dropna=False)['spearman'].max().sort_values(ascending=False).head(int(top_n)).index.tolist()
    cont = cont.loc[cont['target'].isin(target_order)].copy()
    nb = neighbor_df.copy().rename(columns={'variable': 'target'})
    bubble = cont.merge(nb[['target', 'k', 'improvement_ratio']], on=['target', 'k'], how='left')
    if fdr_df is not None and len(fdr_df) > 0:
        fdr_small = fdr_df[['variable', 'global_fdr_sig_overall']].drop_duplicates().rename(columns={'variable': 'target'})
        bubble = bubble.merge(fdr_small, on='target', how='left')
    else:
        bubble['global_fdr_sig_overall'] = True
    bubble['target'] = pd.Categorical(bubble['target'], categories=target_order[::-1], ordered=True)
    bubble = bubble.sort_values(['target', 'k'])
    size_src = bubble['improvement_ratio'].fillna(1.0)
    sizes = np.interp(size_src.to_numpy(), (float(size_src.min()), float(size_src.max()) if float(size_src.max()) > float(size_src.min()) else float(size_src.min()) + 1e-6), (70, 360))
    sc = ax.scatter(bubble['k'], bubble['target'].astype(str), c=bubble['spearman'], s=sizes,
                    cmap=sns.blend_palette(['#EFF3FF', '#3182BD', '#08306B'], as_cmap=True),
                    edgecolor=['black' if bool(x) else '#333333' for x in bubble['global_fdr_sig_overall'].fillna(True)],
                    linewidth=[1.0 if bool(x) else 1.6 for x in bubble['global_fdr_sig_overall'].fillna(True)],
                    marker='o')
    ax.set_xlabel('Top-k acoustic neighbors')
    ax.set_ylabel('')
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='both')
    cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label('Retrieval Spearman')
    # size legend
    imp_vals = [1.08, 1.10, 1.12]
    handles = [plt.scatter([], [], s=np.interp(v, (float(size_src.min()), float(size_src.max()) if float(size_src.max()) > float(size_src.min()) else float(size_src.min()) + 1e-6), (70, 360)), color='#9A9A9A') for v in imp_vals]
    leg = ax.legend(handles, [f'{v:.2f}' for v in imp_vals], title='Improvement ratio', loc='upper left', bbox_to_anchor=(1.12, 0.98), borderaxespad=0.0)
    ax.add_artist(leg)


def _ax_discovery_validation_paired(summary_df: pd.DataFrame, ax: plt.Axes, plot_cfg: PlotConfig, metric_col: str, ylabel: str) -> None:
    pivot = summary_df.pivot(index='split_id', columns='subset', values=metric_col)
    if not {'discovery', 'validation'}.issubset(set(pivot.columns)):
        ax.axis('off')
        return
    for _, row in pivot.iterrows():
        ax.plot([0, 1], [row['discovery'], row['validation']], color='#B7B7B7', linewidth=1.5, zorder=1)
    means = pivot[['discovery', 'validation']].mean(axis=0)
    ax.scatter([0, 1], [means['discovery'], means['validation']], s=180, color='#2F6DB3', zorder=3)
    ax.plot([0, 1], [means['discovery'], means['validation']], color='#4C4C4C', linewidth=2.3, zorder=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Discovery', 'Validation'])
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='y')


def _ax_baseline_cleveland(baseline_df: pd.DataFrame, ax: plt.Axes, plot_cfg: PlotConfig) -> None:
    plot_df = baseline_df.loc[baseline_df['analysis'] == 'global_alignment'].copy()
    plot_df = plot_df.loc[plot_df['group'].isin(['function', 'structure', 'burden'])].copy()
    plot_df = _apply_group_labels(plot_df)
    group_order = [g for g in GROUP_ORDER if g in plot_df['group'].unique()]
    spaces = plot_df['space'].dropna().unique().tolist()
    palette = {sp: SPACE_COLORS[i % len(SPACE_COLORS)] for i, sp in enumerate(spaces)}
    y = np.arange(len(group_order))
    for yi, group in zip(y, group_order):
        sub = plot_df.loc[plot_df['group'] == group].sort_values('spearman_r')
        if len(sub) >= 2:
            ax.plot(sub['spearman_r'], [yi] * len(sub), color='#C7C7C7', linewidth=2.0, zorder=1)
        for _, row in sub.iterrows():
            ax.scatter(row['spearman_r'], yi, s=120, color=palette.get(row['space'], '#4C4C4C'), zorder=3)
    ax.axvline(0.0, linestyle='--', color='#7F7F7F', linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(group_order)
    ax.set_xlabel('Spearman correlation')
    ax.set_ylabel('')
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='x')
    handles = [Line2D([0], [0], marker='o', linestyle='None', color=palette[sp], markersize=8) for sp in spaces]
    ax.legend(handles, spaces, loc='upper left', bbox_to_anchor=(0.0, 1.18), ncol=min(len(spaces), 4), frameon=False)


def _ax_distance_space_heatmap(comparison_df: pd.DataFrame, ax: plt.Axes, plot_cfg: PlotConfig, metric: str = 'spearman_r') -> None:
    pivot = comparison_df.pivot(index='space_i', columns='space_j', values=metric)
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap=sns.blend_palette(['#F7FBFF', '#9ECAE1', '#3182BD', '#08519C'], as_cmap=True),
                vmin=-1, vmax=1, square=True,
                linewidths=_cfg_attr(plot_cfg, 'heatmap_grid_linewidth', 0.8),
                linecolor=_cfg_attr(plot_cfg, 'heatmap_grid_color', '#F3F3F3'),
                cbar_kws={'label': metric}, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    _format_heatmap_xticklabels(ax)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False)


def render_composites(root: Path, fig_dir: Path, plot_cfg: PlotConfig, created: List[str]) -> None:
    print_banner('Rendering composite figures')
    set_publication_plot_style(plot_cfg)
    # Figure 1
    gsum = _safe_read_csv_rel(root, 'global_alignment/global_alignment_summary.csv')
    bsum = _safe_read_csv_rel(root, 'robustness/bootstrap/global_alignment/bootstrap_global_alignment_summary.csv')
    adj = _safe_read_csv_rel(root, 'robustness/adjusted/adjusted_global_alignment_summary.csv')
    if gsum is not None and bsum is not None and adj is not None:
        gsum = gsum.loc[gsum['group'] != 'all'].copy()
        bsum = bsum.loc[bsum['group'] != 'all'].copy()
        adj = adj.loc[adj['group'] != 'all'].copy()
        fig, axes = plt.subplots(1, 3, figsize=(24, 6.5), gridspec_kw={'width_ratios': [1.05, 1.7, 1.05]})
        _ax_global_alignment_bar(gsum, axes[0], plot_cfg, ylabel='Correlation')
        _panel_label(axes[0], 'A')
        _ax_global_alignment_forest(bsum, axes[1], plot_cfg)
        _panel_label(axes[1], 'B')
        _ax_global_alignment_bar(adj, axes[2], plot_cfg, ylabel='Adjusted correlation')
        _panel_label(axes[2], 'C')
        handles = [Line2D([0],[0], marker='s', linestyle='None', color=METRIC_COLORS['Pearson r'], markersize=10),
                   Line2D([0],[0], marker='s', linestyle='None', color=METRIC_COLORS['Spearman r'], markersize=10)]
        fig.legend(handles, ['Pearson r', 'Spearman r'], loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
        fig.subplots_adjust(top=0.86, wspace=0.35)
        fp = fig_dir / 'figure_01_global_alignment_composite.png'
        _save_fig(fig, fp, dpi=plot_cfg.dpi)
        created.append(str(fp))

    # Figure 2
    fdr = _safe_read_csv_rel(root, 'interpretability/variablewise_consolidation/variablewise_global_main_table.csv')
    retr = _safe_read_csv_rel(root, 'retrieval/retrieval_summary.csv')
    neigh = _safe_read_csv_rel(root, 'neighborhood/neighborhood_variable_summary.csv')
    if fdr is not None and retr is not None and neigh is not None:
        fig, axes = plt.subplots(1, 2, figsize=(24, 7.2), gridspec_kw={'width_ratios': [1.25, 1.0]})
        _ax_variablewise_fdr_lollipop(fdr, axes[0], plot_cfg, top_n=12)
        _panel_label(axes[0], 'A')
        _ax_variable_neighbor_bubble(retr, neigh, fdr, axes[1], plot_cfg, top_n=10)
        _panel_label(axes[1], 'B')
        handles = [Line2D([0],[0], marker='o', linestyle='None', color=GROUP_COLORS[g], markersize=8) for g in GROUP_ORDER]
        labels = GROUP_ORDER.copy()
        handles += [Line2D([0],[0], marker='o', linestyle='None', color='black', markersize=8),
                    Line2D([0],[0], marker='X', linestyle='None', color='black', markersize=8)]
        labels += ['FDR significant', 'Not FDR significant']
        fig.legend(handles, labels, loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03))
        fig.subplots_adjust(top=0.86, wspace=0.34)
        fp = fig_dir / 'figure_02_variable_interpretation_composite.png'
        _save_fig(fig, fp, dpi=plot_cfg.dpi)
        created.append(str(fp))

    # Figure 3
    dv = _safe_read_csv_rel(root, 'robustness/discovery_validation/discovery_validation_summary.csv')
    base = _safe_read_csv_rel(root, 'robustness/baselines/baseline_comparison_summary.csv')
    comp = _safe_read_csv_rel(root, 'robustness/baselines/distance_space_comparison.csv')
    if dv is not None and base is not None and comp is not None:
        fig, axes = plt.subplots(1, 3, figsize=(24, 6.6), gridspec_kw={'width_ratios': [1.2, 1.25, 1.15]})
        _ax_discovery_validation_paired(dv, axes[0], plot_cfg, 'global_spearman_best_group', 'Spearman correlation')
        _panel_label(axes[0], 'A')
        _ax_baseline_cleveland(base, axes[1], plot_cfg)
        _panel_label(axes[1], 'B')
        _ax_distance_space_heatmap(comp, axes[2], plot_cfg, metric='spearman_r')
        _panel_label(axes[2], 'C')
        fig.subplots_adjust(top=0.86, wspace=0.42)
        fp = fig_dir / 'figure_03_robustness_composite.png'
        _save_fig(fig, fp, dpi=plot_cfg.dpi)
        created.append(str(fp))

def render_all(root: Path, fig_dir: Path, plot_cfg: PlotConfig) -> List[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    created: List[str] = []
    render_global(root, fig_dir, plot_cfg, created)
    render_neighborhood(root, fig_dir, plot_cfg, created)
    render_retrieval(root, fig_dir, plot_cfg, created)
    render_adjusted(root, fig_dir, plot_cfg, created)
    render_bootstrap(root, fig_dir, plot_cfg, created)
    render_baselines(root, fig_dir, plot_cfg, created)
    render_discovery_validation(root, fig_dir, plot_cfg, created)
    render_composites(root, fig_dir, plot_cfg, created)
    manifest = {'root': str(root), 'figure_dir': str(fig_dir), 'n_figures': len(created), 'files': created}
    with (fig_dir / 'plot_manifest.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log_done(f'Finished rendering {len(created)} figures into: {fig_dir}')
    return created


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Redraw all clinical-alignment figures from saved result tables / matrices.')
    p.add_argument('--out-root', type=str, required=True, help='Clinical_alignment output root containing prepared/, global_alignment/, retrieval/, robustness/ ...')
    p.add_argument('--fig-dir', type=str, default=None, help='Single folder to save all regenerated figures. Default: <out-root>/all_redrawn_figures')
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = build_default_config()
    plot_cfg = cfg.plot
    root = Path(args.out_root)
    fig_dir = Path(args.fig_dir) if args.fig_dir else root / 'all_redrawn_figures'
    render_all(root, fig_dir, plot_cfg)


if __name__ == '__main__':
    main()
