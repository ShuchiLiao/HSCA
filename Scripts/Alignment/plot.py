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

try:
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
        palette: List[str] = field(default_factory=lambda: ['#158F8C','#2F6DB3','#E38B2C','#7B5AA6','#5FB0B7','#F1B44C','#C75C5C'])
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
        x = X(); x.plot = PlotConfig(); return x
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
        line = '=' * 96
        print(f'\n{line}\n{title}\n{line}')


DISPLAY_GROUP = {
    'function': 'Functional',
    'structure': 'Structural',
    'burden': 'Burden',
    'all': 'Integrated',
}
GROUP_ORDER = ['Functional', 'Structural', 'Burden']
POSITION_ORDER = ['A', 'E', 'M', 'P', 'T']


def _cfg_attr(plot_cfg: PlotConfig, name: str, default):
    return getattr(plot_cfg, name, default)


def _display_group_value(x: object) -> str:
    return DISPLAY_GROUP.get(str(x), str(x))


def _apply_group_labels(df: pd.DataFrame, col: str = 'group') -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = out[col].map(_display_group_value)
    return out


def _apply_axes_style(ax: plt.Axes, plot_cfg: Optional[PlotConfig] = None, show_grid: bool = False, grid_axis: str = 'both') -> None:
    plot_cfg = PlotConfig() if plot_cfg is None else plot_cfg
    if show_grid:
        ax.grid(True, axis=grid_axis, alpha=_cfg_attr(plot_cfg, 'axis_grid_alpha', 0.22), linewidth=_cfg_attr(plot_cfg, 'axis_grid_linewidth', 0.9))
    else:
        ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.4)


def _legend_outside(ax: plt.Axes, right: float = 0.78) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) == 0:
        return
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l in uniq_l or l.startswith('_'):
            continue
        uniq_h.append(h)
        uniq_l.append(l)
    ax.legend(uniq_h, uniq_l, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    try:
        ax.figure.subplots_adjust(right=right)
    except Exception:
        pass


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
        rows.append({'x_mid': float((left + right)/2), 'y_med': float(np.nanmedian(ys)), 'y_lo': float(np.nanquantile(ys,0.25)), 'y_hi': float(np.nanquantile(ys,0.75))})
    return pd.DataFrame(rows)


def _metric_display_name(metric: str) -> str:
    return {'pearson_r':'Pearson r','spearman_r':'Spearman r','improvement_ratio':'Improvement ratio','balanced_accuracy':'Balanced accuracy','spearman':'Spearman'}.get(metric, metric)


def _is_numeric_like(text: str) -> bool:
    try:
        float(str(text).replace(',', ''))
        return True
    except Exception:
        return False


def _format_heatmap_xticklabels(ax: plt.Axes) -> None:
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    rotate = any((lab.strip() != '' and not _is_numeric_like(lab)) for lab in labels)
    ax.set_xticklabels(labels, rotation=45 if rotate else 0, ha='right' if rotate else 'center')
    ylabels = [tick.get_text() for tick in ax.get_yticklabels()]
    ax.set_yticklabels(ylabels, rotation=0)


def plot_global_alignment_scatter(acoustic_distance: np.ndarray, clinical_distance: np.ndarray, group_name: str, out_path: str | Path, plot_cfg: PlotConfig, seed: int = 42) -> None:
    set_publication_plot_style(plot_cfg)
    x = upper_triangle_vector(_ensure_square_distance_matrix(acoustic_distance))
    y = upper_triangle_vector(_ensure_square_distance_matrix(clinical_distance))
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    x_s, y_s = _sample_for_scatter(x, y, max_points=_cfg_attr(plot_cfg,'global_scatter_max_points',60000), seed=seed)
    fig, ax = plt.subplots(figsize=(9.6, 8.0))
    cmap = sns.blend_palette(['#F1B44C','#158F8C','#2F6DB3','#7B5AA6'], as_cmap=True)
    hb = ax.hexbin(x_s, y_s, gridsize=46, cmap=cmap, mincnt=1)
    trend = _binned_trend(x_s, y_s, n_bins=24)
    if len(trend) > 0:
        ax.fill_between(trend['x_mid'], trend['y_lo'], trend['y_hi'], color='#C75C5C', alpha=0.18, linewidth=0)
        ax.plot(trend['x_mid'], trend['y_med'], color='#C75C5C', linewidth=2.5)
    cbar = fig.colorbar(hb, ax=ax, shrink=0.82); cbar.set_label('Pair count')
    ax.set_xlabel('Acoustic distance'); ax.set_ylabel('Clinical distance')
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False)
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_global_alignment_summary(summary_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig) -> None:
    if len(summary_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = _apply_group_labels(summary_df[['group','pearson_r','spearman_r']].copy())
    group_cat = pd.Categorical(plot_df['group'], categories=GROUP_ORDER + ['Integrated'], ordered=True)
    plot_df = plot_df.assign(group_cat=group_cat).sort_values('group_cat', ascending=True).drop(columns='group_cat').reset_index(drop=True)
    y_pos = np.arange(len(plot_df), dtype=float)
    fig, ax = plt.subplots(figsize=(10.0, max(4.8, 1.2 * len(plot_df))))
    for y, (_, row) in zip(y_pos, plot_df.iterrows()):
        ax.plot([row['pearson_r'], row['spearman_r']], [y, y], color='#B7B7B7', linewidth=2.0, zorder=1)
    ax.scatter(plot_df['pearson_r'], y_pos, s=130, color='#158F8C', label='Pearson r', zorder=3)
    ax.scatter(plot_df['spearman_r'], y_pos, s=130, color='#2F6DB3', label='Spearman r', zorder=3)
    ax.axvline(0.0, color='#7F7F7F', linestyle='--', linewidth=1.2)
    ax.set_yticks(y_pos); ax.set_yticklabels(plot_df['group'].tolist())
    ax.set_xlabel('Correlation'); ax.set_ylabel('')
    _legend_outside(ax, right=0.82)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='x')
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_global_alignment_bar(summary_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig) -> None:
    if len(summary_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = _apply_group_labels(summary_df[['group','pearson_r','spearman_r']].copy())
    plot_df = plot_df.melt(id_vars='group', value_vars=['pearson_r','spearman_r'], var_name='metric', value_name='value')
    plot_df['metric'] = plot_df['metric'].map({'pearson_r':'Pearson r','spearman_r':'Spearman r'})
    order = [g for g in GROUP_ORDER if g in plot_df['group'].unique().tolist()] + [g for g in ['Integrated'] if g in plot_df['group'].unique().tolist()]
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    sns.barplot(data=plot_df, x='group', y='value', hue='metric', order=order, ax=ax)
    ax.axhline(0.0, color='#7F7F7F', linestyle='--', linewidth=1.2)
    ax.set_xlabel(''); ax.set_ylabel('Correlation')
    _legend_outside(ax, right=0.82)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='y')
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_global_alignment_forest(summary_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig) -> None:
    if len(summary_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = summary_df.loc[summary_df['group'] != 'all'].copy()
    plot_df = _apply_group_labels(plot_df)
    plot_df['label'] = plot_df['group'].astype(str) + ' | ' + plot_df['metric'].map(_metric_display_name)
    plot_df = plot_df.sort_values(['group','metric']).reset_index(drop=True)
    y = np.arange(len(plot_df))[::-1]
    fig, ax = plt.subplots(figsize=(10.8, max(5.6, 0.7 * len(plot_df))))
    colors = {'pearson_r':'#158F8C','spearman_r':'#2F6DB3','improvement_ratio':'#E38B2C','balanced_accuracy':'#7B5AA6','spearman':'#C75C5C'}
    for yi, (_, row) in zip(y, plot_df.iterrows()):
        ax.plot([row['ci_lower'], row['ci_upper']], [yi, yi], color=colors.get(str(row['metric']), '#4C4C4C'), linewidth=2.6)
        ax.scatter(row['estimate_mean'], yi, s=110, color=colors.get(str(row['metric']), '#4C4C4C'), zorder=3)
    ax.axvline(0.0, linestyle='--', color='#7F7F7F', linewidth=1.2)
    ax.set_yticks(y); ax.set_yticklabels(plot_df['label'].tolist())
    ax.set_xlabel('Estimate (mean with bootstrap CI)'); ax.set_ylabel('')
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='x')
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_bootstrap_global_distribution(raw_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig) -> None:
    raw_df = raw_df.loc[raw_df['group'] != 'all'].copy()
    if len(raw_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = raw_df.melt(id_vars=['group','repeat'], value_vars=['pearson_r','spearman_r'], var_name='metric', value_name='value')
    plot_df = _apply_group_labels(plot_df)
    order = [g for g in GROUP_ORDER if g in plot_df['group'].unique().tolist()]
    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    sns.violinplot(data=plot_df, x='group', y='value', hue='metric', order=order, inner=None, cut=0, linewidth=0, ax=ax)
    sns.boxplot(data=plot_df, x='group', y='value', hue='metric', order=order, showcaps=True, boxprops={'facecolor':'none','edgecolor':'black'}, whiskerprops={'linewidth':1.0}, medianprops={'color':'black'}, showfliers=False, width=0.28, ax=ax)
    sample_df = plot_df.sample(min(len(plot_df), 1200), random_state=42) if len(plot_df) > 0 else plot_df
    sns.stripplot(data=sample_df, x='group', y='value', hue='metric', order=order, dodge=True, alpha=0.18, size=3.0, color='black', ax=ax)
    handles, labels = ax.get_legend_handles_labels(); ax.legend(handles[:2], ['Pearson r','Spearman r'], loc='upper left', bbox_to_anchor=(1.02,1.0), borderaxespad=0.0)
    ax.set_xlabel(''); ax.set_ylabel('Bootstrap estimate')
    ax.axhline(0.0, linestyle='--', color='#7F7F7F', linewidth=1.2)
    ax.figure.subplots_adjust(right=0.82)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='y')
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def _plot_variable_lollipop(df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig, top_n: Optional[int] = None, fdr_col: Optional[str] = None) -> None:
    if len(df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = df.copy()
    if top_n is not None:
        plot_df = plot_df.sort_values(['spearman_r','coverage'], ascending=[False,False]).head(int(top_n)).copy()
    plot_df = _apply_group_labels(plot_df)
    plot_df = plot_df.sort_values('spearman_r', ascending=False).reset_index(drop=True)
    y_pos = np.arange(len(plot_df), dtype=float)
    fig, ax = plt.subplots(figsize=(11.8, max(6.0, 0.42 * len(plot_df))))
    group_colors = {'Functional':'#158F8C','Structural':'#2F6DB3','Burden':'#E38B2C','Integrated':'#7B5AA6'}
    ax.hlines(y=y_pos, xmin=0.0, xmax=plot_df['spearman_r'], color='#D0D0D0', linewidth=1.8, zorder=1)
    marker_map = None
    if fdr_col is not None and fdr_col in plot_df.columns:
        marker_map = plot_df[fdr_col].map({True:'o', False:'X'}).fillna('o').tolist()
    for i, (_, row) in enumerate(plot_df.iterrows()):
        marker = marker_map[i] if marker_map is not None else 'o'
        ax.scatter(row['spearman_r'], y_pos[i], s=120, color=group_colors.get(row['group'], '#4C4C4C'), marker=marker, zorder=3)
    ax.axvline(0.0, linestyle='--', color='#7F7F7F', linewidth=1.2)
    ax.set_yticks(y_pos); ax.set_yticklabels(plot_df['variable'].tolist())
    ax.invert_yaxis()
    ax.set_xlabel('Spearman correlation'); ax.set_ylabel('')
    # manual legend
    handles = []
    labels = []
    shown = set()
    for g, c in group_colors.items():
        if g in plot_df['group'].unique() and g not in shown:
            handles.append(plt.Line2D([0],[0], marker='o', linestyle='None', color=c, markersize=8))
            labels.append(g)
            shown.add(g)
    if marker_map is not None:
        handles.extend([
            plt.Line2D([0],[0], marker='o', linestyle='None', color='black', markersize=8),
            plt.Line2D([0],[0], marker='X', linestyle='None', color='black', markersize=8),
        ])
        labels.extend(['FDR significant', 'Not FDR significant'])
    if handles:
        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02,1.0), borderaxespad=0.0)
        ax.figure.subplots_adjust(right=0.80)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='x')
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_variablewise_global_alignment_topn(summary_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig, top_n: int = 12) -> None:
    _plot_variable_lollipop(summary_df, out_path, plot_cfg, top_n=top_n)


def plot_variablewise_global_alignment_all(summary_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig) -> None:
    _plot_variable_lollipop(summary_df, out_path, plot_cfg, top_n=None)


def plot_neighborhood_summary(summary_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig) -> None:
    if len(summary_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = _apply_group_labels(summary_df.copy())
    plot_df = plot_df.loc[plot_df['group'] != 'Integrated'].copy()
    order = [g for g in GROUP_ORDER if g in plot_df['group'].unique().tolist()]
    fig, ax = plt.subplots(figsize=(10.2, 6.8))
    sns.barplot(data=plot_df, x='k', y='improvement_ratio', hue='group', hue_order=order, ax=ax)
    ax.axhline(1.0, linestyle='--', color='#7F7F7F', linewidth=1.2)
    ax.set_xlabel('Top-k acoustic neighbors'); ax.set_ylabel('Improvement ratio vs random')
    _legend_outside(ax, right=0.82)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='y')
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_bootstrap_neighborhood_summary(summary_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig) -> None:
    summary_df = summary_df.loc[summary_df['group'] != 'all'].copy()
    if len(summary_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = _apply_group_labels(summary_df)
    order = [g for g in GROUP_ORDER if g in plot_df['group'].unique().tolist()]
    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    palette = {'Functional':'#158F8C','Structural':'#2F6DB3','Burden':'#E38B2C'}
    x_levels = sorted(plot_df['k'].dropna().unique().tolist())
    offsets = np.linspace(-0.24, 0.24, num=max(len(order),1))
    for off, group in zip(offsets, order):
        sub = plot_df.loc[plot_df['group'] == group].sort_values('k')
        xs = np.array([x_levels.index(k) for k in sub['k']]) + off
        ax.errorbar(xs, sub['estimate_mean'], yerr=[sub['estimate_mean'] - sub['ci_lower'], sub['ci_upper'] - sub['estimate_mean']], fmt='o', color=palette.get(group, '#4C4C4C'), capsize=4, linewidth=1.8, label=group)
    ax.axhline(1.0, linestyle='--', color='#7F7F7F', linewidth=1.2)
    ax.set_xticks(range(len(x_levels))); ax.set_xticklabels([str(k) for k in x_levels])
    ax.set_xlabel('Top-k acoustic neighbors'); ax.set_ylabel('Improvement ratio')
    _legend_outside(ax, right=0.82)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='y')
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_variable_level_neighbor_heatmap(variable_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig) -> None:
    if len(variable_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    ranked = variable_df.groupby('variable', dropna=False)['improvement_ratio'].mean().sort_values(ascending=False).head(int(_cfg_attr(plot_cfg,'neighborhood_heatmap_top_n',18))).index.tolist()
    plot_df = variable_df.loc[variable_df['variable'].isin(ranked)].copy()
    pivot = plot_df.pivot_table(index='variable', columns='k', values='improvement_ratio', aggfunc='mean').reindex(ranked)
    fig, ax = plt.subplots(figsize=(10.5, max(6.5, 0.48 * len(pivot))))
    cmap = sns.blend_palette(['#F1B44C','#158F8C','#2F6DB3','#7B5AA6'], as_cmap=True)
    sns.heatmap(pivot, cmap=cmap, ax=ax, linewidths=_cfg_attr(plot_cfg,'heatmap_grid_linewidth',0.8), linecolor=_cfg_attr(plot_cfg,'heatmap_grid_color','#F3F3F3'), annot=_cfg_attr(plot_cfg,'heatmap_annot',True), fmt='.2f', cbar_kws={'label':'Improvement ratio'})
    ax.set_xlabel('Top-k acoustic neighbors'); ax.set_ylabel('')
    _format_heatmap_xticklabels(ax)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False)
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def _plot_metric_heatmap(df: pd.DataFrame, value_col: str, out_path: str | Path, plot_cfg: PlotConfig, top_n: int, annot: bool = True) -> None:
    if len(df) == 0 or value_col not in df.columns:
        return
    ranked = df.groupby('target', dropna=False)[value_col].max().sort_values(ascending=False).head(int(top_n)).index.tolist()
    plot_df = df.loc[df['target'].isin(ranked)].copy()
    pivot = plot_df.pivot_table(index='target', columns='k', values=value_col, aggfunc='mean').reindex(ranked)
    fig, ax = plt.subplots(figsize=(9.8, max(6.0, 0.5 * len(pivot))))
    cmap = sns.blend_palette(['#F7FBFF','#9ECAE1','#3182BD','#08519C'], as_cmap=True)
    sns.heatmap(pivot, annot=annot, fmt='.2f', cmap=cmap, linewidths=_cfg_attr(plot_cfg,'heatmap_grid_linewidth',0.8), linecolor=_cfg_attr(plot_cfg,'heatmap_grid_color','#F3F3F3'), cbar_kws={'label':_metric_display_name(value_col)}, ax=ax)
    ax.set_xlabel('Top-k acoustic neighbors'); ax.set_ylabel('')
    _format_heatmap_xticklabels(ax)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False)
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_retrieval_summary(summary_df: pd.DataFrame, out_dir: str | Path, plot_cfg: PlotConfig, stem_prefix: str='retrieval') -> List[Path]:
    created: List[Path] = []
    if len(summary_df) == 0:
        return created
    set_publication_plot_style(plot_cfg); out_dir = Path(out_dir)
    cont_df = summary_df.loc[summary_df['target_type'].isin(['continuous','ordinal'])].copy()
    if len(cont_df) > 0 and 'spearman' in cont_df.columns:
        fp = out_dir / f'{stem_prefix}_continuous_heatmap.png'
        _plot_metric_heatmap(cont_df, 'spearman', fp, plot_cfg, _cfg_attr(plot_cfg,'bootstrap_retrieval_top_n',12) if stem_prefix.startswith('bootstrap') else _cfg_attr(plot_cfg,'retrieval_heatmap_top_n_continuous',12), annot=True)
        created.append(fp)
    cls_df = summary_df.loc[summary_df['target_type'].isin(['binary','categorical'])].copy()
    if len(cls_df) > 0 and 'balanced_accuracy' in cls_df.columns:
        fp = out_dir / f'{stem_prefix}_categorical_heatmap.png'
        _plot_metric_heatmap(cls_df, 'balanced_accuracy', fp, plot_cfg, _cfg_attr(plot_cfg,'retrieval_heatmap_top_n_categorical',8), annot=True)
        created.append(fp)
    return created


def plot_baseline_comparison(baseline_df: pd.DataFrame, out_dir: str | Path, plot_cfg: PlotConfig) -> List[Path]:
    created: List[Path] = []
    if len(baseline_df) == 0:
        return created
    set_publication_plot_style(plot_cfg)
    plot_df = baseline_df.loc[baseline_df['analysis']=='global_alignment'].copy(); out_dir = Path(out_dir)
    if len(plot_df) == 0:
        return created
    plot_df = _apply_group_labels(plot_df)
    main_df = plot_df.loc[plot_df['group']!='Integrated'].copy()
    order = [g for g in GROUP_ORDER if g in main_df['group'].unique().tolist()]
    if len(main_df) > 0:
        fig, ax = plt.subplots(figsize=(11,7.0))
        sns.barplot(data=main_df, x='group', y='spearman_r', hue='space', order=order, ax=ax)
        ax.set_xlabel(''); ax.set_ylabel('Spearman correlation')
        _legend_outside(ax, right=0.80)
        _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='y')
        fp = out_dir/'baseline_comparison_global_alignment.png'; _save_fig(fig, fp, dpi=plot_cfg.dpi); created.append(fp)
    return created


def plot_distance_space_correlation_heatmap(comparison_df: pd.DataFrame, metric: str, out_path: str | Path, plot_cfg: Optional[PlotConfig] = None) -> None:
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True); set_publication_plot_style(plot_cfg); plot_cfg = PlotConfig() if plot_cfg is None else plot_cfg
    pivot = comparison_df.pivot(index='space_i', columns='space_j', values=metric)
    fig, ax = plt.subplots(figsize=(8.6,7.6)); cmap = sns.blend_palette(['#F7FBFF','#9ECAE1','#3182BD','#08519C'], as_cmap=True)
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap=cmap, vmin=-1, vmax=1, square=True, linewidths=_cfg_attr(plot_cfg,'heatmap_grid_linewidth',0.8), linecolor=_cfg_attr(plot_cfg,'heatmap_grid_color','#F3F3F3'), cbar_kws={'shrink':0.85,'label':metric}, ax=ax)
    ax.set_xlabel(''); ax.set_ylabel('')
    _format_heatmap_xticklabels(ax)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False)
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_discovery_validation_selection_frequency(summary_df: pd.DataFrame, out_dir: str | Path, plot_cfg: PlotConfig) -> List[Path]:
    created: List[Path] = []
    if len(summary_df) == 0:
        return created
    set_publication_plot_style(plot_cfg); disc = summary_df.loc[summary_df['subset']=='discovery'].copy(); out_dir = Path(out_dir)
    if len(disc) == 0:
        return created
    disc = _apply_group_labels(disc, col='best_group_from_discovery')
    fig, ax = plt.subplots(figsize=(8.0,5.6))
    freq = disc['best_group_from_discovery'].value_counts().rename_axis('group').reset_index(name='count')
    ax.bar(freq['group'].astype(str), freq['count'].to_numpy())
    ax.set_xlabel(''); ax.set_ylabel('Count across splits')
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='y')
    fp = out_dir/'discovery_validation_best_group_frequency.png'; _save_fig(fig, fp, dpi=plot_cfg.dpi); created.append(fp)
    fig, ax = plt.subplots(figsize=(6.8,5.6))
    kfreq = disc['best_k_from_discovery'].value_counts().rename_axis('k').reset_index(name='count')
    ax.bar(kfreq['k'].astype(str), kfreq['count'].to_numpy())
    ax.set_xlabel(''); ax.set_ylabel('Count across splits')
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='y')
    fp = out_dir/'discovery_validation_best_k_frequency.png'; _save_fig(fig, fp, dpi=plot_cfg.dpi); created.append(fp)
    return created


def plot_discovery_validation_paired(summary_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig, metric_col: str, ylabel: str) -> None:
    if len(summary_df) == 0 or metric_col not in summary_df.columns:
        return
    set_publication_plot_style(plot_cfg); pivot = summary_df.pivot(index='split_id', columns='subset', values=metric_col)
    if not {'discovery','validation'}.issubset(set(pivot.columns)):
        return
    fig, ax = plt.subplots(figsize=(8.0,6.4))
    for _, row in pivot.iterrows():
        ax.plot([0,1],[row['discovery'],row['validation']], color='#B7B7B7', linewidth=1.8, zorder=1)
        ax.scatter([0,1],[row['discovery'],row['validation']], s=95, zorder=3)
    ax.set_xticks([0,1]); ax.set_xticklabels(['Discovery','Validation'])
    ax.set_xlabel(''); ax.set_ylabel(ylabel)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='y'); _save_fig(fig, out_path, dpi=plot_cfg.dpi)

# ---- interpretability plots ----

def plot_variablewise_fdr_topn(df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig, top_n: int = 12) -> None:
    _plot_variable_lollipop(df, out_path, plot_cfg, top_n=top_n, fdr_col='global_fdr_sig_overall')


def plot_position_contribution_global_heatmap(df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig) -> None:
    if len(df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = _apply_group_labels(df)
    pivot = plot_df.pivot(index='dropped_position', columns='group', values='drop_from_reference_spearman_r')
    pivot = pivot.reindex(index=[p for p in POSITION_ORDER if p in pivot.index], columns=[g for g in GROUP_ORDER if g in pivot.columns])
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    cmap = sns.blend_palette(['#F7FBFF','#9ECAE1','#3182BD','#08519C'], as_cmap=True)
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap, linewidths=_cfg_attr(plot_cfg,'heatmap_grid_linewidth',0.8), linecolor=_cfg_attr(plot_cfg,'heatmap_grid_color','#F3F3F3'), cbar_kws={'label':'Drop in Spearman r'}, ax=ax)
    ax.set_xlabel(''); ax.set_ylabel('Dropped position')
    _format_heatmap_xticklabels(ax)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False); _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_position_contribution_neighborhood_heatmap(df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig) -> None:
    if len(df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = _apply_group_labels(df)
    plot_df = plot_df.loc[plot_df['group'] != 'Integrated'].copy()
    plot_df['group_k'] = plot_df['group'].astype(str) + ' | k=' + plot_df['k'].astype(str)
    desired_cols = []
    for g in GROUP_ORDER:
        for k in sorted(plot_df['k'].dropna().unique().tolist()):
            desired_cols.append(f'{g} | k={k}')
    pivot = plot_df.pivot(index='dropped_position', columns='group_k', values='drop_from_reference_improvement_ratio')
    pivot = pivot.reindex(index=[p for p in POSITION_ORDER if p in pivot.index], columns=[c for c in desired_cols if c in pivot.columns])
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    cmap = sns.blend_palette(['#FFF7BC','#FEC44F','#FE9929','#D95F0E','#993404'], as_cmap=True)
    sns.heatmap(pivot, annot=False, cmap=cmap, linewidths=_cfg_attr(plot_cfg,'heatmap_grid_linewidth',0.8), linecolor=_cfg_attr(plot_cfg,'heatmap_grid_color','#F3F3F3'), cbar_kws={'label':'Drop in improvement ratio'}, ax=ax)
    ax.set_xlabel(''); ax.set_ylabel('Dropped position')
    _format_heatmap_xticklabels(ax)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False); _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_position_contribution_retrieval_heatmap(df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig, n_targets: int = 12, target_type: Optional[str] = None) -> None:
    if len(df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = df.copy()
    if target_type == 'continuous':
        plot_df = plot_df.loc[plot_df['target_type'].isin(['continuous','ordinal'])].copy()
    elif target_type == 'binary':
        plot_df = plot_df.loc[plot_df['target_type'].isin(['binary','categorical'])].copy()
    if len(plot_df) == 0:
        return
    target_scores = plot_df.groupby('target', dropna=False)['reference_metric'].max().sort_values(ascending=False)
    top_targets = target_scores.head(int(n_targets)).index.tolist()
    plot_df = plot_df.loc[plot_df['target'].isin(top_targets)].copy()
    agg = plot_df.groupby(['target','dropped_position'], dropna=False)['drop_from_reference_metric'].mean().reset_index()
    pivot = agg.pivot(index='target', columns='dropped_position', values='drop_from_reference_metric')
    pivot = pivot.reindex(index=top_targets, columns=[p for p in POSITION_ORDER if p in pivot.columns])
    fig, ax = plt.subplots(figsize=(8.4, max(5.6, 0.42 * len(pivot))))
    cmap = sns.blend_palette(['#F7FBFF','#9ECAE1','#3182BD','#08519C'], as_cmap=True)
    sns.heatmap(pivot, annot=False, cmap=cmap, linewidths=_cfg_attr(plot_cfg,'heatmap_grid_linewidth',0.8), linecolor=_cfg_attr(plot_cfg,'heatmap_grid_color','#F3F3F3'), cbar_kws={'label':'Drop from reference'}, ax=ax)
    ax.set_xlabel('Dropped position'); ax.set_ylabel('')
    _format_heatmap_xticklabels(ax)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False); _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def _locate_file(root: Path, relative_path: str) -> Path | None:
    candidate = root / relative_path
    if candidate.exists():
        return candidate
    matches = list(root.rglob(Path(relative_path).name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
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
        for group in ['structure','function','burden']:
            D_clin = _safe_load_npy_rel(root, f'global_alignment/clinical_distance_{group}.npy')
            if D_clin is not None:
                fp = fig_dir / f'global_acoustic_vs_clinical_distance_{group}.png'
                plot_global_alignment_scatter(D_ac, D_clin, group, fp, plot_cfg)
                created.append(str(fp))
    summary = _safe_read_csv_rel(root, 'global_alignment/global_alignment_summary.csv')
    if summary is not None:
        main_df = summary.loc[summary['group']!='all'].copy()
        if len(main_df) > 0:
            fp = fig_dir / 'global_alignment_summary.png'
            plot_global_alignment_summary(main_df, fp, plot_cfg)
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
        main_df = group_df.loc[group_df['group']!='all'].copy()
        if len(main_df) > 0:
            fp = fig_dir / 'neighborhood_group_summary.png'
            plot_neighborhood_summary(main_df, fp, plot_cfg)
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
        main_df = summary.loc[summary['group']!='all'].copy()
        if len(main_df) > 0:
            fp = fig_dir / 'adjusted_global_alignment_summary.png'
            plot_global_alignment_bar(main_df, fp, plot_cfg)
            created.append(str(fp))


def render_bootstrap(root: Path, fig_dir: Path, plot_cfg: PlotConfig, created: List[str]) -> None:
    print_banner('Rendering bootstrap-robustness figures')
    g_raw = _safe_read_csv_rel(root, 'robustness/bootstrap/global_alignment/bootstrap_global_alignment_raw.csv')
    g_sum = _safe_read_csv_rel(root, 'robustness/bootstrap/global_alignment/bootstrap_global_alignment_summary.csv')
    if g_sum is not None:
        g_sum = g_sum.loc[g_sum['group'] != 'all'].copy()
        fp = fig_dir / 'bootstrap_global_alignment_forest.png'
        plot_global_alignment_forest(g_sum, fp, plot_cfg)
        created.append(str(fp))
    if g_raw is not None:
        g_raw = g_raw.loc[g_raw['group'] != 'all'].copy()
        fp = fig_dir / 'bootstrap_global_alignment_distribution.png'
        plot_bootstrap_global_distribution(g_raw, fp, plot_cfg)
        created.append(str(fp))
    n_sum = _safe_read_csv_rel(root, 'robustness/bootstrap/neighborhood/bootstrap_neighborhood_summary.csv')
    if n_sum is not None:
        n_sum = n_sum.loc[n_sum['group'] != 'all'].copy()
        fp = fig_dir / 'bootstrap_neighborhood_summary.png'
        plot_bootstrap_neighborhood_summary(n_sum, fp, plot_cfg)
        created.append(str(fp))
    r_sum = _safe_read_csv_rel(root, 'robustness/bootstrap/retrieval/bootstrap_retrieval_summary.csv')
    if r_sum is not None:
        cont = r_sum.loc[r_sum['metric']=='spearman'].rename(columns={'estimate_mean':'spearman'})
        cls = r_sum.loc[r_sum['metric']=='balanced_accuracy'].rename(columns={'estimate_mean':'balanced_accuracy'})
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
    plot_discovery_validation_paired(summary, fp, plot_cfg, 'global_spearman_best_group', 'Spearman correlation')
    if fp.exists(): created.append(str(fp))
    fp = fig_dir / 'discovery_validation_neighborhood_improvement.png'
    plot_discovery_validation_paired(summary, fp, plot_cfg, 'neighborhood_improvement_best_k', 'Improvement ratio')
    if fp.exists(): created.append(str(fp))
    fp = fig_dir / 'discovery_validation_retrieval_mean_spearman.png'
    plot_discovery_validation_paired(summary, fp, plot_cfg, 'retrieval_mean_spearman', 'Mean retrieval Spearman')
    if fp.exists(): created.append(str(fp))


def render_interpretability(root: Path, fig_dir: Path, plot_cfg: PlotConfig, created: List[str]) -> None:
    print_banner('Rendering interpretability figures')
    global_main = _safe_read_csv_rel(root, 'interpretability/variablewise_consolidation/variablewise_global_main_table.csv')
    if global_main is not None:
        fp = fig_dir / 'interpretability_variablewise_fdr_topn.png'
        plot_variablewise_fdr_topn(global_main, fp, plot_cfg, top_n=12)
        created.append(str(fp))
    pos_g = _safe_read_csv_rel(root, 'interpretability/position_contribution/position_leave_one_out_global_summary.csv')
    if pos_g is not None:
        fp = fig_dir / 'position_leave_one_out_global_heatmap.png'
        plot_position_contribution_global_heatmap(pos_g, fp, plot_cfg)
        created.append(str(fp))
    pos_n = _safe_read_csv_rel(root, 'interpretability/position_contribution/position_leave_one_out_neighborhood_summary.csv')
    if pos_n is not None:
        fp = fig_dir / 'position_leave_one_out_neighborhood_heatmap.png'
        plot_position_contribution_neighborhood_heatmap(pos_n, fp, plot_cfg)
        created.append(str(fp))
    pos_r = _safe_read_csv_rel(root, 'interpretability/position_contribution/position_leave_one_out_retrieval_summary.csv')
    if pos_r is not None:
        cont = pos_r.loc[pos_r['target_type'].isin(['continuous','ordinal'])].copy()
        if len(cont) > 0:
            fp = fig_dir / 'position_leave_one_out_retrieval_continuous_heatmap.png'
            plot_position_contribution_retrieval_heatmap(cont, fp, plot_cfg, n_targets=12, target_type='continuous')
            created.append(str(fp))
        cls = pos_r.loc[pos_r['target_type'].isin(['binary','categorical'])].copy()
        if len(cls) > 0:
            fp = fig_dir / 'position_leave_one_out_retrieval_categorical_heatmap.png'
            plot_position_contribution_retrieval_heatmap(cls, fp, plot_cfg, n_targets=8, target_type='binary')
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
    render_interpretability(root, fig_dir, plot_cfg, created)
    manifest = {'root':str(root), 'figure_dir':str(fig_dir), 'n_figures':len(created), 'files':created}
    with (fig_dir/'plot_manifest.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log_done(f'Finished rendering {len(created)} figures into: {fig_dir}')
    return created


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Redraw all clinical-alignment and interpretability figures from saved result tables / matrices.')
    p.add_argument('--out-root', type=str, required=True, help='Clinical_alignment output root containing prepared/, global_alignment/, retrieval/, robustness/, interpretability/ ...')
    p.add_argument('--fig-dir', type=str, default=None, help='Single folder to save all regenerated figures. Default: <out-root>/all_redrawn_figures')
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = build_default_config(); plot_cfg = cfg.plot
    root = Path(args.out_root)
    fig_dir = Path(args.fig_dir) if args.fig_dir else root / 'all_redrawn_figures'
    render_all(root, fig_dir, plot_cfg)


if __name__ == '__main__':
    main()
