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
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.lines import Line2D
import seaborn as sns

try:
    from config import PlotConfig, build_default_config, set_publication_plot_style, log_done, log_warn, log_info, print_banner
except Exception:
    from dataclasses import dataclass, field

    @dataclass
    class PlotConfig:
        dpi: int = 300
        font_family: str = 'Arial'
        base_font_size: int = 18
        title_font_size: int = 22
        label_font_size: int = 18
        tick_font_size: int = 16
        legend_font_size: int = 14
        context: str = 'talk'
        style: str = 'white'
        palette: List[str] = field(default_factory=lambda: ['#158F8C', '#2F6DB3', '#E38B2C', '#7B5AA6', '#5FB0B7', '#F1B44C', '#C75C5C'])
        axis_grid_alpha: float = 0.20
        axis_grid_linewidth: float = 0.9
        heatmap_grid_linewidth: float = 0.8
        heatmap_grid_color: str = '#F2F2F2'
        heatmap_annot: bool = True
        global_scatter_max_points: int = 60000
        neighborhood_heatmap_top_n: int = 18
        bootstrap_retrieval_top_n: int = 12
        retrieval_heatmap_top_n_continuous: int = 12
        retrieval_heatmap_top_n_categorical: int = 8

    def build_default_config():
        class X:
            pass
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

    def log_done(msg):
        print(f'[done] {msg}')

    def log_warn(msg):
        print(f'[warn] {msg}')

    def log_info(msg):
        print(f'[info] {msg}')

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
GROUP_COLORS = {
    'Functional': '#158F8C',
    'Structural': '#2F6DB3',
    'Burden': '#E38B2C',
    'Integrated': '#7B5AA6',
}
METRIC_COLORS = {
    'Pearson r': '#158F8C',
    'Spearman r': '#2F6DB3',
}
SPACE_COLORS = ['#158F8C', '#2F6DB3', '#E38B2C', '#7B5AA6', '#5FB0B7', '#F1B44C']


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
        ax.grid(True, axis=grid_axis, alpha=_cfg_attr(plot_cfg, 'axis_grid_alpha', 0.20), linewidth=_cfg_attr(plot_cfg, 'axis_grid_linewidth', 0.9))
    else:
        ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.4)


def _panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(-0.12, 1.02, text, transform=ax.transAxes, ha='left', va='bottom', fontweight='bold')


def _save_fig(fig: plt.Figure, out_path: str | Path, dpi: int = 300) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    log_done(f'Saved figure to: {out_path}')


def _make_standalone_figure(width: float = 7.0, height: float = 5.0) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def _compose_panels_from_files(
    panel_paths: Sequence[str | Path],
    out_path: str | Path,
    panel_labels: Sequence[str],
    width_scale: float = 4.9,
    height: float = 5.4,
    top_pad: float = 0.92,
    wspace: float = 0.05,
) -> None:
    paths = [Path(p) for p in panel_paths if p is not None and Path(p).exists()]
    if len(paths) == 0:
        return
    images = [plt.imread(str(p)) for p in paths]
    ratios = [img.shape[1] / max(img.shape[0], 1) for img in images]
    fig_w = max(12.0, width_scale * float(np.sum(ratios)))
    fig, axes = plt.subplots(
        1,
        len(images),
        figsize=(fig_w, height),
        gridspec_kw={'width_ratios': ratios},
    )
    if len(images) == 1:
        axes = [axes]
    for ax, img, label in zip(axes, images, panel_labels):
        ax.imshow(img)
        ax.axis('off')
        ax.text(-0.035, 1.02, label, transform=ax.transAxes, ha='left', va='bottom', fontweight='bold')
    fig.subplots_adjust(wspace=wspace, top=top_pad)
    _save_fig(fig, out_path)


def _filter_main_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'group' in out.columns:
        key = out['group'].astype(str).str.lower()
        out = out.loc[key.isin(['function', 'structure', 'burden', 'functional', 'structural'])].copy()
        out = _apply_group_labels(out)
        out = out.loc[out['group'].isin(GROUP_ORDER)].copy()
    return out


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
            'x_mid': float((left + right) / 2),
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


def _legend_from_pairs(pairs: Sequence[Tuple[object, str]]) -> Tuple[List[object], List[str]]:
    handles: List[object] = []
    labels: List[str] = []
    seen = set()
    for h, l in pairs:
        if l in seen:
            continue
        handles.append(h)
        labels.append(l)
        seen.add(l)
    return handles, labels


def _safe_center_norm(vmin: float, vmax: float, center: float = 0.0) -> Normalize:
    if vmin >= center or vmax <= center:
        vmax2 = max(abs(vmin), abs(vmax), 1e-9)
        return Normalize(vmin=-vmax2, vmax=vmax2)
    return TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)


def _value_text_color(value: float, vmax_abs: float) -> str:
    return 'white' if abs(value) >= 0.55 * vmax_abs else 'black'


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


def _extract_variable_group_map(root: Path) -> Dict[str, str]:
    for rel in [
        'interpretability/variablewise_consolidation/variablewise_global_main_table.csv',
        'global_alignment/variablewise_global_alignment_summary.csv',
    ]:
        df = _safe_read_csv_rel(root, rel)
        if df is None or 'variable' not in df.columns or 'group' not in df.columns:
            continue
        out = df[['variable', 'group']].dropna().drop_duplicates().copy()
        out['group'] = out['group'].map(_display_group_value)
        return dict(zip(out['variable'].astype(str), out['group'].astype(str)))
    return {}


def _group_order_present(values: Sequence[str]) -> List[str]:
    vals = list(pd.unique(pd.Series(list(values)).dropna()))
    ordered = [g for g in GROUP_ORDER if g in vals]
    for v in vals:
        if v not in ordered:
            ordered.append(v)
    return ordered


# -----------------------------------------------------------------------------
# Global alignment and summary panels
# -----------------------------------------------------------------------------

def plot_global_scatter_triptych(acoustic_distance: np.ndarray, group_to_clinical: Dict[str, np.ndarray], out_path: str | Path, plot_cfg: PlotConfig, seed: int = 42) -> None:
    if acoustic_distance is None or len(group_to_clinical) == 0:
        return
    set_publication_plot_style(plot_cfg)
    D_ac = _ensure_square_distance_matrix(acoustic_distance)
    x_all = upper_triangle_vector(D_ac)
    xlim = [np.nanmin(x_all), np.nanmax(x_all)]

    ordered = [('function', 'Functional'), ('structure', 'Structural'), ('burden', 'Burden')]
    avail = [(k, lab) for k, lab in ordered if k in group_to_clinical and group_to_clinical[k] is not None]
    if len(avail) == 0:
        return

    y_mins, y_maxs = [], []
    for key, _ in avail:
        yv = upper_triangle_vector(_ensure_square_distance_matrix(group_to_clinical[key], f'clinical_{key}'))
        y_mins.append(np.nanmin(yv))
        y_maxs.append(np.nanmax(yv))
    ylim = [float(np.min(y_mins)), float(np.max(y_maxs))]

    fig, axes = plt.subplots(1, len(avail), figsize=(5.6 * len(avail), 5.4), sharex=True, sharey=True)
    if len(avail) == 1:
        axes = [axes]
    cmap = sns.blend_palette(['#F1B44C', '#158F8C', '#2F6DB3', '#7B5AA6'], as_cmap=True)
    mappable = None
    for i, ((key, lab), ax) in enumerate(zip(avail, axes)):
        y = upper_triangle_vector(_ensure_square_distance_matrix(group_to_clinical[key], f'clinical_{key}'))
        mask = np.isfinite(x_all) & np.isfinite(y)
        x = x_all[mask]
        y = y[mask]
        x_s, y_s = _sample_for_scatter(x, y, max_points=_cfg_attr(plot_cfg, 'global_scatter_max_points', 60000), seed=seed)
        hb = ax.hexbin(x_s, y_s, gridsize=44, cmap=cmap, mincnt=1, extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
        mappable = hb
        trend = _binned_trend(x_s, y_s, n_bins=22)
        if len(trend) > 0:
            ax.fill_between(trend['x_mid'], trend['y_lo'], trend['y_hi'], color='#C75C5C', alpha=0.16, linewidth=0)
            ax.plot(trend['x_mid'], trend['y_med'], color='#C75C5C', linewidth=2.3)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel('Acoustic distance')
        if i == 0:
            ax.set_ylabel('Clinical distance')
        else:
            ax.set_ylabel('')
        _panel_label(ax, chr(ord('A') + i))
        _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False)
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, fraction=0.03, pad=0.02)
        cbar.set_label('Pair count')
    fig.subplots_adjust(wspace=0.12)
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def _draw_grouped_bar(ax: plt.Axes, df: pd.DataFrame, value_cols: Sequence[str], y_label: str, group_order: Sequence[str]) -> None:
    bar_w = 0.36
    x = np.arange(len(group_order), dtype=float)
    colors = [METRIC_COLORS[_metric_display_name(c)] for c in value_cols]
    for i, col in enumerate(value_cols):
        vals = []
        for g in group_order:
            sub = df.loc[df['group'] == g, col]
            vals.append(float(sub.iloc[0]) if len(sub) else np.nan)
        ax.bar(x + (i - (len(value_cols)-1)/2) * bar_w, vals, width=bar_w, color=colors[i], edgecolor='none', label=_metric_display_name(col))
    ax.set_xticks(x)
    ax.set_xticklabels(group_order)
    ax.set_xlabel('')
    ax.set_ylabel(y_label)
    ax.axhline(0.0, color='#7F7F7F', linestyle='--', linewidth=1.1)


def _draw_forest(ax: plt.Axes, df: pd.DataFrame, metrics: Sequence[str]) -> None:
    plot_df = df.copy()
    plot_df = plot_df.loc[plot_df['metric'].isin(metrics)].copy()
    plot_df['group'] = plot_df['group'].map(_display_group_value)
    plot_df = plot_df.loc[plot_df['group'].isin(GROUP_ORDER)].copy()
    if len(plot_df) == 0:
        return
    rows = []
    for g in GROUP_ORDER:
        for m in metrics:
            sub = plot_df.loc[(plot_df['group'] == g) & (plot_df['metric'] == m)]
            if len(sub) == 0:
                continue
            rows.append(sub.iloc[0])
    plot_df = pd.DataFrame(rows)
    y = np.arange(len(plot_df))[::-1]
    for yi, (_, row) in zip(y, plot_df.iterrows()):
        color = METRIC_COLORS.get(_metric_display_name(str(row['metric'])), '#4C4C4C')
        ax.plot([row['ci_lower'], row['ci_upper']], [yi, yi], color=color, linewidth=2.2)
        ax.scatter(row['estimate_mean'], yi, s=85, color=color, zorder=3)
    labels = [f"{row['group']} | {_metric_display_name(str(row['metric']))}" for _, row in plot_df.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0.0, color='#7F7F7F', linestyle='--', linewidth=1.1)
    ax.set_xlabel('Estimate (mean with 95% CI)')
    ax.set_ylabel('')



def plot_global_alignment_summary_bar(summary_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig) -> None:
    if summary_df is None or len(summary_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    raw_df = _filter_main_groups(summary_df[['group', 'pearson_r', 'spearman_r']].copy())
    if len(raw_df) == 0:
        return
    group_order = [g for g in GROUP_ORDER if g in raw_df['group'].unique().tolist()]
    fig, ax = _make_standalone_figure(width=7.1, height=5.0)
    _draw_grouped_bar(ax, raw_df, ['pearson_r', 'spearman_r'], 'Correlation', group_order)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='y')
    handles = [
        Line2D([0], [0], marker='s', markersize=9, linestyle='None', color=METRIC_COLORS['Pearson r']),
        Line2D([0], [0], marker='s', markersize=9, linestyle='None', color=METRIC_COLORS['Spearman r']),
    ]
    ax.legend(handles, ['Pearson r', 'Spearman r'], loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.subplots_adjust(right=0.82)
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_global_alignment_bootstrap_forest(bootstrap_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig) -> None:
    if bootstrap_df is None or len(bootstrap_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    bsum = _filter_main_groups(bootstrap_df.copy())
    if len(bsum) == 0:
        return
    fig, ax = _make_standalone_figure(width=10.0, height=5.0)
    _draw_forest(ax, bsum, metrics=['pearson_r', 'spearman_r'])
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='x')
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_global_alignment_adjusted_bar(adjusted_df: Optional[pd.DataFrame], out_path: str | Path, plot_cfg: PlotConfig) -> None:
    if adjusted_df is None or len(adjusted_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = _filter_main_groups(adjusted_df[['group', 'pearson_r', 'spearman_r']].copy())
    if len(plot_df) == 0:
        return
    group_order = [g for g in GROUP_ORDER if g in plot_df['group'].unique().tolist()]
    fig, ax = _make_standalone_figure(width=7.1, height=5.0)
    _draw_grouped_bar(ax, plot_df, ['pearson_r', 'spearman_r'], 'Adjusted correlation', group_order)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='y')
    handles = [
        Line2D([0], [0], marker='s', markersize=9, linestyle='None', color=METRIC_COLORS['Pearson r']),
        Line2D([0], [0], marker='s', markersize=9, linestyle='None', color=METRIC_COLORS['Spearman r']),
    ]
    ax.legend(handles, ['Pearson r', 'Spearman r'], loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.subplots_adjust(right=0.82)
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_discovery_validation_paired_standalone(summary_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig, metric_col: str = 'global_spearman_best_group') -> None:
    if summary_df is None or len(summary_df) == 0 or metric_col not in summary_df.columns:
        return
    set_publication_plot_style(plot_cfg)
    fig, ax = _make_standalone_figure(width=7.0, height=5.0)
    plot_discovery_validation_paired_clean(summary_df, ax, metric_col)
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_baseline_connected_dot_standalone(baseline_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig, value_col: str = 'spearman_r') -> None:
    if baseline_df is None or len(baseline_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    fig, ax = _make_standalone_figure(width=7.6, height=5.0)
    color_map = plot_baseline_connected_dot(baseline_df, ax, value_col=value_col)
    if color_map:
        pairs = [(Line2D([0], [0], marker='o', linestyle='None', color=c, markersize=8), sp) for sp, c in color_map.items()]
        handles, labels = _legend_from_pairs(pairs)
        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
        fig.subplots_adjust(right=0.80)
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_distance_space_heatmap_standalone(comparison_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig, metric: str = 'spearman_r') -> None:
    if comparison_df is None or len(comparison_df) == 0 or metric not in comparison_df.columns:
        return
    set_publication_plot_style(plot_cfg)
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    plot_distance_space_correlation_heatmap(comparison_df, metric, ax, plot_cfg, annot=True)
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_composite_global_alignment(panel_paths: Sequence[str | Path], out_path: str | Path) -> None:
    _compose_panels_from_files(panel_paths, out_path, panel_labels=['A', 'B', 'C'], width_scale=5.2, height=5.7, top_pad=0.95, wspace=0.04)


def plot_bootstrap_global_raincloud(raw_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig) -> None:
    raw_df = raw_df.loc[raw_df['group'] != 'all'].copy()
    if len(raw_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    plot_df = raw_df.melt(id_vars=['group', 'repeat'], value_vars=['pearson_r', 'spearman_r'], var_name='metric', value_name='value')
    plot_df = _apply_group_labels(plot_df)
    plot_df['metric'] = plot_df['metric'].map(_metric_display_name)
    order = [g for g in GROUP_ORDER if g in plot_df['group'].unique().tolist()]
    hue_order = ['Pearson r', 'Spearman r']
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    sns.violinplot(data=plot_df, x='group', y='value', hue='metric', order=order, hue_order=hue_order, inner=None, cut=0, linewidth=0, dodge=True, ax=ax, saturation=1.0)
    sns.boxplot(data=plot_df, x='group', y='value', hue='metric', order=order, hue_order=hue_order, dodge=True, width=0.22, showfliers=False,
                boxprops={'facecolor': 'white', 'alpha': 0.8, 'linewidth': 1.2}, whiskerprops={'linewidth': 1.1}, medianprops={'color': 'black', 'linewidth': 1.2}, ax=ax)
    sample_df = plot_df.sample(min(len(plot_df), 800), random_state=42) if len(plot_df) > 0 else plot_df
    sns.stripplot(data=sample_df, x='group', y='value', hue='metric', order=order, hue_order=hue_order, dodge=True, alpha=0.18, size=2.6, color='black', ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    pairs = []
    for h, l in zip(handles, labels):
        if l in hue_order:
            pairs.append((h, l))
    handles2, labels2 = _legend_from_pairs(pairs)
    if handles2:
        ax.legend(handles2, labels2, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    ax.axhline(0.0, linestyle='--', color='#7F7F7F', linewidth=1.1)
    ax.set_xlabel('')
    ax.set_ylabel('Bootstrap estimate')
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='y')
    fig.subplots_adjust(top=0.88)
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


# -----------------------------------------------------------------------------
# Variable-wise interpretation
# -----------------------------------------------------------------------------

def _plot_variable_lollipop(ax: plt.Axes, df: pd.DataFrame, top_n: Optional[int] = None, fdr_col: Optional[str] = None) -> None:
    if len(df) == 0:
        return
    plot_df = df.copy()
    if top_n is not None:
        sort_cols = [c for c in ['spearman_r', 'coverage'] if c in plot_df.columns]
        asc = [False] * len(sort_cols)
        if len(sort_cols) > 0:
            plot_df = plot_df.sort_values(sort_cols, ascending=asc).head(int(top_n)).copy()
    plot_df = _apply_group_labels(plot_df)
    plot_df = plot_df.sort_values('spearman_r', ascending=False).reset_index(drop=True)
    y_pos = np.arange(len(plot_df), dtype=float)
    ax.hlines(y=y_pos, xmin=0.0, xmax=plot_df['spearman_r'], color='#D1D1D1', linewidth=1.6, zorder=1)
    marker_map: Optional[List[str]] = None
    if fdr_col is not None and fdr_col in plot_df.columns:
        marker_map = plot_df[fdr_col].map({True: 'o', False: 'X'}).fillna('o').tolist()
    for i, (_, row) in enumerate(plot_df.iterrows()):
        marker = marker_map[i] if marker_map is not None else 'o'
        color = GROUP_COLORS.get(str(row['group']), '#4C4C4C')
        ax.scatter(row['spearman_r'], y_pos[i], s=110, color=color, marker=marker, zorder=3)
    ax.axvline(0.0, linestyle='--', color='#7F7F7F', linewidth=1.1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['variable'].tolist())
    ax.invert_yaxis()
    ax.set_xlabel('Spearman correlation')
    ax.set_ylabel('')
    for tick, (_, row) in zip(ax.get_yticklabels(), plot_df.iterrows()):
        tick.set_color(GROUP_COLORS.get(str(row['group']), 'black'))
    _apply_axes_style(ax, show_grid=True, grid_axis='x')


def plot_variablewise_global_alignment_all(df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig) -> None:
    if len(df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    fig, ax = plt.subplots(figsize=(11.8, max(6.0, 0.42 * len(df))))
    _plot_variable_lollipop(ax, df, top_n=None)
    handles = [Line2D([0], [0], marker='o', linestyle='None', color=GROUP_COLORS[g], markersize=8) for g in GROUP_ORDER if g in _apply_group_labels(df)['group'].unique()]
    labels = [g for g in GROUP_ORDER if g in _apply_group_labels(df)['group'].unique()]
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.subplots_adjust(right=0.80)
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_variablewise_fdr_topn(df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig, top_n: int = 12) -> None:
    if len(df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    fig, ax = plt.subplots(figsize=(10.8, max(5.8, 0.48 * top_n)))
    _plot_variable_lollipop(ax, df, top_n=top_n, fdr_col='global_fdr_sig_overall')
    handles = [Line2D([0], [0], marker='o', linestyle='None', color=GROUP_COLORS[g], markersize=8) for g in GROUP_ORDER if g in _apply_group_labels(df)['group'].unique()]
    labels = [g for g in GROUP_ORDER if g in _apply_group_labels(df)['group'].unique()]
    if 'global_fdr_sig_overall' in df.columns:
        handles.extend([
            Line2D([0], [0], marker='o', linestyle='None', color='black', markersize=8),
            Line2D([0], [0], marker='X', linestyle='None', color='black', markersize=8),
        ])
        labels.extend(['FDR significant', 'Not FDR significant'])
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.subplots_adjust(right=0.78)
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_variable_neighbor_bubble(neighborhood_df: pd.DataFrame, retrieval_df: pd.DataFrame, var_group_map: Dict[str, str], out_path: str | Path, plot_cfg: PlotConfig, top_n: int = 10) -> None:
    if len(neighborhood_df) == 0 or len(retrieval_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    ret = retrieval_df.loc[retrieval_df['target_type'].isin(['continuous', 'ordinal'])].copy()
    if len(ret) == 0 or 'spearman' not in ret.columns:
        return
    ret = ret.rename(columns={'target': 'variable', 'spearman': 'spearman_r'})
    nei = neighborhood_df[['variable', 'k', 'improvement_ratio']].copy()
    merged = ret.merge(nei, on=['variable', 'k'], how='inner')
    if len(merged) == 0:
        return
    if 'group' not in merged.columns:
        merged['group'] = merged['variable'].map(var_group_map)
    merged['group'] = merged['group'].map(_display_group_value)
    merged = merged.dropna(subset=['group']).copy()
    ranking = merged.groupby('variable')['spearman_r'].max().sort_values(ascending=False)
    top_vars = ranking.head(int(top_n)).index.tolist()
    merged = merged.loc[merged['variable'].isin(top_vars)].copy()
    order = top_vars
    x_levels = sorted(merged['k'].dropna().unique().tolist())
    x_map = {k: i for i, k in enumerate(x_levels)}
    merged['x'] = merged['k'].map(x_map).astype(float)
    y_map = {v: i for i, v in enumerate(order[::-1])}
    merged['y'] = merged['variable'].map(y_map).astype(float)

    fig, ax = plt.subplots(figsize=(9.0, max(5.6, 0.55 * len(order))))
    vmin = float(merged['spearman_r'].min())
    vmax = float(merged['spearman_r'].max())
    size_min, size_max = 80.0, 420.0
    ir = merged['improvement_ratio'].to_numpy(dtype=float)
    ir_lo, ir_hi = float(np.nanmin(ir)), float(np.nanmax(ir))
    if np.isclose(ir_lo, ir_hi):
        merged['size'] = (size_min + size_max) / 2
    else:
        merged['size'] = size_min + (merged['improvement_ratio'] - ir_lo) / (ir_hi - ir_lo) * (size_max - size_min)

    sc = ax.scatter(merged['x'], merged['y'], s=merged['size'], c=merged['spearman_r'], cmap='Blues', vmin=vmin, vmax=vmax, edgecolor='white', linewidth=0.9)
    ax.set_xticks(range(len(x_levels)))
    ax.set_xticklabels([str(k) for k in x_levels])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order[::-1])
    ax.set_xlabel('Top-k acoustic neighbors')
    ax.set_ylabel('')
    for tick in ax.get_yticklabels():
        grp = var_group_map.get(tick.get_text(), None)
        grp = _display_group_value(grp) if grp is not None else None
        if grp in GROUP_COLORS:
            tick.set_color(GROUP_COLORS[grp])
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label('Retrieval Spearman')
    # size legend
    if ir_hi > 0:
        ref_vals = np.quantile(ir, [0.25, 0.50, 0.75]) if len(np.unique(ir)) >= 3 else np.array([ir_lo, np.nanmedian(ir), ir_hi])
        ref_vals = np.unique(np.round(ref_vals, 2))
        legend_handles = []
        legend_labels = []
        for rv in ref_vals:
            if np.isclose(ir_lo, ir_hi):
                sz = (size_min + size_max) / 2
            else:
                sz = size_min + (rv - ir_lo) / (ir_hi - ir_lo) * (size_max - size_min)
            legend_handles.append(plt.scatter([], [], s=sz, color='#A0A0A0', edgecolor='white'))
            legend_labels.append(f'{rv:.2f}')
        ax.legend(legend_handles, legend_labels, title='Improvement ratio', loc='upper left', bbox_to_anchor=(1.02, 0.98), borderaxespad=0.0)
        fig.subplots_adjust(right=0.80)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='both')
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_composite_variable_interpretation(panel_paths: Sequence[str | Path], out_path: str | Path) -> None:
    _compose_panels_from_files(panel_paths, out_path, panel_labels=['A', 'B'], width_scale=5.4, height=6.0, top_pad=0.95, wspace=0.05)


def _rank_targets_for_retrieval(df: pd.DataFrame, value_col: str, top_n: int) -> List[str]:
    return df.groupby('target', dropna=False)[value_col].max().sort_values(ascending=False).head(int(top_n)).index.tolist()


def plot_retrieval_continuous_heatmap(summary_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig, top_n: int = 10, full: bool = False) -> None:
    cont_df = summary_df.loc[summary_df['target_type'].isin(['continuous', 'ordinal'])].copy()
    if len(cont_df) == 0 or 'spearman' not in cont_df.columns:
        return
    set_publication_plot_style(plot_cfg)
    top_targets = _rank_targets_for_retrieval(cont_df, 'spearman', top_n if not full else max(top_n, len(cont_df['target'].unique())))
    plot_df = cont_df.loc[cont_df['target'].isin(top_targets)].copy()
    pivot = plot_df.pivot_table(index='target', columns='k', values='spearman', aggfunc='mean').reindex(top_targets)
    fig, ax = plt.subplots(figsize=(8.8, max(5.8, 0.46 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='Blues', linewidths=_cfg_attr(plot_cfg, 'heatmap_grid_linewidth', 0.8), linecolor=_cfg_attr(plot_cfg, 'heatmap_grid_color', '#F2F2F2'), cbar_kws={'label': 'Spearman'}, ax=ax)
    ax.set_xlabel('Top-k acoustic neighbors')
    ax.set_ylabel('')
    _format_heatmap_xticklabels(ax)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False)
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_retrieval_categorical_delta_heatmap(summary_df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig, top_n: int = 8) -> None:
    cls_df = summary_df.loc[summary_df['target_type'].isin(['binary', 'categorical'])].copy()
    if len(cls_df) == 0 or 'balanced_accuracy' not in cls_df.columns:
        return
    set_publication_plot_style(plot_cfg)
    cls_df['delta_ba'] = cls_df['balanced_accuracy'] - 0.5
    top_targets = cls_df.groupby('target', dropna=False)['delta_ba'].max().sort_values(ascending=False).head(int(top_n)).index.tolist()
    plot_df = cls_df.loc[cls_df['target'].isin(top_targets)].copy()
    pivot = plot_df.pivot_table(index='target', columns='k', values='delta_ba', aggfunc='mean').reindex(top_targets)
    vmax = float(np.nanmax(np.abs(pivot.to_numpy()))) if np.isfinite(pivot.to_numpy()).any() else 0.05
    vmax = max(vmax, 1e-3)
    fig, ax = plt.subplots(figsize=(8.4, max(5.4, 0.42 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu', norm=_safe_center_norm(-vmax, vmax, 0.0), linewidths=_cfg_attr(plot_cfg, 'heatmap_grid_linewidth', 0.8), linecolor=_cfg_attr(plot_cfg, 'heatmap_grid_color', '#F2F2F2'), cbar_kws={'label': 'Balanced accuracy - 0.5'}, ax=ax)
    ax.set_xlabel('Top-k acoustic neighbors')
    ax.set_ylabel('')
    _format_heatmap_xticklabels(ax)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False)
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


# -----------------------------------------------------------------------------
# Baselines, discovery-validation, and robustness
# -----------------------------------------------------------------------------

def plot_baseline_connected_dot(baseline_df: pd.DataFrame, ax: plt.Axes, value_col: str = 'spearman_r') -> None:
    if len(baseline_df) == 0:
        return
    plot_df = baseline_df.loc[(baseline_df['analysis'] == 'global_alignment') & (baseline_df['group'] != 'all')].copy()
    if len(plot_df) == 0 or value_col not in plot_df.columns:
        return
    plot_df = _apply_group_labels(plot_df)
    groups = [g for g in GROUP_ORDER if g in plot_df['group'].unique().tolist()]
    spaces = list(plot_df['space'].dropna().astype(str).unique())
    color_map = {sp: SPACE_COLORS[i % len(SPACE_COLORS)] for i, sp in enumerate(spaces)}
    y = np.arange(len(groups), dtype=float)
    for yi, g in zip(y, groups):
        sub = plot_df.loc[plot_df['group'] == g].copy().sort_values(value_col)
        if len(sub) == 0:
            continue
        ax.plot(sub[value_col], np.full(len(sub), yi), color='#D0D0D0', linewidth=1.8, zorder=1)
        for _, row in sub.iterrows():
            ax.scatter(row[value_col], yi, s=110, color=color_map[str(row['space'])], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(groups)
    ax.set_xlabel('Spearman correlation')
    ax.set_ylabel('')
    ax.axvline(0.0, color='#7F7F7F', linestyle='--', linewidth=1.1)
    _apply_axes_style(ax, show_grid=True, grid_axis='x')
    return color_map


def plot_distance_space_correlation_heatmap(comparison_df: pd.DataFrame, metric: str, ax: plt.Axes, plot_cfg: PlotConfig, annot: bool = True) -> None:
    if len(comparison_df) == 0 or metric not in comparison_df.columns:
        return
    pivot = comparison_df.pivot(index='space_i', columns='space_j', values=metric)
    sns.heatmap(pivot, annot=annot, fmt='.2f', cmap='Blues', vmin=-1, vmax=1, square=True,
                linewidths=_cfg_attr(plot_cfg, 'heatmap_grid_linewidth', 0.8), linecolor=_cfg_attr(plot_cfg, 'heatmap_grid_color', '#F2F2F2'),
                cbar_kws={'shrink': 0.80, 'label': metric}, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    _format_heatmap_xticklabels(ax)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False)


def plot_discovery_validation_paired_clean(summary_df: pd.DataFrame, ax: plt.Axes, metric_col: str) -> None:
    if len(summary_df) == 0 or metric_col not in summary_df.columns:
        return
    pivot = summary_df.pivot(index='split_id', columns='subset', values=metric_col)
    if not {'discovery', 'validation'}.issubset(set(pivot.columns)):
        return
    for _, row in pivot.iterrows():
        ax.plot([0, 1], [row['discovery'], row['validation']], color='#B8B8B8', linewidth=1.5, zorder=1)
    means = [pivot['discovery'].mean(), pivot['validation'].mean()]
    ax.plot([0, 1], means, color='#5A5A5A', linewidth=2.4, zorder=2)
    ax.scatter([0, 1], means, s=150, color='#2F6DB3', zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Discovery', 'Validation'])
    ax.set_xlabel('')
    ax.set_ylabel('Spearman correlation')
    _apply_axes_style(ax, show_grid=True, grid_axis='y')


def plot_composite_robustness(panel_paths: Sequence[str | Path], out_path: str | Path) -> None:
    _compose_panels_from_files(panel_paths, out_path, panel_labels=['A', 'B', 'C'], width_scale=5.1, height=5.6, top_pad=0.95, wspace=0.04)


# -----------------------------------------------------------------------------
# Position contribution and leave-one-out
# -----------------------------------------------------------------------------
# Position contribution and leave-one-out
# -----------------------------------------------------------------------------

def _prepare_position_contribution_long(global_df: Optional[pd.DataFrame], neighborhood_df: Optional[pd.DataFrame], retrieval_df: Optional[pd.DataFrame], var_group_map: Dict[str, str]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    if global_df is not None and len(global_df) > 0:
        tmp = global_df.copy()
        tmp = _apply_group_labels(tmp)
        tmp = tmp.loc[tmp['group'].isin(GROUP_ORDER)].copy()
        tmp = tmp[['dropped_position', 'group', 'drop_from_reference_spearman_r']].rename(columns={'drop_from_reference_spearman_r': 'value'})
        tmp['panel'] = 'Global alignment'
        rows.append(tmp)
    if neighborhood_df is not None and len(neighborhood_df) > 0:
        tmp = neighborhood_df.copy()
        tmp = _apply_group_labels(tmp)
        tmp = tmp.loc[tmp['group'].isin(GROUP_ORDER)].copy()
        tmp = tmp.groupby(['dropped_position', 'group'], as_index=False)['drop_from_reference_improvement_ratio'].mean().rename(columns={'drop_from_reference_improvement_ratio': 'value'})
        tmp['panel'] = 'Neighborhood consistency'
        rows.append(tmp)
    if retrieval_df is not None and len(retrieval_df) > 0:
        tmp = retrieval_df.loc[retrieval_df['target_type'].isin(['continuous', 'ordinal'])].copy()
        if len(tmp) > 0:
            tmp['group'] = tmp['target'].map(var_group_map).map(_display_group_value)
            tmp = tmp.dropna(subset=['group']).copy()
            tmp = tmp.loc[tmp['group'].isin(GROUP_ORDER)].copy()
            if 'k' in tmp.columns:
                tmp = tmp.groupby(['target', 'dropped_position', 'group'], as_index=False)['drop_from_reference_metric'].mean()
            else:
                tmp = tmp.groupby(['target', 'dropped_position', 'group'], as_index=False)['drop_from_reference_metric'].mean()
            tmp = tmp.groupby(['dropped_position', 'group'], as_index=False)['drop_from_reference_metric'].mean().rename(columns={'drop_from_reference_metric': 'value'})
            tmp['panel'] = 'Continuous retrieval'
            rows.append(tmp)
    if len(rows) == 0:
        return pd.DataFrame(columns=['dropped_position', 'group', 'value', 'panel'])
    out = pd.concat(rows, ignore_index=True, sort=False)
    out['dropped_position'] = pd.Categorical(out['dropped_position'], categories=POSITION_ORDER, ordered=True)
    return out


def plot_position_contribution_faceted_dot(global_df: Optional[pd.DataFrame], neighborhood_df: Optional[pd.DataFrame], retrieval_df: Optional[pd.DataFrame], var_group_map: Dict[str, str], out_path: str | Path, plot_cfg: PlotConfig) -> None:
    long_df = _prepare_position_contribution_long(global_df, neighborhood_df, retrieval_df, var_group_map)
    if len(long_df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    panels = ['Global alignment', 'Neighborhood consistency', 'Continuous retrieval']
    panels = [p for p in panels if p in long_df['panel'].unique().tolist()]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.3 * len(panels), 5.6), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    vmax = float(np.nanmax(np.abs(long_df['value'].to_numpy(dtype=float)))) if len(long_df) else 0.02
    vmax = max(vmax, 1e-4)
    for i, (panel, ax) in enumerate(zip(panels, axes)):
        sub = long_df.loc[long_df['panel'] == panel].copy()
        for g in GROUP_ORDER:
            gsub = sub.loc[sub['group'] == g].sort_values('dropped_position')
            if len(gsub) == 0:
                continue
            ypos = np.arange(len(POSITION_ORDER), dtype=float)
            pos_map = {p: j for j, p in enumerate(POSITION_ORDER)}
            xs = gsub['value'].to_numpy(dtype=float)
            ys = gsub['dropped_position'].map(pos_map).to_numpy(dtype=float)
            ax.hlines(ys, 0.0, xs, color='#D0D0D0', linewidth=1.6, zorder=1)
            ax.scatter(xs, ys, s=95, color=GROUP_COLORS[g], zorder=3, label=g if i == 0 else None)
        ax.axvline(0.0, linestyle='--', color='#7F7F7F', linewidth=1.1)
        ax.set_xlim(-vmax * 1.10, vmax * 1.10)
        ax.set_yticks(range(len(POSITION_ORDER)))
        ax.set_yticklabels(POSITION_ORDER)
        ax.set_xlabel('Drop from reference')
        if i == 0:
            ax.set_ylabel('Dropped position')
        else:
            ax.set_ylabel('')
        _panel_label(ax, chr(ord('A') + i))
        _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=True, grid_axis='x')
    handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = _legend_from_pairs(list(zip(handles, labels)))
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig.subplots_adjust(top=0.86)
    fig.subplots_adjust(wspace=0.18)
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


def plot_position_contribution_retrieval_heatmap(df: pd.DataFrame, out_path: str | Path, plot_cfg: PlotConfig, var_group_map: Dict[str, str], target_type: str = 'continuous', top_n: int = 12) -> None:
    if len(df) == 0:
        return
    set_publication_plot_style(plot_cfg)
    if target_type == 'continuous':
        plot_df = df.loc[df['target_type'].isin(['continuous', 'ordinal'])].copy()
    else:
        plot_df = df.loc[df['target_type'].isin(['binary', 'categorical'])].copy()
    if len(plot_df) == 0:
        return
    plot_df['group'] = plot_df['target'].map(var_group_map).map(_display_group_value)
    rank_col = 'reference_metric' if 'reference_metric' in plot_df.columns else 'drop_from_reference_metric'
    ranked_targets = plot_df.groupby('target', dropna=False)[rank_col].max().sort_values(ascending=False).head(int(top_n)).index.tolist()
    plot_df = plot_df.loc[plot_df['target'].isin(ranked_targets)].copy()
    if target_type == 'binary':
        value_label = 'Drop in (BA - 0.5)'
    else:
        value_label = 'Drop from reference'
    if 'k' in plot_df.columns:
        plot_df = plot_df.groupby(['target', 'dropped_position'], as_index=False)['drop_from_reference_metric'].mean()
    else:
        plot_df = plot_df.groupby(['target', 'dropped_position'], as_index=False)['drop_from_reference_metric'].mean()
    ordered_targets = []
    for g in GROUP_ORDER:
        members = [t for t in ranked_targets if _display_group_value(var_group_map.get(t)) == g]
        members_sorted = sorted(members, key=lambda x: ranked_targets.index(x))
        ordered_targets.extend(members_sorted)
    for t in ranked_targets:
        if t not in ordered_targets:
            ordered_targets.append(t)
    pivot = plot_df.pivot(index='target', columns='dropped_position', values='drop_from_reference_metric').reindex(index=ordered_targets, columns=[p for p in POSITION_ORDER if p in plot_df['dropped_position'].unique()])
    vmax = float(np.nanmax(np.abs(pivot.to_numpy()))) if np.isfinite(pivot.to_numpy()).any() else 0.01
    vmax = max(vmax, 1e-4)
    fig, ax = plt.subplots(figsize=(8.6, max(5.6, 0.44 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu', norm=_safe_center_norm(-vmax, vmax, 0.0), linewidths=_cfg_attr(plot_cfg, 'heatmap_grid_linewidth', 0.8), linecolor=_cfg_attr(plot_cfg, 'heatmap_grid_color', '#F2F2F2'), cbar_kws={'label': value_label}, ax=ax)
    ax.set_xlabel('Dropped position')
    ax.set_ylabel('')
    _format_heatmap_xticklabels(ax)
    _apply_axes_style(ax, plot_cfg=plot_cfg, show_grid=False)
    _save_fig(fig, out_path, dpi=plot_cfg.dpi)


# -----------------------------------------------------------------------------
# Composite renderer helpers
# -----------------------------------------------------------------------------

def render_main_figures(root: Path, fig_dir: Path, plot_cfg: PlotConfig, created: List[str]) -> None:
    print_banner('Rendering main figure panels and composites')

    # Figure 1 panels
    global_summary = _safe_read_csv_rel(root, 'global_alignment/global_alignment_summary.csv')
    bootstrap_global_summary = _safe_read_csv_rel(root, 'robustness/bootstrap/global_alignment/bootstrap_global_alignment_summary.csv')
    adjusted_summary = _safe_read_csv_rel(root, 'robustness/adjusted/adjusted_global_alignment_summary.csv')

    fig1_panel_paths: List[Path] = []
    if global_summary is not None:
        fp = fig_dir / 'figure_01A_global_alignment_summary.png'
        plot_global_alignment_summary_bar(global_summary, fp, plot_cfg)
        if fp.exists():
            created.append(str(fp))
            fig1_panel_paths.append(fp)
    if bootstrap_global_summary is not None:
        fp = fig_dir / 'figure_01B_global_alignment_bootstrap_forest.png'
        plot_global_alignment_bootstrap_forest(bootstrap_global_summary, fp, plot_cfg)
        if fp.exists():
            created.append(str(fp))
            fig1_panel_paths.append(fp)
    if adjusted_summary is not None and len(adjusted_summary) > 0:
        fp = fig_dir / 'figure_01C_global_alignment_adjusted.png'
        plot_global_alignment_adjusted_bar(adjusted_summary, fp, plot_cfg)
        if fp.exists():
            created.append(str(fp))
            fig1_panel_paths.append(fp)
    if len(fig1_panel_paths) >= 2:
        fp = fig_dir / 'figure_01_global_alignment_composite.png'
        plot_composite_global_alignment(fig1_panel_paths, fp)
        if fp.exists():
            created.append(str(fp))

    # Figure 2 panels
    global_main = _safe_read_csv_rel(root, 'interpretability/variablewise_consolidation/variablewise_global_main_table.csv')
    if global_main is None:
        global_main = _safe_read_csv_rel(root, 'global_alignment/variablewise_global_alignment_summary.csv')
    neighborhood_var = _safe_read_csv_rel(root, 'neighborhood/neighborhood_variable_summary.csv')
    retrieval_summary = _safe_read_csv_rel(root, 'retrieval/retrieval_summary.csv')
    var_group_map = _extract_variable_group_map(root)

    fig2_panel_paths: List[Path] = []
    if global_main is not None and len(global_main) > 0:
        fp = fig_dir / 'figure_02A_variablewise_global_top.png'
        plot_variablewise_fdr_topn(global_main, fp, plot_cfg, top_n=12)
        if fp.exists():
            created.append(str(fp))
            fig2_panel_paths.append(fp)
    if neighborhood_var is not None and retrieval_summary is not None and len(neighborhood_var) > 0 and len(retrieval_summary) > 0:
        fp = fig_dir / 'figure_02B_variable_neighbor_bubble.png'
        plot_variable_neighbor_bubble(neighborhood_var, retrieval_summary, var_group_map, fp, plot_cfg, top_n=10)
        if fp.exists():
            created.append(str(fp))
            fig2_panel_paths.append(fp)
    if len(fig2_panel_paths) == 2:
        fp = fig_dir / 'figure_02_variable_interpretation_composite.png'
        plot_composite_variable_interpretation(fig2_panel_paths, fp)
        if fp.exists():
            created.append(str(fp))

    # Figure 3 panels
    discovery_val = _safe_read_csv_rel(root, 'robustness/discovery_validation/discovery_validation_summary.csv')
    baseline_summary = _safe_read_csv_rel(root, 'robustness/baselines/baseline_comparison_summary.csv')
    distance_space = _safe_read_csv_rel(root, 'robustness/baselines/distance_space_comparison.csv')

    fig3_panel_paths: List[Path] = []
    if discovery_val is not None and len(discovery_val) > 0:
        fp = fig_dir / 'figure_03A_discovery_validation_paired.png'
        plot_discovery_validation_paired_standalone(discovery_val, fp, plot_cfg, metric_col='global_spearman_best_group')
        if fp.exists():
            created.append(str(fp))
            fig3_panel_paths.append(fp)
    if baseline_summary is not None and len(baseline_summary) > 0:
        fp = fig_dir / 'figure_03B_baseline_connected_dot.png'
        plot_baseline_connected_dot_standalone(baseline_summary, fp, plot_cfg, value_col='spearman_r')
        if fp.exists():
            created.append(str(fp))
            fig3_panel_paths.append(fp)
    if distance_space is not None and len(distance_space) > 0:
        fp = fig_dir / 'figure_03C_distance_space_similarity_heatmap.png'
        plot_distance_space_heatmap_standalone(distance_space, fp, plot_cfg, metric='spearman_r')
        if fp.exists():
            created.append(str(fp))
            fig3_panel_paths.append(fp)
    if len(fig3_panel_paths) == 3:
        fp = fig_dir / 'figure_03_robustness_composite.png'
        plot_composite_robustness(fig3_panel_paths, fp)
        if fp.exists():
            created.append(str(fp))


def render_supplementary_figures(root: Path, fig_dir: Path, plot_cfg: PlotConfig, created: List[str]) -> None:
    print_banner('Rendering supplementary figures')
    # 1) Global scatter triptych
    D_ac = _safe_load_npy_rel(root, 'prepared/acoustic_distance.npy')
    group_to_clinical = {
        'function': _safe_load_npy_rel(root, 'global_alignment/clinical_distance_function.npy'),
        'structure': _safe_load_npy_rel(root, 'global_alignment/clinical_distance_structure.npy'),
        'burden': _safe_load_npy_rel(root, 'global_alignment/clinical_distance_burden.npy'),
    }
    if D_ac is not None and any(v is not None for v in group_to_clinical.values()):
        fp = fig_dir / 'supp_global_scatter_triptych.png'
        plot_global_scatter_triptych(D_ac, group_to_clinical, fp, plot_cfg)
        if fp.exists():
            created.append(str(fp))

    # 2) Raw bootstrap raincloud
    raw_boot = _safe_read_csv_rel(root, 'robustness/bootstrap/global_alignment/bootstrap_global_alignment_raw.csv')
    if raw_boot is not None:
        fp = fig_dir / 'supp_bootstrap_global_raincloud.png'
        plot_bootstrap_global_raincloud(raw_boot, fp, plot_cfg)
        if fp.exists():
            created.append(str(fp))

    # 3) Full variablewise global alignment
    vw_all = _safe_read_csv_rel(root, 'global_alignment/variablewise_global_alignment_summary.csv')
    if vw_all is not None:
        fp = fig_dir / 'supp_variablewise_global_alignment_all.png'
        plot_variablewise_global_alignment_all(vw_all, fp, plot_cfg)
        if fp.exists():
            created.append(str(fp))

    # 4) Retrieval summary supplementary
    retrieval_summary = _safe_read_csv_rel(root, 'retrieval/retrieval_summary.csv')
    if retrieval_summary is not None:
        fp = fig_dir / 'supp_retrieval_continuous_top10_heatmap.png'
        plot_retrieval_continuous_heatmap(retrieval_summary, fp, plot_cfg, top_n=10, full=False)
        if fp.exists():
            created.append(str(fp))
        fp = fig_dir / 'supp_retrieval_continuous_full_heatmap.png'
        plot_retrieval_continuous_heatmap(retrieval_summary, fp, plot_cfg, top_n=20, full=True)
        if fp.exists():
            created.append(str(fp))
        fp = fig_dir / 'supp_retrieval_categorical_delta_heatmap.png'
        plot_retrieval_categorical_delta_heatmap(retrieval_summary, fp, plot_cfg, top_n=_cfg_attr(plot_cfg, 'retrieval_heatmap_top_n_categorical', 8))
        if fp.exists():
            created.append(str(fp))

    # 5) Position contribution summary and heatmaps
    var_group_map = _extract_variable_group_map(root)
    pos_g = _safe_read_csv_rel(root, 'interpretability/position_contribution/position_leave_one_out_global_summary.csv')
    pos_n = _safe_read_csv_rel(root, 'interpretability/position_contribution/position_leave_one_out_neighborhood_summary.csv')
    pos_r = _safe_read_csv_rel(root, 'interpretability/position_contribution/position_leave_one_out_retrieval_summary.csv')
    if pos_g is not None or pos_n is not None or pos_r is not None:
        fp = fig_dir / 'supp_position_contribution_faceted_dot.png'
        plot_position_contribution_faceted_dot(pos_g, pos_n, pos_r, var_group_map, fp, plot_cfg)
        if fp.exists():
            created.append(str(fp))
    if pos_r is not None:
        fp = fig_dir / 'supp_position_leave_one_out_retrieval_continuous_heatmap.png'
        plot_position_contribution_retrieval_heatmap(pos_r, fp, plot_cfg, var_group_map, target_type='continuous', top_n=10)
        if fp.exists():
            created.append(str(fp))
        fp = fig_dir / 'supp_position_leave_one_out_retrieval_categorical_delta_heatmap.png'
        plot_position_contribution_retrieval_heatmap(pos_r, fp, plot_cfg, var_group_map, target_type='binary', top_n=8)
        if fp.exists():
            created.append(str(fp))

    # 6) Raw individual robustness panels if needed
    baseline_summary = _safe_read_csv_rel(root, 'robustness/baselines/baseline_comparison_summary.csv')
    if baseline_summary is not None:
        set_publication_plot_style(plot_cfg)
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        color_map = plot_baseline_connected_dot(baseline_summary, ax, value_col='spearman_r')
        if color_map:
            pairs = [(Line2D([0], [0], marker='o', linestyle='None', color=c, markersize=8), sp) for sp, c in color_map.items()]
            handles, labels = _legend_from_pairs(pairs)
            ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
            fig.subplots_adjust(right=0.80)
        fp = fig_dir / 'supp_baseline_connected_dot.png'
        _save_fig(fig, fp, dpi=plot_cfg.dpi)
        if fp.exists():
            created.append(str(fp))

    dist_df = _safe_read_csv_rel(root, 'robustness/baselines/distance_space_comparison.csv')
    if dist_df is not None:
        set_publication_plot_style(plot_cfg)
        fig, ax = plt.subplots(figsize=(7.6, 6.4))
        plot_distance_space_correlation_heatmap(dist_df, 'pearson_r', ax, plot_cfg, annot=True)
        fp = fig_dir / 'supp_distance_space_pearson_heatmap.png'
        _save_fig(fig, fp, dpi=plot_cfg.dpi)
        if fp.exists():
            created.append(str(fp))


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def render_all(root: Path, fig_dir: Path, plot_cfg: PlotConfig) -> List[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    created: List[str] = []
    render_main_figures(root, fig_dir, plot_cfg, created)
    render_supplementary_figures(root, fig_dir, plot_cfg, created)
    manifest = {
        'root': str(root),
        'figure_dir': str(fig_dir),
        'n_figures': len(created),
        'files': created,
    }
    with (fig_dir / 'plot_manifest.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log_done(f'Finished rendering {len(created)} figures into: {fig_dir}')
    return created


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Redraw publication-ready clinical-alignment figures from saved result tables / matrices.')
    p.add_argument('--out-root', type=str, required=True, help='Clinical_alignment output root containing prepared/, global_alignment/, retrieval/, robustness/, interpretability/ ...')
    p.add_argument('--fig-dir', type=str, default=None, help='Folder to save regenerated figures. Default: <out-root>/all_redrawn_figures')
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
