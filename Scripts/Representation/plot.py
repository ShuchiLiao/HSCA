#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据上传的心音表示学习检索结果表格绘图。

设计原则：
1. 方法总体比较：采用带误差条的分组柱状图，最适合展示离散方法在多项指标上的整体对比；
2. 窗口设置趋势：采用折线图，最适合展示随有序条件（3_3 / 4_4 / 5_5）变化的趋势，
   也便于观察“不同窗口设置下排序是否保持一致”；
3. 稳定性：采用箱线图；
4. 位置敏感性：采用热图。

全局绘图要求：
- seaborn 配色
- 四个边框都保留
- 字号 21-24
- 字体 Arial（若系统无 Arial，则自动回退到 DejaVu Sans / Liberation Sans）

用法：
    python plot.py --input_dir /mnt/data --output_dir /mnt/data/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_FONT = 22
TITLE_FONT = 24
TICK_FONT = 21
LEGEND_FONT = 21
ANNOT_FONT = 21
LINE_WIDTH = 2.6
SPINE_WIDTH = 2.0

ROUTE_ORDER = ["beats", "ast", "panns", "ead"]
METHOD_ORDER_FOR_COMPARISON = ["ead", "panns", "ast", "beats"]

ROUTE_LABELS = {
    "beats": "BEATs",
    "ast": "AST",
    "panns": "PANNs",
    "ead": "EAD",
}
WINDOW_ORDER = ["3_3", "4_4", "5_5"]
POSITION_ORDER = ["A", "E", "M", "P", "T"]
TASK_LABELS = {
    "five_view": "Five-view retrieval",
    "single_to_four": "Single-to-four retrieval",
}


def set_style() -> None:
    sns.set_theme(style="white", palette="deep")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": BASE_FONT,
            "axes.titlesize": TITLE_FONT,
            "axes.labelsize": BASE_FONT,
            "xtick.labelsize": TICK_FONT,
            "ytick.labelsize": TICK_FONT,
            "legend.fontsize": LEGEND_FONT,
            "figure.titlesize": TITLE_FONT,
            "axes.linewidth": SPINE_WIDTH,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )


def ensure_four_spines(ax: plt.Axes) -> None:
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(SPINE_WIDTH)
    ax.tick_params(width=SPINE_WIDTH * 0.8, length=6)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[Saved] {png_path}")
    print(f"[Saved] {pdf_path}")


def load_csv(input_dir: Path, name: str) -> pd.DataFrame:
    path = input_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Cannot find file: {path}")
    return pd.read_csv(path)


def load_all_data(input_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "five_view_per_repeat": load_csv(input_dir, "mc_retrieval_five_view_per_repeat.csv"),
        "single_to_four_per_repeat": load_csv(input_dir, "mc_retrieval_single_to_four_per_repeat.csv"),
        "pairwise_position_per_repeat": load_csv(input_dir, "mc_retrieval_pairwise_position_per_repeat_4_4.csv"),
        "pairwise_position_summary": load_csv(input_dir, "mc_retrieval_pairwise_position_summary_4_4.csv"),
        "single_to_four_by_position": load_csv(input_dir, "mc_retrieval_single_to_four_by_position.csv"),
        "summary_across_windows": load_csv(input_dir, "mc_retrieval_summary_across_windows.csv"),
        "summary_by_window": load_csv(input_dir, "mc_retrieval_summary_by_window.csv"),
    }


def add_route_display(df: pd.DataFrame, col: str = "route_name") -> pd.DataFrame:
    out = df.copy()
    out["route_display"] = out[col].map(ROUTE_LABELS)
    return out


def prettify_route_names(df: pd.DataFrame, col: str = "route_name", order=None) -> pd.DataFrame:
    out = df.copy()
    if order is None:
        order = ROUTE_ORDER
    out[col] = pd.Categorical(out[col], categories=order, ordered=True)
    out = add_route_display(out, col=col)
    return out


def prettify_windows(df: pd.DataFrame, col: str = "window_setting") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.Categorical(out[col], categories=WINDOW_ORDER, ordered=True)
    return out


def plot_overall_four_metric_comparison(summary_across: pd.DataFrame, output_dir: Path) -> None:
    df = summary_across.copy()
    df = prettify_route_names(df, order=METHOD_ORDER_FOR_COMPARISON)
    df = df.sort_values("route_name")

    metric_specs = [
        ("five_view_rank1_mean", "five_view_rank1_std", "Five-view\nRank-1"),
        ("five_view_mAP_mean", "five_view_mAP_std", "Five-view\nmAP"),
        ("single_to_four_rank1_mean", "single_to_four_rank1_std", "Single-to-four\nRank-1"),
        ("single_to_four_mAP_mean", "single_to_four_mAP_std", "Single-to-four\nmAP"),
    ]

    records = []
    for _, row in df.iterrows():
        for mean_col, std_col, metric_label in metric_specs:
            records.append(
                {
                    "route_name": row["route_name"],
                    "route_display": row["route_display"],
                    "metric": metric_label,
                    "mean": row[mean_col],
                    "std": row[std_col],
                }
            )
    long_df = pd.DataFrame(records)
    long_df["route_name"] = pd.Categorical(long_df["route_name"], categories=METHOD_ORDER_FOR_COMPARISON, ordered=True)
    long_df["metric"] = pd.Categorical(long_df["metric"], categories=[m[2] for m in metric_specs], ordered=True)
    long_df = long_df.sort_values(["route_name", "metric"])

    fig, ax = plt.subplots(figsize=(18, 10))
    sns.barplot(data=long_df, x="route_display", y="mean", hue="metric", errorbar=None, ax=ax)

    for patch, (_, row) in zip(ax.patches, long_df.iterrows()):
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        ax.errorbar(
            x=x,
            y=y,
            yerr=row["std"],
            fmt="none",
            ecolor="black",
            elinewidth=1.7,
            capsize=5,
            capthick=1.7,
            zorder=5,
        )

    ax.set_xlabel("Method")
    ax.set_ylabel("Performance")
    ax.set_title("Overall performance on two retrieval tasks")
    ax.set_ylim(0, 1.08)
    ax.legend(title=None, frameon=False, ncol=2, loc="upper left")
    ensure_four_spines(ax)
    save_figure(fig, output_dir, "fig6_overall_four_metric_comparison")


def plot_rank1_vs_window(summary_by_window: pd.DataFrame, output_dir: Path) -> None:
    df = summary_by_window.copy()
    df = prettify_windows(df)
    df = prettify_route_names(df, order=METHOD_ORDER_FOR_COMPARISON)
    df = df.sort_values(["route_name", "window_setting"])

    fig, axes = plt.subplots(1, 2, figsize=(22, 9), sharey=False)

    sns.lineplot(
        data=df,
        x="window_setting",
        y="five_view_rank1",
        hue="route_display",
        style="route_display",
        hue_order=[ROUTE_LABELS[r] for r in METHOD_ORDER_FOR_COMPARISON],
        style_order=[ROUTE_LABELS[r] for r in METHOD_ORDER_FOR_COMPARISON],
        markers=True,
        dashes=False,
        linewidth=LINE_WIDTH,
        markersize=11,
        ax=axes[0],
    )
    axes[0].set_xlabel("Window setting")
    axes[0].set_ylabel("Rank-1")
    axes[0].set_title("Five-view Rank-1 vs window setting")
    axes[0].set_ylim(0, 1.03)
    axes[0].legend(title=None, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ensure_four_spines(axes[0])

    sns.lineplot(
        data=df,
        x="window_setting",
        y="single_to_four_rank1",
        hue="route_display",
        style="route_display",
        hue_order=[ROUTE_LABELS[r] for r in METHOD_ORDER_FOR_COMPARISON],
        style_order=[ROUTE_LABELS[r] for r in METHOD_ORDER_FOR_COMPARISON],
        markers=True,
        dashes=False,
        linewidth=LINE_WIDTH,
        markersize=11,
        ax=axes[1],
    )
    axes[1].set_xlabel("Window setting")
    axes[1].set_ylabel("Rank-1")
    axes[1].set_title("Single-to-four Rank-1 vs window setting")
    axes[1].set_ylim(0, max(0.12, df["single_to_four_rank1"].max() * 1.15))
    axes[1].legend(title=None, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ensure_four_spines(axes[1])

    save_figure(fig, output_dir, "fig7_rank1_vs_window_setting")


def plot_overall_selection(summary_across: pd.DataFrame, output_dir: Path) -> None:
    df = summary_across.copy()
    df = prettify_route_names(df)

    long_df = pd.DataFrame(
        {
            "route_name": list(df["route_name"]) * 2,
            "route_display": list(df["route_display"]) * 2,
            "task": [TASK_LABELS["five_view"]] * len(df) + [TASK_LABELS["single_to_four"]] * len(df),
            "mean": list(df["five_view_mAP_mean"]) + list(df["single_to_four_mAP_mean"]),
            "std": list(df["five_view_mAP_std"]) + list(df["single_to_four_mAP_std"]),
        }
    )
    long_df["route_name"] = pd.Categorical(long_df["route_name"], categories=ROUTE_ORDER, ordered=True)
    long_df = long_df.sort_values(["route_name", "task"])

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.barplot(data=long_df, x="route_display", y="mean", hue="task", errorbar=None, ax=ax)

    for patch, (_, row) in zip(ax.patches, long_df.iterrows()):
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        ax.errorbar(x=x, y=y, yerr=row["std"], fmt="none", ecolor="black", elinewidth=1.8, capsize=5, capthick=1.8)

    ax.set_xlabel("Representation")
    ax.set_ylabel("mAP")
    ax.set_title("Overall comparison across windows")
    ax.set_ylim(0, min(1.05, long_df["mean"].max() + 0.10))
    ax.legend(title=None, frameon=False, loc="upper right")
    ensure_four_spines(ax)
    save_figure(fig, output_dir, "fig1_overall_selection_map")


def plot_window_trends(summary_by_window: pd.DataFrame, output_dir: Path) -> None:
    df = summary_by_window.copy()
    df = prettify_route_names(prettify_windows(df))

    fig, axes = plt.subplots(1, 2, figsize=(22, 9), sharey=False)

    sns.lineplot(
        data=df,
        x="window_setting",
        y="five_view_mAP",
        hue="route_display",
        style="route_display",
        markers=True,
        dashes=False,
        linewidth=LINE_WIDTH,
        markersize=10,
        ax=axes[0],
    )
    axes[0].set_xlabel("Window setting")
    axes[0].set_ylabel("mAP")
    axes[0].set_title("Five-view retrieval")
    axes[0].legend(title=None, frameon=False, loc="lower right")
    ensure_four_spines(axes[0])

    sns.lineplot(
        data=df,
        x="window_setting",
        y="single_to_four_mAP",
        hue="route_display",
        style="route_display",
        markers=True,
        dashes=False,
        linewidth=LINE_WIDTH,
        markersize=10,
        ax=axes[1],
    )
    axes[1].set_xlabel("Window setting")
    axes[1].set_ylabel("mAP")
    axes[1].set_title("Single-to-four retrieval")
    axes[1].legend(title=None, frameon=False, loc="upper left")
    ensure_four_spines(axes[1])

    save_figure(fig, output_dir, "fig2_window_trends_map")


def plot_repeat_stability(five_view_per_repeat: pd.DataFrame, single_to_four_per_repeat: pd.DataFrame, output_dir: Path) -> None:
    df_fv = prettify_route_names(prettify_windows(five_view_per_repeat.copy()))
    df_stf = single_to_four_per_repeat.groupby(["window_setting", "route_name", "repeat_idx"], as_index=False)["mAP"].mean()
    df_stf = prettify_route_names(prettify_windows(df_stf))

    fig, axes = plt.subplots(1, 2, figsize=(24, 9), sharey=False)

    sns.boxplot(data=df_fv, x="route_display", y="mAP", hue="window_setting", linewidth=1.6, fliersize=1.8, ax=axes[0])
    axes[0].set_xlabel("Representation")
    axes[0].set_ylabel("mAP")
    axes[0].set_title("Five-view repeat stability")
    axes[0].legend(title="Window", frameon=False, loc="lower right")
    ensure_four_spines(axes[0])

    sns.boxplot(data=df_stf, x="route_display", y="mAP", hue="window_setting", linewidth=1.6, fliersize=1.8, ax=axes[1])
    axes[1].set_xlabel("Representation")
    axes[1].set_ylabel("mAP")
    axes[1].set_title("Single-to-four repeat stability")
    axes[1].legend(title="Window", frameon=False, loc="upper left")
    ensure_four_spines(axes[1])

    save_figure(fig, output_dir, "fig3_repeat_stability_map")


def plot_beats_position_heatmap(single_to_four_by_position: pd.DataFrame, output_dir: Path) -> None:
    df = single_to_four_by_position.copy()
    df = df[df["route_name"] == "beats"].copy()
    df["query_position"] = pd.Categorical(df["query_position"], categories=POSITION_ORDER, ordered=True)
    df["window_setting"] = pd.Categorical(df["window_setting"], categories=WINDOW_ORDER, ordered=True)

    pivot = df.pivot(index="query_position", columns="window_setting", values="single_to_four_mAP")
    pivot = pivot.loc[POSITION_ORDER, WINDOW_ORDER]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pivot, annot=True, fmt=".3f", linewidths=1.2, cbar_kws={"label": "mAP"}, annot_kws={"size": ANNOT_FONT}, ax=ax)
    ax.set_xlabel("Window setting")
    ax.set_ylabel("Query position")
    ax.set_title("BEATs: single-to-four mAP by query position")
    ensure_four_spines(ax)
    save_figure(fig, output_dir, "fig4_beats_position_heatmap")


def plot_pairwise_heatmaps(pairwise_summary: pd.DataFrame, output_dir: Path) -> None:
    df = pairwise_summary.copy()
    df["route_name"] = pd.Categorical(df["route_name"], categories=ROUTE_ORDER, ordered=True)
    df["query_position"] = pd.Categorical(df["query_position"], categories=POSITION_ORDER, ordered=True)
    df["gallery_position"] = pd.Categorical(df["gallery_position"], categories=POSITION_ORDER, ordered=True)

    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    axes = axes.flatten()

    for ax, route in zip(axes, ROUTE_ORDER):
        sub = df[df["route_name"] == route].copy()
        full = pd.DataFrame(index=POSITION_ORDER, columns=POSITION_ORDER, dtype=float)
        for _, row in sub.iterrows():
            full.loc[row["query_position"], row["gallery_position"]] = row["mAP"]

        mask = full.isna()
        sns.heatmap(
            full,
            mask=mask,
            annot=True,
            fmt=".3f",
            linewidths=1.0,
            cbar=(route == ROUTE_ORDER[-1]),
            cbar_kws={"label": "mAP"},
            annot_kws={"size": 18},
            ax=ax,
        )
        ax.set_xlabel("Gallery position")
        ax.set_ylabel("Query position")
        ax.set_title(ROUTE_LABELS[route])
        ensure_four_spines(ax)

    fig.suptitle("Pairwise position retrieval (window = 4_4)", y=1.02)
    save_figure(fig, output_dir, "fig5_pairwise_position_heatmaps_4_4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot figures for heart sound representation learning.")
    parser.add_argument("--input_dir", type=str, default=".", help="Directory containing the uploaded CSV files.")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save figures.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_style()
    data = load_all_data(input_dir)

    plot_overall_selection(data["summary_across_windows"], output_dir)
    plot_window_trends(data["summary_by_window"], output_dir)
    plot_repeat_stability(data["five_view_per_repeat"], data["single_to_four_per_repeat"], output_dir)
    plot_beats_position_heatmap(data["single_to_four_by_position"], output_dir)
    plot_pairwise_heatmaps(data["pairwise_position_summary"], output_dir)
    plot_overall_four_metric_comparison(data["summary_across_windows"], output_dir)
    plot_rank1_vs_window(data["summary_by_window"], output_dir)

    print("\nAll figures have been generated successfully.")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
