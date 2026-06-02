
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
import constants

POSITION_ORDER: Tuple[str, ...] = ("A", "E", "M", "P", "T")
WINDOW_ORDER: Tuple[str, ...] = ("2_5_3_3", "2_5_4_4", "2_5_5_5")

ROUTE_DISPLAY_NAME = {
    "beats": "BEATs",
    "panns": "PANNs",
    "ast": "AST",
    "ead": "EAD",
}

ROUTE_PALETTE = {
    "beats": "#1f7a8c",      # teal-blue
    "panns": "#f28e2b",      # warm orange
    "ast": "#4e79a7",        # cool blue
    # "beats_adapt": "#b07aa1",# purple
    # "byola": "#59a14f",      # green
    "ead": "#9c755f",        # muted brown
}
WINDOW_PALETTE = {
    "2_5_3_3": "#1f7a8c",
    "2_5_4_4": "#7b6fd0",
    "2_5_5_5": "#f28e2b",
}
METRIC_PALETTE = {
    "rank1": "#1f7a8c",
    "top5": "#7b6fd0",
    "mAP": "#f28e2b",
}


def set_publication_style() -> None:
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 21,
            "axes.titlesize": 24,
            "axes.labelsize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 19,
            "figure.titlesize": 24,
            "axes.grid": False,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
        }
    )


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.6)
        ax.spines[side].set_color("#222222")
    ax.tick_params(axis="both", width=1.4, length=6)


def _save_figure(fig: plt.Figure, out_stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _require_columns(df: pd.DataFrame, required_cols: List[str], df_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def _validate_meta_and_embeddings(meta_df: pd.DataFrame, embeddings: np.ndarray, name: str) -> None:
    if len(meta_df) != embeddings.shape[0]:
        raise ValueError(
            f"{name} size mismatch: len(meta_df)={len(meta_df)} vs embeddings.shape[0]={embeddings.shape[0]}"
        )

def _resolve_route_setting_dir(embedding_root: Path, route_name: str, window_setting: str) -> Path:
    """Resolve one embedding directory.

    Expected layout:
        Outputs/representation/Embeddings/{route_name}/{window_setting}/

    Example:
        Outputs/representation/Embeddings/beats/4_5_4_1/
    """
    return embedding_root / route_name / window_setting


def _load_route_window_embeddings(route_dir: str) -> tuple[pd.DataFrame, np.ndarray]:
    route_path = Path(route_dir)
    meta = pd.read_csv(route_path / "window_meta.csv")
    emb = np.load(route_path / "window_embeddings.npy", allow_pickle=False).astype(np.float32)

    _require_columns(meta, ["patient_id", "position"], "window_meta")
    _validate_meta_and_embeddings(meta, emb, "window-level")

    meta = meta.copy().reset_index(drop=True)
    meta["patient_id"] = meta["patient_id"].astype(str)
    meta["position"] = meta["position"].astype(str).str.upper()
    meta["_row_idx"] = np.arange(len(meta), dtype=np.int64)
    return meta, emb


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return x / norms


def _build_patient_position_index(
    meta_df: pd.DataFrame,
    patient_subset: List[str] | None = None,
    min_windows_per_position: int = 2,
) -> tuple[List[str], Dict[str, Dict[str, np.ndarray]]]:
    patient_ids = sorted(meta_df["patient_id"].astype(str).unique().tolist())
    if patient_subset is not None:
        subset = set(patient_subset)
        patient_ids = [pid for pid in patient_ids if pid in subset]

    index_map: Dict[str, Dict[str, np.ndarray]] = {}
    for pid in patient_ids:
        g = meta_df[meta_df["patient_id"] == pid]
        pos_map: Dict[str, np.ndarray] = {}
        for pos, gp in g.groupby("position", sort=False):
            row_idx = np.sort(gp["_row_idx"].to_numpy(dtype=np.int64))
            pos_map[str(pos)] = row_idx

        missing = [p for p in POSITION_ORDER if p not in pos_map]
        if missing:
            continue

        ok = True
        for pos in POSITION_ORDER:
            if len(pos_map[pos]) < min_windows_per_position:
                ok = False
                break
        if ok:
            index_map[pid] = pos_map

    patient_ids = sorted(index_map.keys())
    return patient_ids, index_map


def _common_patient_ids_across_datasets(
    dataset_meta_dict: Dict[tuple[str, str], pd.DataFrame],
    min_windows_per_position: int = 2,
) -> List[str]:
    common_ids = None
    for _, meta_df in dataset_meta_dict.items():
        pids, _ = _build_patient_position_index(
            meta_df=meta_df,
            patient_subset=None,
            min_windows_per_position=min_windows_per_position,
        )
        pid_set = set(pids)
        common_ids = pid_set if common_ids is None else (common_ids & pid_set)
    return sorted(common_ids) if common_ids is not None else []


def _metrics_from_distance_matrix(dist_mat: np.ndarray) -> tuple[float, float, float]:
    ranked_idx = np.argsort(dist_mat, axis=1)
    ranks = []
    for i in range(dist_mat.shape[0]):
        rank_pos = int(np.where(ranked_idx[i] == i)[0][0]) + 1
        ranks.append(rank_pos)
    ranks = np.asarray(ranks, dtype=np.int64)
    rank1 = float(np.mean(ranks == 1))
    top5 = float(np.mean(ranks <= min(5, dist_mat.shape[1])))
    mAP = float(np.mean(1.0 / ranks))
    return rank1, top5, mAP


def _sample_two_distinct(rng: np.random.Generator, indices: np.ndarray) -> tuple[int, int]:
    pick = rng.choice(indices, size=2, replace=False)
    return int(pick[0]), int(pick[1])


def mc_retrieval_five_view(
    index_map: Dict[str, Dict[str, np.ndarray]],
    patient_ids: List[str],
    embeddings: np.ndarray,
    n_repeats: int,
    random_seed: int,
) -> pd.DataFrame:
    emb = _l2_normalize_rows(embeddings)
    n_patients = len(patient_ids)
    repeat_rows = []

    for repeat_idx in tqdm(range(n_repeats), desc="MC retrieval 1", leave=False):
        rng = np.random.default_rng(int(random_seed) + 1000 * repeat_idx + 17)
        query_by_pos = {p: np.empty((n_patients, emb.shape[1]), dtype=np.float32) for p in POSITION_ORDER}
        gallery_by_pos = {p: np.empty((n_patients, emb.shape[1]), dtype=np.float32) for p in POSITION_ORDER}

        for i, pid in enumerate(patient_ids):
            for pos in POSITION_ORDER:
                q_idx, g_idx = _sample_two_distinct(rng, index_map[pid][pos])
                query_by_pos[pos][i] = emb[q_idx]
                gallery_by_pos[pos][i] = emb[g_idx]

        dist = np.zeros((n_patients, n_patients), dtype=np.float32)
        for pos in POSITION_ORDER:
            sim = query_by_pos[pos] @ gallery_by_pos[pos].T
            dist += (1.0 - sim).astype(np.float32)
        dist /= float(len(POSITION_ORDER))

        rank1, top5, mAP = _metrics_from_distance_matrix(dist)
        repeat_rows.append({"task": "five_view", "repeat_idx": int(repeat_idx), "rank1": rank1, "top5": top5, "mAP": mAP})

    return pd.DataFrame(repeat_rows)


def mc_retrieval_single_to_four(
    index_map: Dict[str, Dict[str, np.ndarray]],
    patient_ids: List[str],
    embeddings: np.ndarray,
    n_repeats: int,
    random_seed: int,
) -> pd.DataFrame:
    emb = _l2_normalize_rows(embeddings)
    n_patients = len(patient_ids)
    repeat_rows = []

    for query_pos_idx, query_pos in enumerate(POSITION_ORDER):
        other_positions = [p for p in POSITION_ORDER if p != query_pos]
        for repeat_idx in tqdm(range(n_repeats), desc=f"MC retrieval 2 [{query_pos}->others]", leave=False):
            rng = np.random.default_rng(int(random_seed) + 10000 * query_pos_idx + 1000 * repeat_idx + 97)
            query_mat = np.empty((n_patients, emb.shape[1]), dtype=np.float32)
            gallery_by_pos = {p: np.empty((n_patients, emb.shape[1]), dtype=np.float32) for p in other_positions}

            for i, pid in enumerate(patient_ids):
                q_idx = int(rng.choice(index_map[pid][query_pos], size=1, replace=False)[0])
                query_mat[i] = emb[q_idx]
                for pos in other_positions:
                    g_idx = int(rng.choice(index_map[pid][pos], size=1, replace=False)[0])
                    gallery_by_pos[pos][i] = emb[g_idx]

            dist = np.zeros((n_patients, n_patients), dtype=np.float32)
            for pos in other_positions:
                sim = query_mat @ gallery_by_pos[pos].T
                dist += (1.0 - sim).astype(np.float32)
            dist /= float(len(other_positions))

            rank1, top5, mAP = _metrics_from_distance_matrix(dist)
            repeat_rows.append({
                "task": "single_to_four",
                "query_position": query_pos,
                "repeat_idx": int(repeat_idx),
                "rank1": rank1,
                "top5": top5,
                "mAP": mAP,
            })

    return pd.DataFrame(repeat_rows)


def mc_retrieval_pairwise_position(
    index_map: Dict[str, Dict[str, np.ndarray]],
    patient_ids: List[str],
    embeddings: np.ndarray,
    n_repeats: int,
    random_seed: int,
) -> pd.DataFrame:
    emb = _l2_normalize_rows(embeddings)
    n_patients = len(patient_ids)
    repeat_rows = []

    for q_idx, query_pos in enumerate(POSITION_ORDER):
        for g_idx, gallery_pos in enumerate(POSITION_ORDER):
            if gallery_pos == query_pos:
                continue
            for repeat_idx in tqdm(range(n_repeats), desc=f"MC pairwise [{query_pos}->{gallery_pos}]", leave=False):
                rng = np.random.default_rng(int(random_seed) + 100000 * q_idx + 10000 * g_idx + 1000 * repeat_idx + 131)
                query_mat = np.empty((n_patients, emb.shape[1]), dtype=np.float32)
                gallery_mat = np.empty((n_patients, emb.shape[1]), dtype=np.float32)

                for i, pid in enumerate(patient_ids):
                    q_row = int(rng.choice(index_map[pid][query_pos], size=1, replace=False)[0])
                    g_row = int(rng.choice(index_map[pid][gallery_pos], size=1, replace=False)[0])
                    query_mat[i] = emb[q_row]
                    gallery_mat[i] = emb[g_row]

                sim = query_mat @ gallery_mat.T
                dist = (1.0 - sim).astype(np.float32)

                rank1, top5, mAP = _metrics_from_distance_matrix(dist)
                repeat_rows.append({
                    "task": "pairwise_position",
                    "query_position": query_pos,
                    "gallery_position": gallery_pos,
                    "repeat_idx": int(repeat_idx),
                    "rank1": rank1,
                    "top5": top5,
                    "mAP": mAP,
                })
    return pd.DataFrame(repeat_rows)


def summarize_route_window(route_name: str, window_setting: str, five_view_df: pd.DataFrame, single_to_four_df: pd.DataFrame):
    summary = {
        "window_setting": window_setting,
        "route_name": route_name,
        "five_view_rank1": float(five_view_df["rank1"].mean()),
        "five_view_top5": float(five_view_df["top5"].mean()),
        "five_view_mAP": float(five_view_df["mAP"].mean()),
        "single_to_four_rank1": float(single_to_four_df["rank1"].mean()),
        "single_to_four_top5": float(single_to_four_df["top5"].mean()),
        "single_to_four_mAP": float(single_to_four_df["mAP"].mean()),
    }
    per_pos = (
        single_to_four_df.groupby("query_position", as_index=False)[["rank1", "top5", "mAP"]]
        .mean()
        .rename(columns={"rank1": "single_to_four_rank1", "top5": "single_to_four_top5", "mAP": "single_to_four_mAP"})
    )
    per_pos.insert(0, "route_name", route_name)
    per_pos.insert(0, "window_setting", window_setting)
    return summary, per_pos


def summarize_route_across_windows(summary_by_window_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for route_name, sub in summary_by_window_df.groupby("route_name", sort=False):
        row = {"route_name": route_name}
        for col in ["five_view_rank1", "five_view_top5", "five_view_mAP",
                    "single_to_four_rank1", "single_to_four_top5", "single_to_four_mAP"]:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_std"] = float(sub[col].std(ddof=0))
        rows.append(row)
    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["five_view_rank1_mean", "five_view_mAP_mean", "single_to_four_rank1_mean", "single_to_four_mAP_mean", "route_name"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def summarize_pairwise_matrix(route_name: str, window_setting: str, pairwise_df: pd.DataFrame) -> pd.DataFrame:
    out = pairwise_df.groupby(["query_position", "gallery_position"], as_index=False)[["rank1", "top5", "mAP"]].mean()
    out.insert(0, "route_name", route_name)
    out.insert(0, "window_setting", window_setting)
    return out


def _window_ordered(df: pd.DataFrame, window_order: List[str]) -> pd.DataFrame:
    out = df.copy()
    out["window_setting"] = pd.Categorical(
        out["window_setting"],
        categories=list(window_order),
        ordered=True,
    )
    return out.sort_values("window_setting")


def _route_label(route_name: str) -> str:
    return ROUTE_DISPLAY_NAME.get(route_name, route_name)


def plot_window_sensitivity(
    summary_by_window_df: pd.DataFrame,
    window_order: List[str],
    value_col: str,
    title: str,
    ylabel: str,
    out_stem: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.5))
    df = _window_ordered(summary_by_window_df, window_order)
    for route_name, sub in df.groupby("route_name", sort=False):
        ax.plot(
            sub["window_setting"].astype(str),
            sub[value_col].to_numpy(dtype=np.float32),
            marker="o", markersize=9, linewidth=3.0,
            label=_route_label(route_name), color=ROUTE_PALETTE.get(route_name, "#1f7a8c"),
        )
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    ax.legend(frameon=False, loc="best")
    _style_axes(ax)
    _save_figure(fig, out_stem)


def plot_position_line(per_position_df: pd.DataFrame, route_order: List[str], value_col: str, ylabel: str, title: str, out_stem: Path) -> None:
    fig, axes = plt.subplots(1, len(route_order), figsize=(6.4 * len(route_order), 6.0), sharey=True)
    if len(route_order) == 1:
        axes = [axes]
    subdf = per_position_df.copy()
    subdf["query_position"] = pd.Categorical(subdf["query_position"], categories=list(POSITION_ORDER), ordered=True)
    subdf = subdf.sort_values("query_position")
    for ax, route_name in zip(axes, route_order):
        sub = subdf[subdf["route_name"] == route_name]
        ax.plot(
            sub["query_position"].astype(str),
            sub[value_col].to_numpy(dtype=np.float32),
            marker="o", markersize=9, linewidth=3.0,
            color=ROUTE_PALETTE.get(route_name, "#1f7a8c"),
        )
        ax.set_title(_route_label(route_name), pad=10)
        ax.set_xlabel("Query position")
        _style_axes(ax)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, y=1.02)
    _save_figure(fig, out_stem)


def plot_pairwise_heatmaps(pairwise_summary_df: pd.DataFrame, route_order: List[str], value_col: str, title: str, out_stem: Path) -> None:
    if pairwise_summary_df is None or pairwise_summary_df.empty or "route_name" not in pairwise_summary_df.columns:
        return
    fig, axes = plt.subplots(1, len(route_order), figsize=(6.4 * len(route_order), 5.8), sharey=True)
    if len(route_order) == 1:
        axes = [axes]
    vmax = float(np.nanmax(pairwise_summary_df[value_col].to_numpy(dtype=np.float32))) if len(pairwise_summary_df) else 1.0
    vmax = max(vmax, 0.2)
    for ax, route_name in zip(axes, route_order):
        sub = pairwise_summary_df[pairwise_summary_df["route_name"] == route_name].copy()
        heat = pd.DataFrame(np.nan, index=list(POSITION_ORDER), columns=list(POSITION_ORDER))
        for _, row in sub.iterrows():
            heat.loc[row["query_position"], row["gallery_position"]] = float(row[value_col])
        sns.heatmap(
            heat, ax=ax, cmap="YlGnBu", vmin=0.0, vmax=vmax, annot=True, fmt=".3f",
            cbar=(ax == axes[-1]), square=True, linewidths=1.0, linecolor="white",
            annot_kws={"fontsize": 15},
        )
        ax.set_title(_route_label(route_name), pad=10)
        ax.set_xlabel("Gallery position")
        ax.set_ylabel("Query position")
        _style_axes(ax)
    fig.suptitle(title, y=1.02)
    _save_figure(fig, out_stem)


def make_all_plots(
    summary_by_window_df: pd.DataFrame,
    route_average_df: pd.DataFrame,
    per_position_pairwise_df: pd.DataFrame,
    pairwise_summary_df: pd.DataFrame,
    fig_dir: Path,
    window_order: List[str],
    pairwise_window: str,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    route_order = route_average_df["route_name"].tolist()

    plot_window_sensitivity(
        summary_by_window_df,
        window_order,
        "five_view_rank1",
        "Window-setting sensitivity: five-position Monte Carlo retrieval",
        "Rank-1",
        fig_dir / "fig_01_five_view_rank1_window_sensitivity",
    )
    plot_window_sensitivity(
        summary_by_window_df,
        window_order,
        "five_view_mAP",
        "Window-setting sensitivity: five-position Monte Carlo retrieval",
        "mAP",
        fig_dir / "fig_02_five_view_mAP_window_sensitivity",
    )
    plot_window_sensitivity(
        summary_by_window_df,
        window_order,
        "single_to_four_rank1",
        "Window-setting sensitivity: single-position to four-position retrieval",
        "Rank-1",
        fig_dir / "fig_03_single_to_four_rank1_window_sensitivity",
    )
    plot_window_sensitivity(
        summary_by_window_df,
        window_order,
        "single_to_four_mAP",
        "Window-setting sensitivity: single-position to four-position retrieval",
        "mAP",
        fig_dir / "fig_04_single_to_four_mAP_window_sensitivity",
    )

    plot_position_line(
        per_position_pairwise_df,
        route_order,
        "single_to_four_rank1",
        "Rank-1",
        f"Single-position to four-position retrieval by query position ({pairwise_window})",
        fig_dir / f"fig_05_single_to_four_rank1_by_position_{pairwise_window}",
    )
    plot_position_line(
        per_position_pairwise_df,
        route_order,
        "single_to_four_mAP",
        "mAP",
        f"Single-position to four-position retrieval by query position ({pairwise_window})",
        fig_dir / f"fig_06_single_to_four_mAP_by_position_{pairwise_window}",
    )
    plot_pairwise_heatmaps(
        pairwise_summary_df,
        route_order,
        "rank1",
        f"Pairwise cross-position retrieval matrix (Rank-1, {pairwise_window})",
        fig_dir / f"fig_07_pairwise_rank1_heatmaps_{pairwise_window}",
    )
    plot_pairwise_heatmaps(
        pairwise_summary_df,
        route_order,
        "mAP",
        f"Pairwise cross-position retrieval matrix (mAP, {pairwise_window})",
        fig_dir / f"fig_08_pairwise_mAP_heatmaps_{pairwise_window}",
    )

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo retrieval evaluation with window-length sensitivity probing and pairwise cross-position probing."
    )
    parser.add_argument("--embedding-root", type=str, required=True)
    parser.add_argument("--routes", type=str, nargs="+", default=["beats", "panns", "ast", "ead"])
    parser.add_argument("--window-settings", type=str, nargs="+", required=True)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairwise-window", type=str, default="2_5_4_4") #pairwise cross-position retrieval，会做 A→E、A→M、A→P、A→T、E→A 等两两部位检索矩阵。代码只对 --pairwise-window 指定的那一个 setting 做。
    parser.add_argument("--skip-pairwise", action="store_true")
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args(argv)

    set_publication_style()

    dataset_meta_dict: Dict[tuple[str, str], pd.DataFrame] = {}
    dataset_emb_dict: Dict[tuple[str, str], np.ndarray] = {}

    embedding_root = Path(args.embedding_root)

    for window_setting in args.window_settings:
        for route_name in args.routes:
            route_dir = _resolve_route_setting_dir(
                embedding_root=embedding_root,
                route_name=route_name,
                window_setting=window_setting,
            )
            if not route_dir.exists():
                raise FileNotFoundError(f"Missing embedding directory: {route_dir}")

            meta_df, emb = _load_route_window_embeddings(str(route_dir))
            dataset_meta_dict[(window_setting, route_name)] = meta_df
            dataset_emb_dict[(window_setting, route_name)] = emb

    common_patient_ids = _common_patient_ids_across_datasets(dataset_meta_dict, min_windows_per_position=2)
    if len(common_patient_ids) == 0:
        raise ValueError("No common patients with >=2 windows per position across all datasets.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"

    summary_rows = []
    per_position_rows = []
    five_repeat_rows = []
    single_repeat_rows = []
    pairwise_repeat_rows = []
    pairwise_summary_rows = []

    for window_setting in tqdm(args.window_settings, desc="Window settings"):
        for route_name in tqdm(args.routes, desc=f"Routes@{window_setting}", leave=False):
            meta_df = dataset_meta_dict[(window_setting, route_name)]
            emb = dataset_emb_dict[(window_setting, route_name)]

            patient_ids, index_map = _build_patient_position_index(meta_df, patient_subset=common_patient_ids, min_windows_per_position=2)
            if patient_ids != common_patient_ids:
                raise ValueError(f"Patient mismatch after subsetting for {window_setting}/{route_name}")

            five_df = mc_retrieval_five_view(index_map, patient_ids, emb, int(args.n_repeats), int(args.seed))
            five_df.insert(0, "route_name", route_name)
            five_df.insert(0, "window_setting", window_setting)

            single_df = mc_retrieval_single_to_four(index_map, patient_ids, emb, int(args.n_repeats), int(args.seed))
            single_df.insert(0, "route_name", route_name)
            single_df.insert(0, "window_setting", window_setting)

            summary_row, per_pos_df = summarize_route_window(route_name, window_setting, five_df, single_df)
            summary_rows.append(summary_row)
            per_position_rows.append(per_pos_df)
            five_repeat_rows.append(five_df)
            single_repeat_rows.append(single_df)

            if (not args.skip_pairwise) and window_setting == args.pairwise_window:
                pairwise_df = mc_retrieval_pairwise_position(index_map, patient_ids, emb, int(args.n_repeats), int(args.seed))
                pairwise_df.insert(0, "route_name", route_name)
                pairwise_df.insert(0, "window_setting", window_setting)
                pairwise_repeat_rows.append(pairwise_df)
                pairwise_summary_rows.append(summarize_pairwise_matrix(route_name, window_setting, pairwise_df))

    summary_by_window_df = pd.DataFrame(summary_rows)
    summary_by_window_df["window_setting"] = pd.Categorical(
        summary_by_window_df["window_setting"],
        categories=list(args.window_settings),
        ordered=True,
    )
    summary_by_window_df = summary_by_window_df.sort_values(
        by=["window_setting", "five_view_rank1", "five_view_mAP", "single_to_four_rank1", "single_to_four_mAP", "route_name"],
        ascending=[True, False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    route_average_df = summarize_route_across_windows(summary_by_window_df)
    per_position_df = pd.concat(per_position_rows, ignore_index=True)
    five_repeat_df = pd.concat(five_repeat_rows, ignore_index=True)
    single_repeat_df = pd.concat(single_repeat_rows, ignore_index=True)
    per_position_pairwise_df = per_position_df[
        per_position_df["window_setting"] == args.pairwise_window
        ].copy()

    if pairwise_repeat_rows:
        pairwise_repeat_df = pd.concat(pairwise_repeat_rows, ignore_index=True)
        pairwise_summary_df = pd.concat(pairwise_summary_rows, ignore_index=True)
    else:
        pairwise_repeat_df = pd.DataFrame()
        pairwise_summary_df = pd.DataFrame()

    summary_by_window_df.to_csv(out_dir / "mc_retrieval_summary_by_window.csv", index=False, encoding="utf-8-sig")
    route_average_df.to_csv(out_dir / "mc_retrieval_summary_across_windows.csv", index=False, encoding="utf-8-sig")
    five_repeat_df.to_csv(out_dir / "mc_retrieval_five_view_per_repeat.csv", index=False, encoding="utf-8-sig")
    single_repeat_df.to_csv(out_dir / "mc_retrieval_single_to_four_per_repeat.csv", index=False, encoding="utf-8-sig")
    per_position_df.to_csv(out_dir / "mc_retrieval_single_to_four_by_position.csv", index=False, encoding="utf-8-sig")
    if not pairwise_repeat_df.empty:
        pairwise_repeat_df.to_csv(
            out_dir / f"mc_retrieval_pairwise_position_per_repeat_{args.pairwise_window}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pairwise_summary_df.to_csv(
            out_dir / f"mc_retrieval_pairwise_position_summary_{args.pairwise_window}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    make_all_plots(
        summary_by_window_df=summary_by_window_df,
        route_average_df=route_average_df,
        per_position_pairwise_df=per_position_pairwise_df,
        pairwise_summary_df=pairwise_summary_df,
        fig_dir=fig_dir,
        window_order=list(args.window_settings),
        pairwise_window=args.pairwise_window,
    )

    print(f"Saved summary-by-window to: {out_dir / 'mc_retrieval_summary_by_window.csv'}")
    print(f"Saved route-average summary to: {out_dir / 'mc_retrieval_summary_across_windows.csv'}")
    print(f"Saved five-view per-repeat detail to: {out_dir / 'mc_retrieval_five_view_per_repeat.csv'}")
    print(f"Saved single-to-four per-repeat detail to: {out_dir / 'mc_retrieval_single_to_four_per_repeat.csv'}")
    print(f"Saved single-to-four by-position summary to: {out_dir / 'mc_retrieval_single_to_four_by_position.csv'}")
    if not pairwise_repeat_df.empty:
        print(
            f"Saved pairwise per-repeat detail to: "
            f"{out_dir / f'mc_retrieval_pairwise_position_per_repeat_{args.pairwise_window}.csv'}"
        )
        print(
            f"Saved pairwise summary to: "
            f"{out_dir / f'mc_retrieval_pairwise_position_summary_{args.pairwise_window}.csv'}"
        )
    print(f"Saved figures to: {fig_dir}")
    print(route_average_df.to_string(index=False))


if __name__ == "__main__":

    embedding_root = constants.OUTPUT_FOLDER / "representation" / "Embeddings"
    out_dir = constants.OUTPUT_FOLDER / "representation" / "Selection"

    routes = ["beats", "panns", "ast", "ead"]
    window_settings = ["2_5_3_3", "2_5_4_4", "2_5_5_5"]

    n_repeats = 20
    seed = 42
    pairwise_window = "2_5_4_4"
    skip_pairwise = True

    main_args = ["--embedding-root", str(embedding_root), "--routes", *routes, "--window-settings", *window_settings,
                 "--n-repeats", str(n_repeats), "--seed", str(seed), "--pairwise-window", pairwise_window, "--out-dir", str(out_dir)]
    main_args += ["--skip-pairwise"] if skip_pairwise else []

    main(main_args)
