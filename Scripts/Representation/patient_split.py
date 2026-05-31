from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_patient_ids(window_index_csv: str, recording_manifest_csv: str) -> list[str]:
    win_df = pd.read_csv(window_index_csv, dtype={"patient_id": str})
    rec_df = pd.read_csv(recording_manifest_csv, dtype={"patient_id": str})

    if "patient_id" not in win_df.columns:
        raise ValueError("window_index.csv 缺少 patient_id 列")
    if "patient_id" not in rec_df.columns:
        raise ValueError("recording_manifest.csv 缺少 patient_id 列")

    win_ids = set(win_df["patient_id"].astype(str).str.strip())
    rec_ids = set(rec_df["patient_id"].astype(str).str.strip())

    patient_ids = sorted(win_ids & rec_ids)
    if len(patient_ids) == 0:
        raise ValueError("没有找到共同的 patient_id")

    return patient_ids


def save_list(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for v in values:
            f.write(f"{v}\n")


def split_patients(
    patient_ids: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"train/val/test 比例之和必须为 1，目前为 {total}")

    patient_ids = np.array(patient_ids, dtype=str)

    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    train_ids = patient_ids[:n_train].tolist()
    val_ids = patient_ids[n_train:n_train + n_val].tolist()
    test_ids = patient_ids[n_train + n_val:].tolist()

    return train_ids, val_ids, test_ids


def main():
    parser = argparse.ArgumentParser(description="Make patient-wise train/val/test split")
    parser.add_argument("--window-lib-path", required=True)
    # parser.add_argument("--window-index-csv", required=True)
    # parser.add_argument("--recording-manifest-csv", required=True)
    parser.add_argument("--out-dir", default="splits")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    csv_path = args.window_lib_path
    patient_ids = load_patient_ids(
        window_index_csv=f"{csv_path}/window_index.csv",
        recording_manifest_csv=f"{csv_path}/recording_manifest.csv",
    )

    train_ids, val_ids, test_ids = split_patients(
        patient_ids=patient_ids,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    save_list(out_dir / "all_patients.txt", patient_ids)
    save_list(out_dir / "train_patients.txt", train_ids)
    save_list(out_dir / "val_patients.txt", val_ids)
    save_list(out_dir / "test_patients.txt", test_ids)

    print(f"Total patients: {len(patient_ids)}")
    print(f"Train: {len(train_ids)}")
    print(f"Val:   {len(val_ids)}")
    print(f"Test:  {len(test_ids)}")
    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()


#python patient_split.py --window-lib-path ../Data_preprocessing/window_lib_3_3 --out-dir splits_3_3 --train-ratio 0.7 --val-ratio 0.1 --test-ratio 0.2 --seed 42
#python patient_split.py --window-lib-path ../Data_preprocessing/window_lib_4_4 --out-dir splits_4_4 --train-ratio 0.7 --val-ratio 0.1 --test-ratio 0.2 --seed 42
#python patient_split.py --window-lib-path ../Data_preprocessing/window_lib_5_5 --out-dir splits_5_5 --train-ratio 0.7 --val-ratio 0.1 --test-ratio 0.2 --seed 42
