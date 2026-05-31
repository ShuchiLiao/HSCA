from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


DEFAULT_SRC_SR = 8000
DEFAULT_POSITIONS: Tuple[str, ...] = ("A", "E", "M", "P", "T")
PATIENT_ID_CANDIDATES = ("patient_id", "pid", "PID", "编码", "id", "ID")
REQUIRED_WINDOW_INDEX_COLUMNS = (
    "patient_id",
    "position",
    "window_id",
    "window_idx",
    "window_library_path",
)
REQUIRED_RECORDING_MANIFEST_COLUMNS = (
    "patient_id",
    "position",
    "window_library_path",
)


class WindowLibraryCache:
    """Cache loaded ``*_windows.npy`` arrays to avoid repeated np.load calls."""

    def __init__(self) -> None:
        self._cache: Dict[str, np.ndarray] = {}

    def get_window(self, window_library_path: str, window_idx: int) -> np.ndarray:
        """Return one cached window as a float32 1D waveform.

        Parameters
        ----------
        window_library_path:
            Path to one ``*_windows.npy`` file. The file is expected to contain
            a 2D array with shape ``[n_windows, T]``.
        window_idx:
            Index of the requested window within that file.

        Returns
        -------
        np.ndarray
            1D waveform with dtype ``float32`` and shape ``[T]``.
        """
        path_key = str(window_library_path)
        if path_key not in self._cache:
            arr = np.load(path_key, allow_pickle=False)
            if arr.ndim != 2:
                raise ValueError(
                    f"Expected 2D window library array, but got shape={arr.shape} "
                    f"from: {window_library_path}"
                )
            self._cache[path_key] = arr

        arr = self._cache[path_key]
        idx = int(window_idx)
        if idx < 0 or idx >= arr.shape[0]:
            raise IndexError(
                f"window_idx={idx} out of range for array with {arr.shape[0]} windows: "
                f"{window_library_path}"
            )

        waveform = arr[idx]
        if waveform.ndim != 1:
            raise ValueError(
                f"Expected one window to be 1D, but got shape={waveform.shape} "
                f"from: {window_library_path}[{idx}]"
            )
        return np.asarray(waveform, dtype=np.float32)


class WindowDataset(Dataset):
    """Read fixed windows from the prepared window library.

    This dataset only handles window loading and metadata filtering.
    It does not perform resampling, spectrogram conversion, or embedding.
    """

    def __init__(
        self,
        window_index_csv: str,
        recording_manifest_csv: str,
        patient_ids: Optional[List[str]] = None,
        positions: Tuple[str, ...] = DEFAULT_POSITIONS,
    ) -> None:
        self.window_index_csv = str(window_index_csv)
        self.recording_manifest_csv = str(recording_manifest_csv)
        self._window_index_root = Path(self.window_index_csv).resolve().parent
        self._recording_manifest_root = Path(self.recording_manifest_csv).resolve().parent
        self.positions = tuple(str(p).upper() for p in positions)
        self.src_sr = DEFAULT_SRC_SR
        self.cache = WindowLibraryCache()

        window_df = pd.read_csv(self.window_index_csv)
        manifest_df = pd.read_csv(self.recording_manifest_csv)

        self._validate_required_columns(window_df, REQUIRED_WINDOW_INDEX_COLUMNS, "window_index_csv")
        self._validate_required_columns(
            manifest_df,
            REQUIRED_RECORDING_MANIFEST_COLUMNS,
            "recording_manifest_csv",
        )

        window_df["patient_id"] = window_df["patient_id"].astype(str).str.strip()
        window_df["position"] = window_df["position"].astype(str).str.upper().str.strip()
        window_df["window_idx"] = window_df["window_idx"].astype(int)
        window_df["window_id"] = window_df["window_id"].astype(str)
        window_df["window_library_path"] = window_df["window_library_path"].astype(str)

        manifest_df["patient_id"] = manifest_df["patient_id"].astype(str).str.strip()
        manifest_df["position"] = manifest_df["position"].astype(str).str.upper().str.strip()
        manifest_df["window_library_path"] = manifest_df["window_library_path"].astype(str)

        if patient_ids is not None:
            patient_set = {str(pid).strip() for pid in patient_ids}
            window_df = window_df[window_df["patient_id"].isin(patient_set)]

        if self.positions:
            window_df = window_df[window_df["position"].isin(self.positions)]

        # Keep only rows that also exist in recording_manifest.csv.
        # We validate at the (patient_id, position) level rather than exact path
        # string equality, because relative path strings may differ across folders
        # while still referring to the same saved window library file.
        valid_keys = set(zip(manifest_df["patient_id"], manifest_df["position"]))
        row_keys = list(zip(window_df["patient_id"], window_df["position"]))
        keep_mask = [key in valid_keys for key in row_keys]
        window_df = window_df.loc[keep_mask].copy()

        if len(window_df) == 0:
            raise ValueError("No windows remain after filtering by patient_ids / positions / manifest consistency.")

        window_df = window_df.reset_index(drop=True)
        self.df = window_df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[int(idx)]
        resolved_path = self._resolve_window_library_path(str(row["window_library_path"]))
        waveform = self.cache.get_window(str(resolved_path), int(row["window_idx"]))

        return {
            "patient_id": str(row["patient_id"]),
            "position": str(row["position"]),
            "window_id": str(row["window_id"]),
            "window_idx": int(row["window_idx"]),
            "window_library_path": str(resolved_path),
            "waveform": waveform,          # [T], float32
            "src_sr": self.src_sr,         # fixed at 8000
        }

    def _resolve_window_library_path(self, raw_path: str) -> Path:
        """Resolve a window library path robustly.

        The prepared CSVs often store relative paths such as
        ``Data_representation/window_library/...``. In the current project,
        these files live under the ``Data_processing`` folder, so we try a small
        set of sensible candidate roots.
        """
        p = Path(raw_path)
        candidates = []

        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(p)
            candidates.append(self._window_index_root / p)
            candidates.append(self._recording_manifest_root / p)
            candidates.append(self._window_index_root.parent / p)
            candidates.append(self._recording_manifest_root.parent / p)

        for cand in candidates:
            if cand.exists():
                return cand.resolve()

        raise FileNotFoundError(
            f"Cannot resolve window_library_path={raw_path!r}. Tried: "
            + "; ".join(str(c) for c in candidates)
        )

    @staticmethod
    def _validate_required_columns(df: pd.DataFrame, required: Tuple[str, ...], name: str) -> None:
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {name}: {missing}. Available columns: {list(df.columns)}")


def load_patient_ids(patient_list_path: str) -> List[str]:
    """Load patient ids from a txt or csv file.

    Supported formats
    -----------------
    1) TXT: one patient id per non-empty line.
    2) CSV: one column named like patient_id / pid / 编码 / id.
             If no known column name exists and the file has exactly one column,
             that single column will be used.
    """
    path = Path(patient_list_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not ids:
            raise ValueError(f"No patient ids found in txt file: {patient_list_path}")
        return ids

    if suffix == ".csv":
        df = pd.read_csv(path, dtype=str)
        if df.empty:
            raise ValueError(f"CSV file is empty: {patient_list_path}")

        col_name = None
        for candidate in PATIENT_ID_CANDIDATES:
            if candidate in df.columns:
                col_name = candidate
                break
        if col_name is None:
            if len(df.columns) == 1:
                col_name = df.columns[0]
            else:
                raise ValueError(
                    f"Cannot find patient id column in {patient_list_path}. "
                    f"Available columns: {list(df.columns)}"
                )

        ids = (
            df[col_name]
            .dropna()
            .astype(str)
            .str.strip()
        )
        ids = ids[ids != ""].tolist()
        if not ids:
            raise ValueError(f"No patient ids found in csv file: {patient_list_path}")
        return ids

    raise ValueError(
        f"Unsupported patient list file type: {patient_list_path}. "
        f"Only .txt and .csv are supported."
    )
