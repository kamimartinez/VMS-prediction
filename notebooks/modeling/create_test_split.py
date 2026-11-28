"""Create a shared chronological train/validation/test split for model evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_string_dtype


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "dataset_final_serie_tiempo.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "test_data.npz"
TIME_COLUMN = "bloque_tiempo"
ROBOT_COLUMN = "ms"
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15


def _to_numpy(values: pd.Series) -> np.ndarray:
    if is_datetime64_any_dtype(values.dtype):
        return values.to_numpy(dtype="datetime64[ns]")
    if is_string_dtype(values.dtype) or values.dtype == object:
        return values.astype("string").fillna("").to_numpy(dtype=str)
    return values.to_numpy()


def create_temporal_split(
    data_path: Path = DATA_PATH,
    output_path: Path = OUTPUT_PATH,
) -> dict[str, int | str]:
    """Save chronologically ordered 70/15/15 raw data partitions to an NPZ file."""
    df = pd.read_parquet(data_path)
    required = {TIME_COLUMN, ROBOT_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
    df = df.sort_values([TIME_COLUMN, ROBOT_COLUMN], kind="stable").reset_index(drop=True)

    times = np.sort(df[TIME_COLUMN].unique())
    if len(times) < 3:
        raise ValueError(
            f"Dataset is too small for a 70/15/15 split: {len(df)} rows."
        )
    train_cutoff = times[int(len(times) * TRAIN_FRACTION)]
    validation_cutoff = times[
        int(len(times) * (TRAIN_FRACTION + VALIDATION_FRACTION))
    ]
    if train_cutoff >= validation_cutoff or validation_cutoff >= times[-1]:
        raise ValueError(
            f"Dataset is too small for a 70/15/15 split: {len(df)} rows."
        )

    partitions = {
        "train": df.loc[df[TIME_COLUMN] < train_cutoff].reset_index(drop=True),
        "validation": df.loc[
            (df[TIME_COLUMN] >= train_cutoff)
            & (df[TIME_COLUMN] < validation_cutoff)
        ].reset_index(drop=True),
        "test": df.loc[df[TIME_COLUMN] >= validation_cutoff].reset_index(drop=True),
    }
    metadata = {
        "source": data_path.name,
        "time_column": TIME_COLUMN,
        "robot_column": ROBOT_COLUMN,
        "sort_order": [TIME_COLUMN, ROBOT_COLUMN],
        "train_fraction": TRAIN_FRACTION,
        "validation_fraction": VALIDATION_FRACTION,
        "test_fraction": TEST_FRACTION,
        "rows": {name: len(frame) for name, frame in partitions.items()},
        "test_start": partitions["test"][TIME_COLUMN].iloc[0].isoformat(),
        "test_end": partitions["test"][TIME_COLUMN].iloc[-1].isoformat(),
    }

    archive: dict[str, np.ndarray] = {
        "columns": np.asarray(df.columns, dtype=str),
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
    }
    for partition_name, frame in partitions.items():
        for column in df.columns:
            archive[f"{partition_name}__{column}"] = _to_numpy(frame[column])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **archive)
    return metadata


if __name__ == "__main__":
    result = create_temporal_split()
    print(json.dumps(result, indent=2, ensure_ascii=False))
