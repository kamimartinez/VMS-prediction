"""Model-specific feature construction for the production GRU."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from backend.services.preparation import SEASON_CATEGORIES, TIME_STEP

WINDOW_SIZE = 48
EXPECTED_FEATURES = 27


def _sequence_values(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    required = {
        "ms", "bloque_tiempo", "temporada", "hora", "pct_posible_cola",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas para preparar la GRU: {sorted(missing)}")

    ordered = frame.sort_values(
        ["bloque_tiempo", "ms"], kind="stable"
    ).reset_index(drop=True).copy()
    season_values = ordered["temporada"].astype("string").str.lower().str.strip()
    unknown = set(season_values.dropna().unique()) - set(SEASON_CATEGORIES)
    if unknown:
        raise ValueError(f"Temporadas no reconocidas para la GRU: {sorted(unknown)}")
    season_dummies = pd.get_dummies(
        pd.Categorical(season_values, categories=SEASON_CATEGORIES),
        prefix="temp",
    ).astype(np.float32)
    season_dummies.index = ordered.index
    ordered = pd.concat(
        [ordered.drop(columns="temporada"), season_dummies], axis=1
    )
    ordered["hora_sin"] = np.sin(2 * np.pi * ordered["hora"] / 24)
    ordered["hora_cos"] = np.cos(2 * np.pi * ordered["hora"] / 24)
    feature_columns = [
        column for column in ordered.columns
        if column not in {"ms", "bloque_tiempo", "__row_id__"}
    ]
    values = ordered[feature_columns].to_numpy(dtype=np.float32)
    if values.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"La GRU requiere {EXPECTED_FEATURES} variables; "
            f"la preparación produjo {values.shape[1]}."
        )
    ordered["__prediction_row__"] = np.arange(len(ordered), dtype=np.int64)
    return ordered, values


def _scale_windows(windows: np.ndarray, scaler) -> np.ndarray:
    try:
        imputer = scaler.named_steps["imputer"]
        standard_scaler = scaler.named_steps["scaler"]
    except (AttributeError, KeyError) as exc:
        raise ValueError("El artefacto de escalado GRU no tiene imputer/scaler.") from exc
    if standard_scaler.n_features_in_ != windows.shape[-1]:
        raise ValueError(
            "El scaler GRU no coincide con la cantidad de variables preparadas."
        )
    fill_values = np.nan_to_num(imputer.statistics_, nan=0.0)
    flat = windows.reshape(-1, windows.shape[-1])
    flat = np.where(np.isnan(flat), fill_values, flat)
    return standard_scaler.transform(flat).reshape(windows.shape).astype(np.float32)


def iter_gru_batches(
    frame: pd.DataFrame,
    scaler,
    *,
    batch_size: int = 512,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield global row positions and scaled (48, 27) windows in bounded batches."""
    if batch_size < 1:
        raise ValueError("batch_size debe ser mayor que cero.")
    ordered, sequence_values = _sequence_values(frame)
    times = pd.to_datetime(ordered["bloque_tiempo"]).to_numpy(
        dtype="datetime64[ns]"
    )
    for _, group in ordered.groupby("ms", sort=False):
        positions = group["__prediction_row__"].to_numpy(dtype=np.int64)
        if len(positions) <= WINDOW_SIZE:
            continue
        machine_times = times[positions]
        windows = np.lib.stride_tricks.sliding_window_view(
            sequence_values[positions], WINDOW_SIZE, axis=0
        ).transpose(0, 2, 1)
        targets = np.arange(WINDOW_SIZE, len(positions))

        gaps = np.r_[
            0,
            (np.diff(machine_times) != TIME_STEP.to_timedelta64()).astype(np.int64),
        ]
        gap_prefix = np.cumsum(gaps)
        starts = targets - WINDOW_SIZE
        continuous = (gap_prefix[targets] - gap_prefix[starts]) == 0
        valid_targets = targets[continuous]
        if not len(valid_targets):
            continue

        for start in range(0, len(valid_targets), batch_size):
            batch_targets = valid_targets[start:start + batch_size]
            batch_windows = windows[batch_targets - WINDOW_SIZE]
            yield (
                positions[batch_targets],
                _scale_windows(batch_windows, scaler),
            )
