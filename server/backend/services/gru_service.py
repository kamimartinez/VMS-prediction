"""GRU artifact loading and batched inference."""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from config import MODEL_PATH
from services.gru_preparation import iter_gru_batches

MODEL_FILE = MODEL_PATH / "gru" / "gru_E2_w48.keras"
SCALER_FILE = MODEL_PATH / "gru" / "gru_E2_w48_scaler.joblib"
CONFIG_FILE = MODEL_PATH / "gru" / "gru_E2_w48_config.json"


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"No se encontró el modelo GRU: {MODEL_FILE}")
    import tensorflow as tf

    return tf.keras.models.load_model(MODEL_FILE, compile=False)


@lru_cache(maxsize=1)
def load_scaler():
    if not SCALER_FILE.is_file():
        raise FileNotFoundError(f"No se encontró el scaler GRU: {SCALER_FILE}")
    return joblib.load(SCALER_FILE)


@lru_cache(maxsize=1)
def load_config() -> dict:
    if not CONFIG_FILE.is_file():
        raise FileNotFoundError(f"No se encontró la configuración GRU: {CONFIG_FILE}")
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    threshold = config.get("threshold_decision")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(f"Umbral de decisión inválido en {CONFIG_FILE}")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"Umbral de decisión inválido en {CONFIG_FILE}")
    return config


def predict(frame: pd.DataFrame, *, batch_size: int = 512) -> pd.DataFrame:
    """Return one regression score and queue decision per available 48-step window."""
    model = load_model()
    scaler = load_scaler()
    threshold = float(load_config()["threshold_decision"])
    expected_shape = tuple(model.input_shape[1:])
    if expected_shape != (48, 27):
        raise ValueError(
            f"Forma del modelo GRU no compatible: {expected_shape}; se esperaba (48, 27)."
        )

    ordered = frame.sort_values(
        ["bloque_tiempo", "ms"], kind="stable"
    ).reset_index(drop=True)
    row_positions = []
    score_batches = []
    for positions, windows in iter_gru_batches(
        ordered, scaler, batch_size=batch_size
    ):
        scores = model.predict(windows, batch_size=batch_size, verbose=0).reshape(-1)
        row_positions.append(positions)
        score_batches.append(scores)
    if not row_positions:
        raise ValueError(
            "La GRU necesita al menos 49 bloques continuos de 30 minutos por máquina."
        )

    positions = np.concatenate(row_positions)
    scores = np.concatenate(score_batches)
    predictions = ordered.iloc[positions][
        ["ms", "bloque_tiempo"]
    ].reset_index(drop=True)
    predictions["score"] = scores
    predictions["predicted_pct_queue"] = scores * 100
    predictions["predicted_queue"] = (scores >= threshold).astype("int8")
    predictions["decision_threshold"] = threshold
    return predictions.sort_values(
        ["bloque_tiempo", "ms"], kind="stable"
    ).reset_index(drop=True)
