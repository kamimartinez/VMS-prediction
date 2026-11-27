"""Logistic-regression artifact loading and batched inference."""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from config import MODEL_PATH
from services.logistic_regression_preparation import (
    transform_logistic_features,
)

MODEL_FILE = MODEL_PATH / "logistic-regression" / "logistic_regresion.keras"
PREPROCESSOR_FILE = (
    MODEL_PATH / "logistic-regression"
    / "logistic_regresion_preprocessor.joblib"
)
THRESHOLD_FILE = (
    MODEL_PATH / "logistic-regression"
    / "logistic_regresion_threshold.json"
)


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"No se encontró el modelo logístico: {MODEL_FILE}")
    import tensorflow as tf

    return tf.keras.models.load_model(MODEL_FILE, compile=False)


@lru_cache(maxsize=1)
def load_preprocessor():
    if not PREPROCESSOR_FILE.is_file():
        raise FileNotFoundError(
            f"No se encontró el preprocesador logístico: {PREPROCESSOR_FILE}. "
            "Ejecuta la celda de exportación del notebook de regresión logística."
        )
    return joblib.load(PREPROCESSOR_FILE)


@lru_cache(maxsize=1)
def load_threshold() -> float:
    if not THRESHOLD_FILE.is_file():
        raise FileNotFoundError(f"No se encontró el umbral logístico: {THRESHOLD_FILE}")
    config = json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))
    threshold = config.get("decision_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(f"Umbral de decisión inválido en {THRESHOLD_FILE}")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"Umbral de decisión inválido en {THRESHOLD_FILE}")
    return float(threshold)


def predict(frame: pd.DataFrame, *, batch_size: int = 2048) -> pd.DataFrame:
    """Return class probabilities and decisions using the fitted training transformer."""
    if batch_size < 1:
        raise ValueError("batch_size debe ser mayor que cero.")
    model = load_model()
    preprocessor = load_preprocessor()
    threshold = load_threshold()
    matrix, metadata = transform_logistic_features(frame, preprocessor)
    expected_features = int(model.input_shape[-1])
    if matrix.shape[1] != expected_features:
        raise ValueError(
            f"El modelo logístico espera {expected_features} variables; "
            f"el preprocesador produjo {matrix.shape[1]}."
        )

    score_batches = []
    for start in range(0, len(matrix), batch_size):
        scores = model.predict(
            matrix[start:start + batch_size],
            batch_size=batch_size,
            verbose=0,
        ).reshape(-1)
        score_batches.append(scores)
    if not score_batches:
        raise ValueError(
            "No se generaron filas logísticas; se requiere historial contiguo "
            "para calcular las variables con rezago."
        )

    predictions = metadata.copy()
    predictions["score"] = np.concatenate(score_batches)
    predictions["predicted_probability"] = predictions["score"]
    predictions["predicted_queue"] = (
        predictions["score"] >= threshold
    ).astype("int8")
    predictions["decision_threshold"] = threshold
    return predictions.sort_values(
        ["bloque_tiempo", "ms"], kind="stable"
    ).reset_index(drop=True)
