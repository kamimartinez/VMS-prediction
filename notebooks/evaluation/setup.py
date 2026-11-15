"""Utilidades para evaluar los modelos guardados sobre el test cronológico.

Uso en el notebook:

    from model_eval import load_test_data, prepare_models

    data = load_test_data()
    models, predictions = prepare_models(data)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "pct_posible_cola"
LABEL_THRESHOLD = 0.5
SEASON_CATEGORIES = ["invierno", "primavera", "verano", "otoño"]

MODEL_FILES = {
    "Regresión logística": "logistic_regresion.keras",
    "LSTM": "lstm_model.keras",
    "GRU": "gru_E2_w48.keras",
    "TCN": "tcn_classification_base_w48.keras",
}
GRU_SCALER_FILE = "gru_E2_w48_scaler.joblib"


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------
@dataclass
class TestData:
    """Contenedor con todo lo necesario para evaluar los modelos."""

    project_root: Path
    models_path: Path
    metadata: dict
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    full_data: pd.DataFrame  # train + validación + test, con columna __row_id__
    train_end: int
    validation_end: int
    test_rows: np.ndarray
    target: str = TARGET
    label_threshold: float = LABEL_THRESHOLD

    def actual_labels(self, rows) -> np.ndarray:
        """Etiqueta real (hay_cola) para las filas indicadas."""
        values = self.full_data.loc[rows, self.target].to_numpy()
        return (values > self.label_threshold).astype(np.int32)


def _find_project_root() -> Path:
    root = next(
        (path for path in (Path.cwd(), *Path.cwd().parents)
         if (path / "models").is_dir() and (path / "data").is_dir()),
        None,
    )
    if root is None:
        raise FileNotFoundError(
            "No se encontró la raíz del proyecto con las carpetas data/ y models/."
        )
    return root


def load_test_data(project_root=None, verbose: bool = True) -> TestData:
    """Carga data/test_data.npz y reconstruye las particiones train/val/test."""
    project_root = Path(project_root) if project_root else _find_project_root()
    split_path = project_root / "data" / "test_data.npz"
    models_path = project_root / "models"

    if not split_path.exists():
        raise FileNotFoundError(
            f"No existe {split_path}. "
            "Ejecuta notebooks/modeling/create_test_split.py primero."
        )

    with np.load(split_path, allow_pickle=False) as archive:
        columns = archive["columns"].tolist()
        metadata = json.loads(archive["metadata_json"].item())
        partitions = {
            name: pd.DataFrame({
                column: archive[f"{name}__{column}"] for column in columns
            })
            for name in ("train", "validation", "test")
        }

    train, validation, test = (
        partitions["train"], partitions["validation"], partitions["test"]
    )
    full_data = pd.concat([train, validation, test], ignore_index=True)
    full_data["bloque_tiempo"] = pd.to_datetime(full_data["bloque_tiempo"])
    full_data["__row_id__"] = np.arange(len(full_data), dtype=np.int64)

    train_end = len(train)
    validation_end = train_end + len(validation)
    test_rows = np.arange(validation_end, len(full_data), dtype=np.int64)

    assert len(test) == metadata["rows"]["test"]
    assert len(test_rows) == len(test)
    assert TARGET in full_data.columns

    data = TestData(
        project_root=project_root,
        models_path=models_path,
        metadata=metadata,
        train=train,
        validation=validation,
        test=test,
        full_data=full_data,
        train_end=train_end,
        validation_end=validation_end,
        test_rows=test_rows,
    )

    if verbose:
        print(f"Datos: {len(full_data):,} filas; train={len(train):,}, "
              f"validación={len(validation):,}, test={len(test):,}")
        print(f"Test cronológico: {test['bloque_tiempo'].min()} -> "
              f"{test['bloque_tiempo'].max()}")
        print("Modelos disponibles:",
              sorted(p.name for p in models_path.glob("*.keras")))
    return data


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------
def _load_model(models_path: Path, filename: str):
    path = models_path / filename
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el modelo requerido: {path}")
    return tf.keras.models.load_model(path, compile=False)


def _check_input_shape(model, expected_shape, model_name):
    actual_shape = tuple(model.input_shape[1:])
    if actual_shape != tuple(expected_shape):
        raise ValueError(
            f"{model_name}: forma de entrada esperada {actual_shape}, "
            f"pero la preparación produjo {tuple(expected_shape)}."
        )


def _predict_scores(model, features):
    return model.predict(features, batch_size=256, verbose=0).reshape(-1)


# ----- GRU y TCN ----------------------------------------------------------
def _build_sequence_values(data: TestData):
    full_data = data.full_data
    frame = full_data.copy()
    season_values = frame["temporada"].astype("string").str.lower().str.strip()
    unknown = set(season_values.dropna().unique()) - set(SEASON_CATEGORIES)
    if unknown:
        raise ValueError(f"Temporadas no reconocidas: {sorted(unknown)}")

    season_dummies = pd.get_dummies(
        pd.Categorical(season_values, categories=SEASON_CATEGORIES),
        prefix="temp",
    ).astype(np.float32)
    season_dummies.index = frame.index
    frame = pd.concat([frame.drop(columns="temporada"), season_dummies], axis=1)
    frame["hora_sin"] = np.sin(2 * np.pi * frame["hora"] / 24)
    frame["hora_cos"] = np.cos(2 * np.pi * frame["hora"] / 24)

    columns = [
        c for c in frame.columns if c not in {"ms", "bloque_tiempo", "__row_id__"}
    ]
    return columns, frame[columns].to_numpy(dtype=np.float32)


def _make_gru_scaler(models_path: Path, n_features: int):
    """Aplica la imputación a mano y usa el StandardScaler guardado.

    Evita el error de _fill_dtype por diferencias de versión de scikit-learn.
    """
    pipeline = joblib.load(models_path / GRU_SCALER_FILE)
    imputer = pipeline.named_steps["imputer"]
    std = pipeline.named_steps["scaler"]
    if std.n_features_in_ != n_features:
        raise ValueError("El scaler GRU no coincide con las variables secuenciales.")
    fill = np.nan_to_num(imputer.statistics_, nan=0.0)

    def scale(windows):
        flat = windows.reshape(-1, n_features)
        flat = np.where(np.isnan(flat), fill, flat)
        return std.transform(flat).reshape(windows.shape).astype(np.float32)

    return scale


def _make_tcn_scaler(sequence_values, train_end: int, n_features: int):
    """El TCN no guarda scaler: se ajusta solo con train."""
    scaler = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ]).fit(sequence_values[:train_end])

    def scale(windows):
        flat = windows.reshape(-1, n_features)
        return scaler.transform(flat).reshape(windows.shape).astype(np.float32)

    return scale


def _predict_gru_tcn(data: TestData, models: dict) -> dict:
    sequence_columns, sequence_values = _build_sequence_values(data)
    n_features = len(sequence_columns)

    window_size = int(models["GRU"].input_shape[1])
    for name in ("GRU", "TCN"):
        _check_input_shape(models[name], (window_size, n_features), name)
    if int(models["TCN"].input_shape[1]) != window_size:
        raise ValueError(
            "GRU y TCN deben usar el mismo tamaño de ventana para esta evaluación."
        )

    scalers = {
        "GRU": _make_gru_scaler(data.models_path, n_features),
        "TCN": _make_tcn_scaler(sequence_values, data.train_end, n_features),
    }

    ordered = data.full_data.sort_values(["ms", "bloque_tiempo"], kind="stable")
    step = np.timedelta64(30, "m")

    batches = {name: [] for name in scalers}
    row_id_batches = []

    for _, robot_rows in ordered.groupby("ms", sort=False):
        row_ids = robot_rows["__row_id__"].to_numpy(dtype=np.int64)
        times = robot_rows["bloque_tiempo"].to_numpy(dtype="datetime64[ns]")
        if len(row_ids) <= window_size:
            continue

        windows = np.lib.stride_tricks.sliding_window_view(
            sequence_values[row_ids], window_size, axis=0
        ).transpose(0, 2, 1)
        starts = np.arange(len(row_ids) - window_size)
        windows = windows[: len(starts)]  # la última ventana no tiene target
        targets = starts + window_size

        gaps = np.r_[0, (np.diff(times) != step).astype(np.int64)]
        gap_prefix = np.cumsum(gaps)
        continuous = (gap_prefix[targets] - gap_prefix[starts]) == 0
        selected = continuous & (row_ids[targets] >= data.validation_end)
        if not selected.any():
            continue

        selected_windows = windows[selected]
        row_id_batches.append(row_ids[targets[selected]])
        for name, scale in scalers.items():
            scaled = scale(selected_windows)
            batches[name].append(_predict_scores(models[name], scaled))
            del scaled

    if not row_id_batches:
        raise ValueError("No se generaron ventanas de test para GRU/TCN.")

    all_ids = np.concatenate(row_id_batches)
    order = np.argsort(all_ids)
    return {
        name: pd.Series(np.concatenate(scores)[order], index=all_ids[order])
        for name, scores in batches.items()
    }, window_size, n_features


# ----- LSTM ---------------------------------------------------------------
def _predict_lstm(data: TestData, model):
    frame = data.full_data.drop(columns="__row_id__").copy()
    frame["ms"] = frame["ms"].astype(str)
    frame["temporada"] = frame["temporada"].astype(str)
    frame = pd.get_dummies(frame, columns=["ms", "temporada"], dtype="float32")
    frame["cola"] = (frame[data.target] > data.label_threshold).astype(np.float32)
    features = frame.drop(columns="bloque_tiempo")

    dummies = [c for c in features.columns if c.startswith(("ms_", "temporada_"))]
    categoricals = features.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    numerics = [
        c for c in features.select_dtypes(include="number").columns
        if c not in dummies and c != "cola"
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numerics),
            ("bin", "passthrough", dummies + ["cola"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categoricals),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.fit(features.iloc[:data.train_end])
    matrix = preprocessor.transform(features).astype(np.float32)

    window = int(model.input_shape[1])
    _check_input_shape(model, (window, matrix.shape[1]), "LSTM")
    targets = np.arange(max(data.validation_end, window), len(data.full_data))
    indices = targets[:, None] - window + np.arange(window)
    scores = _predict_scores(model, matrix[indices])
    return pd.Series(scores, index=targets), (window, matrix.shape[1])


# ----- Regresión logística --------------------------------------------------
def _predict_logistic(data: TestData, model):
    target = data.target
    frame = data.full_data.sort_values(
        ["ms", "bloque_tiempo"], kind="stable"
    ).reset_index(drop=True)
    groups = frame.groupby("ms", sort=False)
    continuous_rows = (
        frame["bloque_tiempo"] - groups["bloque_tiempo"].shift(1)
    ) == pd.Timedelta("30min")

    lag_columns = [
        c for c in ["total_ordenios", "duracion_total_min",
                    "pct_ocupacion_robot", "hubo_ordenios"]
        if c in frame.columns
    ]
    for column in lag_columns:
        frame[f"{column}_anterior"] = groups[column].shift(1).where(continuous_rows)
    frame["pct_cola_anterior"] = groups[target].shift(1).where(continuous_rows)
    frame["hora_sin"] = np.sin(2 * np.pi * frame["hora"] / 24)
    frame["hora_cos"] = np.cos(2 * np.pi * frame["hora"] / 24)
    frame["dia_sin"] = np.sin(2 * np.pi * frame["dia_semana"] / 7)
    frame["dia_cos"] = np.cos(2 * np.pi * frame["dia_semana"] / 7)
    frame["cola"] = (frame[target] > data.label_threshold).astype(np.float32)

    frame = frame.sort_values(["bloque_tiempo", "ms"], kind="stable").reset_index(drop=True)
    clean = (
        frame.drop(columns=lag_columns)
        .dropna(subset=["pct_cola_anterior"])
        .copy()
        .reset_index(drop=True)
    )
    clean["ms"] = clean["ms"].astype(str)
    clean["temporada"] = clean["temporada"].astype(str)
    clean = pd.get_dummies(
        clean, columns=["ms", "temporada"], dtype="float32", drop_first=True
    )

    drop_columns = [
        target, "bloque_tiempo", "cola", "produccion_promedio", "sin_etapa",
        "hora", "dia_semana", "vacas_activas_dia", "__row_id__",
    ]
    row_ids = clean["__row_id__"].to_numpy(dtype=np.int64)
    features = clean.drop(columns=drop_columns)

    dummies = [c for c in features.columns if c.startswith(("ms_", "temporada_"))]
    categoricals = features.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    numerics = [
        c for c in features.select_dtypes(include="number").columns
        if c not in dummies and c != "cola"
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", StandardScaler()),
            ]), numerics),
            ("bin", "passthrough", dummies),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categoricals),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    train_count = int(len(features) * 0.70)
    preprocessor.fit(features.iloc[:train_count])
    matrix = preprocessor.transform(features).astype(np.float32)

    _check_input_shape(model, (matrix.shape[1],), "Regresión logística")
    is_test = row_ids >= data.validation_end
    scores = _predict_scores(model, matrix[is_test])
    return pd.Series(scores, index=row_ids[is_test]), matrix.shape[1]


# --------------------------------------------------------------------------
# API pública
# --------------------------------------------------------------------------
def prepare_models(data: TestData, verbose: bool = True):
    """Carga los 4 modelos y genera sus predicciones sobre test.

    Devuelve:
        models: dict nombre -> modelo Keras
        predictions: dict nombre -> pd.Series (índice = __row_id__ de la fila objetivo)
    """
    models = {
        name: _load_model(data.models_path, filename)
        for name, filename in MODEL_FILES.items()
    }

    predictions = {}
    seq_preds, window_size, n_seq_features = _predict_gru_tcn(data, models)
    predictions.update(seq_preds)

    predictions["LSTM"], lstm_shape = _predict_lstm(data, models["LSTM"])
    predictions["Regresión logística"], n_logistic = _predict_logistic(
        data, models["Regresión logística"]
    )

    if verbose:
        print("Variables de entrada:", {
            "Regresión logística": n_logistic,
            "LSTM": lstm_shape,
            "GRU/TCN": (window_size, n_seq_features),
        })
        print("Filas de test con predicción por modelo:", {
            name: len(values) for name, values in predictions.items()
        })
    return models, predictions


def common_test_rows(data: TestData, predictions: dict) -> np.ndarray:
    """Filas de test para las que los cuatro modelos tienen predicción (opcional)."""
    common = set(data.test_rows.tolist())
    for values in predictions.values():
        common.intersection_update(values.index.tolist())
    if not common:
        raise ValueError("No hay filas de test comunes para comparar los modelos.")
    return np.asarray(sorted(common), dtype=np.int64)
