"""Model-specific feature construction and fitted preprocessing for logistic regression."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "pct_posible_cola"
LABEL_THRESHOLD = 0.5
MACHINE_CATEGORIES = ["VMS 1", "VMS 2", "VMS 3"]
SEASON_CATEGORIES = ["invierno", "otoño", "primavera", "verano"]
DROP_COLUMNS = [
    TARGET, "bloque_tiempo", "cola", "produccion_promedio", "sin_etapa",
    "hora", "dia_semana", "vacas_activas_dia",
]


def build_logistic_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the logistic notebook's lag, cyclic, dummy, and column logic."""
    required = {
        "ms", "bloque_tiempo", TARGET, "hora", "dia_semana",
        "produccion_promedio", "sin_etapa", "vacas_activas_dia",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas para preparar Regresión logística: {sorted(missing)}"
        )
    ordered = frame.sort_values(
        ["ms", "bloque_tiempo"], kind="stable"
    ).reset_index(drop=True).copy()
    groups = ordered.groupby("ms", sort=False)
    contiguous = (
        ordered["bloque_tiempo"] - groups["bloque_tiempo"].shift(1)
    ) == pd.Timedelta(minutes=30)

    lag_columns = [
        column for column in (
            "total_ordenios", "duracion_total_min",
            "pct_ocupacion_robot", "hubo_ordenios",
        )
        if column in ordered.columns
    ]
    for column in lag_columns:
        ordered[f"{column}_anterior"] = groups[column].shift(1).where(contiguous)
    ordered["pct_cola_anterior"] = groups[TARGET].shift(1).where(contiguous)
    ordered["hora_sin"] = np.sin(2 * np.pi * ordered["hora"] / 24)
    ordered["hora_cos"] = np.cos(2 * np.pi * ordered["hora"] / 24)
    ordered["dia_sin"] = np.sin(2 * np.pi * ordered["dia_semana"] / 7)
    ordered["dia_cos"] = np.cos(2 * np.pi * ordered["dia_semana"] / 7)
    ordered["cola"] = (
        ordered[TARGET] > LABEL_THRESHOLD
    ).astype(np.float32)
    ordered = ordered.sort_values(
        ["bloque_tiempo", "ms"], kind="stable"
    ).reset_index(drop=True)

    clean = ordered.drop(columns=lag_columns).dropna(
        subset=["pct_cola_anterior"]
    ).copy().reset_index(drop=True)
    unknown_machines = set(clean["ms"].dropna().astype(str)) - set(MACHINE_CATEGORIES)
    unknown_seasons = (
        set(clean["temporada"].dropna().astype(str)) - set(SEASON_CATEGORIES)
    )
    if unknown_machines or unknown_seasons:
        raise ValueError(
            "Categorías fuera del conjunto de entrenamiento: "
            f"máquinas={sorted(unknown_machines)}, temporadas={sorted(unknown_seasons)}"
        )
    clean["ms"] = pd.Categorical(
        clean["ms"].astype(str), categories=MACHINE_CATEGORIES
    )
    clean["temporada"] = pd.Categorical(
        clean["temporada"].astype(str), categories=SEASON_CATEGORIES
    )
    clean = pd.get_dummies(
        clean, columns=["ms", "temporada"], dtype="float32", drop_first=True
    )

    missing_drop_columns = set(DROP_COLUMNS) - set(clean.columns)
    if missing_drop_columns:
        raise ValueError(
            f"Faltan columnas descartadas por el modelo: {sorted(missing_drop_columns)}"
        )
    metadata = clean[["bloque_tiempo", "ms_VMS 2", "ms_VMS 3"]].copy()
    metadata["ms"] = np.select(
        [metadata["ms_VMS 2"].eq(1), metadata["ms_VMS 3"].eq(1)],
        ["VMS 2", "VMS 3"],
        default="VMS 1",
    )
    features = clean.drop(columns=DROP_COLUMNS)
    return features, metadata[["ms", "bloque_tiempo"]]


def build_logistic_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create the same fitted transformer used by the training notebook."""
    dummies = [
        column for column in features.columns
        if column.startswith(("ms_", "temporada_"))
    ]
    categoricals = features.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    numerics = [
        column for column in features.select_dtypes(include="number").columns
        if column not in dummies and column != "cola"
    ]
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerics),
            ("bin", "passthrough", dummies),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categoricals,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def fit_logistic_preprocessor(frame: pd.DataFrame):
    """Fit on the chronological training portion for artifact export only."""
    features, _ = build_logistic_features(frame)
    train_count = int(len(features) * 0.70)
    if train_count == 0:
        raise ValueError("No hay suficientes filas para ajustar el preprocesador.")
    preprocessor = build_logistic_preprocessor(features)
    preprocessor.fit(features.iloc[:train_count])
    return preprocessor


def transform_logistic_features(
    frame: pd.DataFrame,
    preprocessor,
) -> tuple[np.ndarray, pd.DataFrame]:
    features, metadata = build_logistic_features(frame)
    expected_columns = list(preprocessor.feature_names_in_)
    features = features.reindex(columns=expected_columns)
    matrix = preprocessor.transform(features).to_numpy(dtype=np.float32)
    return matrix, metadata.reset_index(drop=True)
