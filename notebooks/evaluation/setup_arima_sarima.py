"""Reproduce the selected ARIMA/SARIMA models from ARIMA-SARIMA.ipynb."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.kalman_filter import (
    MEMORY_NO_FILTERED,
    MEMORY_NO_GAIN,
    MEMORY_NO_PREDICTED_COV,
    MEMORY_NO_SMOOTHING,
)
from threadpoolctl import threadpool_limits

TARGET = "pct_posible_cola"
LABEL_THRESHOLD = 0.5
FREQUENCY = "30min"
MEMORY = (
    MEMORY_NO_PREDICTED_COV
    | MEMORY_NO_FILTERED
    | MEMORY_NO_GAIN
    | MEMORY_NO_SMOOTHING
)

SELECTED_MODELS = {
    ("ARIMA", "VMS 1"): {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0), "threshold": 0.35},
    ("SARIMA", "VMS 1"): {"order": (1, 0, 0), "seasonal_order": (0, 0, 1, 48), "threshold": 0.20},
    ("ARIMA", "VMS 2"): {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0), "threshold": 0.38},
    ("SARIMA", "VMS 2"): {"order": (1, 0, 0), "seasonal_order": (1, 0, 0, 48), "threshold": 0.15},
    ("ARIMA", "VMS 3"): {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 0), "threshold": 0.37},
    ("SARIMA", "VMS 3"): {"order": (1, 0, 0), "seasonal_order": (1, 0, 0, 48), "threshold": 0.18},
}


def _converged(result) -> bool:
    details = result.fit_details
    if "arma_results" in details:
        return bool(
            details.converged and details.arma_results[-1].minimize_results.success
        )
    return bool(details.minimize_results.success)


def _fit_model(values: pd.Series, config: dict):
    with threadpool_limits(limits=1):
        for method, options in [
            ("L-BFGS-B", {"maxiter": 200, "ftol": 1e-09}),
            ("BFGS", {"maxiter": 250, "gtol": 0.0001}),
        ]:
            model = ARIMA(
                values,
                order=config["order"],
                seasonal_order=config["seasonal_order"],
                trend="c" if config["order"][1] == 0 else "n",
                enforce_stationarity=True,
                enforce_invertibility=True,
            )
            model.ssm.set_conserve_memory(MEMORY)
            with warnings.catch_warnings(record=True):
                result = model.fit(
                    method="innovations_mle",
                    cov_type="none",
                    method_kwargs={
                        "minimize_kwargs": {
                            "method": method,
                            "options": options,
                        }
                    },
                )
            if _converged(result) and np.isfinite(result.params).all():
                return result
    raise RuntimeError(
        f"No convergió el ajuste final {config['order']} "
        f"{config['seasonal_order']}."
    )


def _continue_filter(result, values: pd.Series):
    values = values.asfreq(FREQUENCY)
    filter_results = result.filter_results
    model = result.model.clone(values)
    model.ssm.set_conserve_memory(MEMORY)
    model.initialize_known(
        filter_results.predicted_state[:, -1],
        filter_results.predicted_state_cov[:, :, -1],
    )
    return model.filter(result.params, cov_type="none")


def _metrics(frame: pd.DataFrame) -> dict:
    actual = frame["Cola_real"].to_numpy(dtype=np.int32)
    predicted = frame["Cola_predicha"].to_numpy(dtype=np.int32)
    output = {
        "N": len(frame),
        "Precision": precision_score(actual, predicted, zero_division=0),
        "Recall": recall_score(actual, predicted, zero_division=0),
        "F1": f1_score(actual, predicted, zero_division=0),
    }
    if frame["Prediccion"].notna().all():
        error = frame["Real"].to_numpy() - frame["Prediccion"].to_numpy()
        output.update(
            MAE=mean_absolute_error(frame["Real"], frame["Prediccion"]),
            RMSE=float(np.sqrt(np.mean(error**2))),
            R2=r2_score(frame["Real"], frame["Prediccion"]),
        )
    else:
        output.update(MAE=np.nan, RMSE=np.nan, R2=np.nan)
    return output


def evaluate_arima_sarima(data):
    """Return ARIMA/SARIMA test predictions and the notebook's global metrics.

    The input is the ``TestData`` returned by ``setup.load_test_data``. This
    uses the same chronological 70/15/15 split, per-robot scaling, fixed
    validation-selected orders/thresholds, and one-step test filtering as the
    ARIMA-SARIMA modeling notebook. Predictions and metrics are restricted to
    the exact row IDs in ``data.test_rows``.
    """
    source_path = data.project_root / "data" / "dataset_final_serie_tiempo.parquet"
    if not source_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset de ARIMA/SARIMA: {source_path}")
    full_data = pd.read_parquet(source_path)
    if TARGET not in full_data or "ms" not in full_data or "bloque_tiempo" not in full_data:
        raise ValueError("Faltan columnas requeridas en los datos de evaluación.")

    full_data["bloque_tiempo"] = pd.to_datetime(full_data["bloque_tiempo"])
    full_data = full_data.sort_values(
        ["bloque_tiempo", "ms"], kind="stable"
    ).reset_index(drop=True)
    dates = pd.DatetimeIndex(full_data["bloque_tiempo"].unique()).sort_values()
    validation_start = dates[int(len(dates) * 0.7)]
    test_start = dates[int(len(dates) * 0.85)]
    test_dates = dates[dates >= test_start]
    robots = sorted(full_data["ms"].unique())
    row_ids = data.full_data.set_index(
        ["ms", "bloque_tiempo"]
    )["__row_id__"]
    prediction_parts = []
    baseline_parts = []

    for robot in robots:
        robot_rows = (
            full_data.loc[full_data["ms"] == robot]
            .sort_values("bloque_tiempo")
            .set_index("bloque_tiempo")
        )
        values = robot_rows[TARGET].astype(float).asfreq(FREQUENCY)
        ids = row_ids.reindex(
            pd.MultiIndex.from_arrays(
                [[robot] * len(test_dates), test_dates],
                names=["ms", "bloque_tiempo"],
            )
        ).to_numpy(dtype=np.int64)
        test_mask = np.isin(ids, data.test_rows)
        robot_test_dates = test_dates[test_mask]
        robot_test_ids = ids[test_mask]
        if not len(robot_test_dates):
            raise ValueError(f"No hay filas de test para {robot}.")

        scaler = StandardScaler().fit(
            values.loc[values.index < validation_start].to_numpy().reshape(-1, 1)
        )
        scaled = pd.Series(
            scaler.transform(values.to_numpy().reshape(-1, 1)).ravel(),
            index=values.index,
            name=TARGET,
        ).asfreq(FREQUENCY)
        final_train = scaled.loc[scaled.index < test_start]

        for family in ("ARIMA", "SARIMA"):
            config = SELECTED_MODELS[(family, str(robot))]
            fitted = _fit_model(final_train, config)
            filtered = _continue_filter(fitted, scaled.loc[test_dates])
            raw_prediction = scaler.inverse_transform(
                np.asarray(filtered.fittedvalues).reshape(-1, 1)
            ).ravel()
            clipped_prediction = np.clip(raw_prediction, 0, 1)[test_mask]
            actual = values.loc[test_dates].to_numpy()[test_mask]
            part = pd.DataFrame(
                {
                    "__row_id__": robot_test_ids,
                    "bloque_tiempo": robot_test_dates,
                    "ms": robot,
                    "Modelo": family,
                    "Real": actual,
                    "Prediccion": clipped_prediction,
                    "Umbral": config["threshold"],
                    "Cola_real": (actual > LABEL_THRESHOLD).astype(np.int32),
                    "Cola_predicha": (
                        clipped_prediction >= config["threshold"]
                    ).astype(np.int32),
                }
            )
            prediction_parts.append(part)

        actual = values.loc[test_dates].to_numpy()
        actual_for_test = actual[test_mask]
        for name, lag in (("Persistencia", 1), ("Mismo bloque ayer", 48)):
            predicted_value = values.shift(lag).loc[test_dates].to_numpy()
            predicted_value = predicted_value[test_mask]
            baseline_parts.append(
                pd.DataFrame(
                    {
                        "Modelo": name,
                        "Real": actual_for_test,
                        "Prediccion": np.clip(predicted_value, 0, 1),
                        "Cola_real": (
                            actual_for_test > LABEL_THRESHOLD
                        ).astype(np.int32),
                        "Cola_predicha": (
                            predicted_value > LABEL_THRESHOLD
                        ).astype(np.int32),
                    }
                )
            )
        baseline_parts.append(pd.DataFrame({
            "Modelo": "Siempre cola",
            "Real": actual_for_test,
            "Prediccion": np.nan,
            "Cola_real": (actual_for_test > LABEL_THRESHOLD).astype(np.int32),
            "Cola_predicha": np.ones(len(actual_for_test), dtype=np.int32),
        }))

    predictions = pd.concat(prediction_parts, ignore_index=True)
    if set(predictions["__row_id__"].unique()) != set(data.test_rows.tolist()):
        raise ValueError("Las predicciones ARIMA/SARIMA no coinciden con las filas de test.")
    all_predictions = pd.concat(
        [
            predictions[["Modelo", "Real", "Prediccion", "Cola_real", "Cola_predicha"]],
            *baseline_parts,
        ],
        ignore_index=True,
    )
    metrics = pd.DataFrame(
        [
            {"Modelo": name, **_metrics(group)}
            for name, group in all_predictions.groupby("Modelo", sort=False)
        ]
    ).set_index("Modelo")
    return predictions, metrics
