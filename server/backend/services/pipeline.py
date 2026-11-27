"""Batch preparation, model inference, and report persistence."""
from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from backend.config import RESULT_DIR, UPLOAD_DIR, WORK_DIR
from backend.services.gru_service import predict as predict_gru
from backend.services.job_store import get_job, update_file, update_job
from backend.services.logistic_regression_service import (
    predict as predict_logistic_regression,
)
from backend.services.preparation import prepare_uploaded_files

MODEL_PREDICTORS = {
    "gru": predict_gru,
    "logistic_regression": predict_logistic_regression,
}


def _report_summary(
    model: str,
    prepared: pd.DataFrame,
    predictions: pd.DataFrame,
    input_files: int,
) -> dict:
    merged = predictions.merge(
        prepared[["ms", "bloque_tiempo", "pct_posible_cola"]],
        on=["ms", "bloque_tiempo"],
        how="left",
        validate="one_to_one",
    )
    comparable = merged["pct_posible_cola"].notna()
    metrics = None
    if comparable.any():
        actual = (
            merged.loc[comparable, "pct_posible_cola"].to_numpy() > 0.5
        ).astype("int8")
        predicted = merged.loc[comparable, "predicted_queue"].to_numpy()
        metrics = {
            "accuracy": float(accuracy_score(actual, predicted)),
            "precision": float(precision_score(actual, predicted, zero_division=0)),
            "recall": float(recall_score(actual, predicted, zero_division=0)),
            "f1": float(f1_score(actual, predicted, zero_division=0)),
        }
    return {
        "model": model,
        "input_files": input_files,
        "feature_rows": int(len(prepared)),
        "prediction_rows": int(len(predictions)),
        "predicted_queue_rows": int(predictions["predicted_queue"].sum()),
        "decision_threshold": float(predictions["decision_threshold"].iloc[0]),
        "mean_score": float(predictions["score"].mean()),
        "comparable_rows": int(comparable.sum()),
        "metrics": metrics,
        "by_machine": [
            {
                "ms": str(machine),
                "prediction_rows": int(len(group)),
                "predicted_queue_rows": int(group["predicted_queue"].sum()),
                "mean_score": float(group["score"].mean()),
            }
            for machine, group in predictions.groupby("ms", sort=True)
        ],
    }


async def run_pipeline(job_id: str) -> None:
    job = get_job(job_id)
    if job is None:
        return
    predictor = MODEL_PREDICTORS.get(job.model)
    if predictor is None:
        update_job(job_id, status="failed", error=f"Modelo no soportado: {job.model}")
        return

    update_job(job_id, status="processing", progress=0, error=None)
    batch_dir = UPLOAD_DIR / job_id
    work_dir = WORK_DIR / job_id
    file_infos = []
    for file in job.files:
        matches = list(batch_dir.glob(f"{file.id}.*"))
        if not matches:
            update_file(
                job_id,
                file.id,
                status="failed",
                error="No se encontró el archivo subido.",
            )
            continue
        file_infos.append(
            {
                "id": file.id,
                "filename": file.filename,
                "path": str(matches[0]),
            }
        )

    if not file_infos:
        update_job(job_id, status="failed", error="No hay archivos disponibles.")
        return

    try:
        def update_read_progress(processed: int, total: int) -> None:
            update_job(
                job_id,
                progress=10 + int(45 * processed / total),
            )

        prepared = await asyncio.to_thread(
            prepare_uploaded_files,
            file_infos,
            work_dir,
            progress=update_read_progress,
        )
        update_job(job_id, progress=60)
        predictions = await asyncio.to_thread(predictor, prepared)
        if predictions.empty:
            raise ValueError("El modelo no generó predicciones para los archivos.")

        summary = _report_summary(
            job.model, prepared, predictions, len(file_infos)
        )
        result = {
            "job_id": job_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "predictions": json.loads(
                predictions.to_json(orient="records", date_format="iso")
            ),
        }
        result_path = RESULT_DIR / f"{job_id}.json"
        temporary_path = result_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(result_path)

        for file in job.files:
            if file.status != "failed":
                update_file(job_id, file.id, status="completed", error=None)
        completed_with_errors = any(file.status == "failed" for file in job.files)
        update_job(
            job_id,
            status="completed_with_errors" if completed_with_errors else "completed",
            progress=100,
            completed_at=datetime.now(timezone.utc),
        )
    except Exception as exc:
        for file in job.files:
            if file.status != "failed":
                update_file(job_id, file.id, status="failed", error=str(exc))
        update_job(job_id, status="failed", error=str(exc), progress=100)
        raise
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
