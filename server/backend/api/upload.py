import uuid
import shutil
from typing import Literal

from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    BackgroundTasks,
    Form,
    HTTPException
)

from backend.schemas.jobs import ProcessingJob, FileJob
from backend.services.storage import save_file
from backend.services.job_store import create_job, get_job
from backend.services.pipeline import run_pipeline
from backend.config import (
    MAX_FILES_PER_JOB,
    RESULT_DIR,
    UPLOAD_DIR,
)

import json

router = APIRouter()


@router.post("/upload", response_model=ProcessingJob)
async def upload_files(
    background_tasks: BackgroundTasks,
    model: Literal["gru", "logistic_regression"] = Form(...),
    files: list[UploadFile] = File(...)
):

    if not files:
        raise HTTPException(400, "No files provided")
    if len(files) > MAX_FILES_PER_JOB:
        raise HTTPException(
            413,
            f"At most {MAX_FILES_PER_JOB} Excel files can be uploaded per job.",
        )

    batch_id = str(uuid.uuid4())

    uploaded_files = []

    try:
        for file in files:

            info = await save_file(file, batch_id)

            uploaded_files.append(info)

    except Exception:
        shutil.rmtree(
            UPLOAD_DIR / batch_id,
            ignore_errors=True
        )
        raise

    # Create file records
    file_jobs = [
        FileJob(
            id=file["id"],
            filename=file["filename"],
            file_size=file["file_size"],
            status="pending"
        )
        for file in uploaded_files
    ]

    # Create batch job
    job = ProcessingJob(
        id=batch_id,
        model=model,
        status="pending",
        progress=0,
        files=file_jobs,
        created_at=datetime.now(timezone.utc)
    )

    create_job(job)

    # Start background processing
    background_tasks.add_task(
        run_pipeline,
        batch_id
    )

    return job


@router.get("/jobs/{job_id}", response_model=ProcessingJob)
async def get_job_status(job_id: str):

    job = get_job(job_id)

    if not job:
        raise HTTPException(404, "Job not found")

    return job


@router.get("/jobs/{job_id}/results")
async def get_job_results(job_id: str):

    job = get_job(job_id)

    if not job:
        raise HTTPException(404, "Job not found")

    result_path = RESULT_DIR / f"{job_id}.json"

    if not result_path.exists():
        raise HTTPException(
            404,
            "Results are not available yet"
        )

    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)