from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from api.upload import router as upload_router

app = FastAPI(
    title="ML Processing API",
    description="Multi-file upload and ML inference pipeline",
    version="1.0.0"
)

app.include_router(
    upload_router,
    prefix="/api",
    tags=["Processing"]
)


@app.get("/health")
async def health():
    return {"status": "healthy"}


app.mount(
    "/",
    StaticFiles(
        directory=Path(__file__).resolve().parents[1] / "frontend",
        html=True,
    ),
    name="frontend",
)