from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

UPLOAD_DIR = SERVER_DIR / "storage" / "uploads"
RESULT_DIR = SERVER_DIR / "storage" / "results"
WORK_DIR = SERVER_DIR / "storage" / "work"
MODEL_PATH = PROJECT_ROOT / "models"

MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_FILES_PER_JOB = 500
ALLOWED_EXTENSIONS = {".xlsx", ".xls"}

CHUNK_SIZE = 1024 * 1024
READ_BATCH_SIZE = 8

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)