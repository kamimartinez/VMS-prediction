import aiofiles
import uuid
from pathlib import Path

from fastapi import UploadFile, HTTPException

from backend.config import (
  UPLOAD_DIR,
  MAX_FILE_SIZE,
  CHUNK_SIZE,
  ALLOWED_EXTENSIONS
)

async def save_file(file: UploadFile, batch_id: str):
  if not file.filename:
    raise HTTPException(400, "Filename is required")
  
  extension = Path(file.filename).suffix.lower()
  
  if extension not in ALLOWED_EXTENSIONS:
    raise HTTPException(400, f"Unsupported file format: {extension}")
  
  file_id = str(uuid.uuid4())
  
  batch_dir = UPLOAD_DIR / batch_id
  batch_dir.mkdir(parents=True, exist_ok=True)
  
  file_path = batch_dir / f"{file_id}{extension}"
  file_size = 0
  
  try: 
    async with aiofiles.open(file_path, "wb") as output:
      while chunk := await file.read(CHUNK_SIZE):
        file_size += len(chunk)
        
        if file_size > MAX_FILE_SIZE:
          raise HTTPException(413, "File exceeds maximum allowed size")
        
        await output.write(chunk)
    
  except Exception:
    file_path.unlink(missing_ok=True)
    raise
  
  finally:
    await file.close()
    
  return {
    "id": file_id,
    "filename": file.filename,
    "file_size": file_size,
    "path": str(file_path)
  }