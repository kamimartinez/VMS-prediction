from pydantic import BaseModel
from datetime import datetime
from typing import Any, Literal, Optional

class FileJob(BaseModel):
  id: str
  filename: str
  file_size: int
  status: str = "pending"
  error: Optional[str] = None
  rows_processed: int = 0
  
class ProcessingJob(BaseModel):
  id: str
  model: Literal["gru", "logistic_regression"]
  status: str = "pending"
  progress: int = 0
  
  files: list[FileJob]
  
  created_at: datetime
  completed_at: Optional[datetime] = None
  
  error: Optional[str] = None