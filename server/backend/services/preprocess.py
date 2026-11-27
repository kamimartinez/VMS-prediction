import pandas as pd
from pathlib import Path

def preprocess_file(file_path: str) -> pd.DataFrame:
  
  extension = Path(file_path).suffix.lower()
  
  if extension == ".csv":
    df = pd.read_csv(file_path)
  
  elif extension in {".xlsx", ".xls"}:
    df = pd.read_excel(file_path)
    
  else:
    raise ValueError(
      f"Unsuported file format: {extension}"
    )
    
  df = df.dropna(how="all")
  
  if df.empty:
    raise ValueError("File contains no usable data")
  
  # Preprocess logic
  
  return df