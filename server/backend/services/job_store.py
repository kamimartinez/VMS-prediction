from backend.schemas.jobs import ProcessingJob

jobs: dict[str, ProcessingJob] = {}

def create_job(job: ProcessingJob):
  jobs[job.id] = job
  
def get_job(job_id: str):
  return jobs.get(job_id)

def update_job(job_id: str, **kwargs): 
  job = jobs.get(job_id)
  
  if job:
    for key, value in kwargs.items():
      setattr(job, key, value)
      
def update_file(job_id: str, file_id: str, **kwargs):
  job = jobs.get(job_id)
  
  if not job:
    return
  
  for file in job.files:
    if file.id == file_id:
      for key, value in kwargs.items():
        setattr(file, key, value)
      break