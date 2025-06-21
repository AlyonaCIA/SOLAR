# In a new file: app/api/pipeline/job_manager.py

import uuid
import threading
import time
from typing import Dict, Any, Optional
from enum import Enum

class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class JobManager:
    def __init__(self):
        self.jobs = {}
        # Clean up jobs older than 24 hours
        self._start_cleanup_thread()
    
    def create_job(self, params: Dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "id": job_id,
            "status": JobStatus.PENDING,
            "params": params,
            "result": None,
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        return job_id
    
    def start_job(self, job_id: str) -> None:
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        self.jobs[job_id]["status"] = JobStatus.PROCESSING
        self.jobs[job_id]["updated_at"] = time.time()
    
    def complete_job(self, job_id: str, result: Any) -> None:
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        self.jobs[job_id]["status"] = JobStatus.COMPLETED
        self.jobs[job_id]["result"] = result
        self.jobs[job_id]["updated_at"] = time.time()
    
    def fail_job(self, job_id: str, error: str) -> None:
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        self.jobs[job_id]["status"] = JobStatus.FAILED
        self.jobs[job_id]["error"] = error
        self.jobs[job_id]["updated_at"] = time.time()
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)
    
    def _start_cleanup_thread(self):
        def cleanup():
            while True:
                current_time = time.time()
                to_delete = []
                
                for job_id, job in self.jobs.items():
                    # Remove jobs older than 24 hours
                    if current_time - job["created_at"] > 86400:  # 24 hours in seconds
                        to_delete.append(job_id)
                
                for job_id in to_delete:
                    del self.jobs[job_id]
                
                # Run cleanup check every hour
                time.sleep(3600)
                
        thread = threading.Thread(target=cleanup, daemon=True)
        thread.start()

# Create a singleton instance
job_manager = JobManager()