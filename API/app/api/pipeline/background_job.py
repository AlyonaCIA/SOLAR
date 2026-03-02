# In app/api/pipeline/background_job.py

import threading

from app.api.pipeline.job_manager import job_manager


def run_in_background(func, job_id, *args, **kwargs):
    """Run a function in a background thread with job status tracking"""

    def wrapper():
        try:
            job_manager.start_job(job_id)
            result = func(*args, **kwargs)
            job_manager.complete_job(job_id, result)
        except Exception as e:
            import traceback

            error = f"{str(e)}\n{traceback.format_exc()}"
            job_manager.fail_job(job_id, error)

    thread = threading.Thread(target=wrapper)
    thread.daemon = True
    thread.start()
    return job_id
