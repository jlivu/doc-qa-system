"""In-memory job store for async ingestion tracking."""

import threading
from datetime import datetime, timezone

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def create_job(job_id: str, filename: str) -> dict:
    """Create a new pending job."""
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        job = {
            "job_id": job_id,
            "status": "pending",
            "filename": filename,
            "created_at": now,
            "updated_at": now,
            "document_id": None,
            "pages": None,
            "chunks": None,
            "replaced": None,
            "error": None,
        }
        _jobs[job_id] = job
        return dict(job)


def update_job(job_id: str, **kwargs) -> None:
    """Update fields on an existing job."""
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)
            _jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()


def get_job(job_id: str) -> dict | None:
    """Return a copy of the job record, or None if not found."""
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def list_jobs() -> list[dict]:
    """Return all jobs ordered by created_at descending."""
    with _lock:
        return sorted(
            [dict(j) for j in _jobs.values()],
            key=lambda j: j["created_at"],
            reverse=True,
        )
