"""GET /jobs and GET /jobs/{job_id} — ingestion job status."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.schemas import JobStatusResponse, JobListResponse
from app.ingestion.job_store import get_job, list_jobs

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Return the status of an ingestion job."""
    job = get_job(job_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content={"error": "JOB_NOT_FOUND", "detail": f"No job found with id {job_id}"},
        )
    return JobStatusResponse(**job)


@router.get("", response_model=JobListResponse)
async def list_all_jobs() -> JobListResponse:
    """Return all jobs ordered by created_at descending."""
    jobs = list_jobs()
    return JobListResponse(
        jobs=[JobStatusResponse(**j) for j in jobs],
        total=len(jobs),
    )
