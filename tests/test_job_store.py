"""Tests for the in-memory job store (Phase 4).

No mocking needed — pure in-memory dict operations.
Imports are inline so earlier-phase tests are not blocked.
"""

import pytest


def test_create_job_returns_record():
    from app.ingestion.job_store import create_job

    job = create_job("job-1", "test.pdf")
    assert job["job_id"] == "job-1"
    assert job["filename"] == "test.pdf"
    assert job["status"] == "pending"
    assert job["document_id"] is None
    assert job["error"] is None
    assert "created_at" in job
    assert "updated_at" in job


def test_get_job_returns_created_job():
    from app.ingestion.job_store import create_job, get_job

    create_job("job-get-1", "a.pdf")
    job = get_job("job-get-1")
    assert job is not None
    assert job["job_id"] == "job-get-1"


def test_get_job_returns_none_for_unknown():
    from app.ingestion.job_store import get_job

    assert get_job("nonexistent-job-id") is None


def test_update_job_changes_status():
    from app.ingestion.job_store import create_job, update_job, get_job

    create_job("job-upd-1", "b.pdf")
    update_job("job-upd-1", status="running")
    job = get_job("job-upd-1")
    assert job["status"] == "running"


def test_update_job_sets_completion_fields():
    from app.ingestion.job_store import create_job, update_job, get_job

    create_job("job-upd-2", "c.pdf")
    update_job("job-upd-2", status="completed", document_id="doc-99",
               pages=10, chunks=50, replaced=False)
    job = get_job("job-upd-2")
    assert job["status"] == "completed"
    assert job["document_id"] == "doc-99"
    assert job["pages"] == 10
    assert job["chunks"] == 50
    assert job["replaced"] is False


def test_update_job_sets_error_on_failure():
    from app.ingestion.job_store import create_job, update_job, get_job

    create_job("job-fail-1", "bad.pdf")
    update_job("job-fail-1", status="failed", error="Invalid PDF")
    job = get_job("job-fail-1")
    assert job["status"] == "failed"
    assert job["error"] == "Invalid PDF"


def test_list_jobs_returns_all():
    from app.ingestion.job_store import create_job, list_jobs

    create_job("job-list-a", "a.pdf")
    create_job("job-list-b", "b.pdf")
    jobs = list_jobs()
    ids = {j["job_id"] for j in jobs}
    assert "job-list-a" in ids
    assert "job-list-b" in ids


def test_list_jobs_ordered_by_created_at_descending():
    from app.ingestion.job_store import create_job, list_jobs
    import time

    create_job("job-ord-1", "first.pdf")
    time.sleep(0.01)
    create_job("job-ord-2", "second.pdf")
    jobs = list_jobs()
    # Find the two jobs we created
    our_jobs = [j for j in jobs if j["job_id"] in ("job-ord-1", "job-ord-2")]
    assert our_jobs[0]["job_id"] == "job-ord-2"  # most recent first
