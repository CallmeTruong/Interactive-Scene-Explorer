from backend.app.core.errors import not_found
from backend.app.db.repositories import Repository
from backend.app.schemas.job import JobResponse


class JobService:
    def __init__(self, *, repository: Repository) -> None:
        self._repository = repository

    def get_job(self, job_id: str) -> JobResponse:
        job = self._repository.get_job(job_id)
        if job is None:
            raise not_found(f"Job not found: {job_id}")

        return JobResponse(
            job_id=job.id,
            status=job.status,
            result=job.output_json,
            error=job.error,
        )
