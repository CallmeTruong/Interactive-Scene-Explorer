from fastapi import APIRouter, Depends

from backend.app.db.repositories import Repository, get_repository
from backend.app.schemas.job import JobResponse
from backend.app.services.job_service import JobService

router = APIRouter(tags=["jobs"])


def get_job_service(
    repository: Repository = Depends(get_repository),
) -> JobService:
    return JobService(repository=repository)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(
    job_id: str,
    service: JobService = Depends(get_job_service),
) -> JobResponse:
    return service.get_job(job_id)
