from fastapi import APIRouter, Depends

from backend.app.db.repositories import Repository, get_repository
from backend.app.schemas.demo import DemoResetResponse
from backend.app.services.demo_service import DemoService

router = APIRouter(tags=["demo"])


def get_demo_service(
    repository: Repository = Depends(get_repository),
) -> DemoService:
    return DemoService(repository=repository)


@router.post("/demo/reset", response_model=DemoResetResponse)
def reset_demo(
    service: DemoService = Depends(get_demo_service),
) -> DemoResetResponse:
    return service.reset()
