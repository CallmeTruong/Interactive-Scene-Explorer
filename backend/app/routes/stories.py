from fastapi import APIRouter, BackgroundTasks, Depends

from backend.app.core.config import settings
from backend.app.db.repositories import Repository, get_repository
from backend.app.schemas.story import StoryCreateRequest, StoryCreateResponse
from backend.app.services.story_service import StoryService

router = APIRouter(tags=["stories"])


def get_story_service(
    repository: Repository = Depends(get_repository),
) -> StoryService:
    return StoryService(repository=repository)


@router.post("/stories", response_model=StoryCreateResponse)
def create_story(
    request: StoryCreateRequest,
    background_tasks: BackgroundTasks,
    service: StoryService = Depends(get_story_service),
) -> StoryCreateResponse:
    if settings.async_jobs_enabled and settings.image_generator_backend != "mock":
        response = service.start_create_story_job(prompt=request.prompt, style=request.style)
        background_tasks.add_task(
            service.complete_create_story_job,
            job_id=response.job_id,
            story_id=response.story_id,
            prompt=request.prompt,
            style=request.style,
        )
        return response

    return service.create_story(prompt=request.prompt, style=request.style)
