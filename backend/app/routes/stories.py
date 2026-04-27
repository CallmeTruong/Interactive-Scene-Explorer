from fastapi import APIRouter, Depends

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
    service: StoryService = Depends(get_story_service),
) -> StoryCreateResponse:
    return service.create_story(prompt=request.prompt, style=request.style)
