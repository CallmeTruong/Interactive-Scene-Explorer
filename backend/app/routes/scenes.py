from fastapi import APIRouter, Depends

from backend.app.db.repositories import Repository, get_repository
from backend.app.schemas.job import JobResponse
from backend.app.schemas.scene import ClickRequest, SceneResponse
from backend.app.schemas.transition import ClickReadyResponse
from backend.app.services.scene_service import SceneService

router = APIRouter(tags=["scenes"])


def get_scene_service(
    repository: Repository = Depends(get_repository),
) -> SceneService:
    """Build the scene service with the current repository dependency."""
    return SceneService(repository=repository)


@router.get("/scenes/{scene_id}", response_model=SceneResponse)
def get_scene(
    scene_id: str,
    service: SceneService = Depends(get_scene_service),
) -> SceneResponse:
    """Return a scene image, metadata, and hotspot list."""
    return service.get_scene(scene_id)


@router.post("/scenes/{scene_id}/click", response_model=ClickReadyResponse)
def click_scene(
    scene_id: str,
    request: ClickRequest,
    service: SceneService = Depends(get_scene_service),
) -> ClickReadyResponse:
    """Handle a scene click and return the next scene transition package."""
    return service.handle_click(
        scene_id=scene_id,
        x=request.x,
        y=request.y,
        hotspot_id=request.hotspot_id,
    )


@router.post("/scenes/{scene_id}/prefetch", response_model=JobResponse)
def prefetch_scene(
    scene_id: str,
    request: ClickRequest,
    service: SceneService = Depends(get_scene_service),
) -> JobResponse:
    """Prepare and cache the next scene for a likely click."""
    return service.prefetch_click(
        scene_id=scene_id,
        x=request.x,
        y=request.y,
        hotspot_id=request.hotspot_id,
    )
