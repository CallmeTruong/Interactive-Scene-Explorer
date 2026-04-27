from fastapi import APIRouter, BackgroundTasks, Depends

from backend.app.core.config import settings
from backend.app.db.repositories import Repository, get_repository
from backend.app.schemas.job import JobResponse
from backend.app.schemas.scene import ClickRequest, SceneResponse
from backend.app.schemas.transition import ClickProcessingResponse, ClickReadyResponse
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


@router.post("/scenes/{scene_id}/click", response_model=ClickReadyResponse | ClickProcessingResponse)
def click_scene(
    scene_id: str,
    request: ClickRequest,
    background_tasks: BackgroundTasks,
    service: SceneService = Depends(get_scene_service),
) -> ClickReadyResponse | ClickProcessingResponse:
    """Handle a scene click and return the next scene transition package."""
    if settings.async_jobs_enabled and settings.image_generator_backend != "mock":
        response = service.start_click_job(
            scene_id=scene_id,
            x=request.x,
            y=request.y,
            hotspot_id=request.hotspot_id,
        )
        if isinstance(response, ClickProcessingResponse):
            cache_key = service.click_cache_key(
                scene_id=scene_id,
                x=request.x,
                y=request.y,
                hotspot_id=request.hotspot_id,
            )
            background_tasks.add_task(
                service.complete_click_job,
                job_id=response.job_id,
                scene_id=scene_id,
                x=request.x,
                y=request.y,
                hotspot_id=request.hotspot_id,
                cache_key=cache_key,
            )
        return response

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
    background_tasks: BackgroundTasks,
    service: SceneService = Depends(get_scene_service),
) -> JobResponse:
    """Prepare and cache the next scene for a likely click."""
    if settings.async_jobs_enabled and settings.image_generator_backend != "mock":
        response = service.start_prefetch_job(
            scene_id=scene_id,
            x=request.x,
            y=request.y,
            hotspot_id=request.hotspot_id,
        )
        if response.status == "processing":
            cache_key = service.click_cache_key(
                scene_id=scene_id,
                x=request.x,
                y=request.y,
                hotspot_id=request.hotspot_id,
            )
            background_tasks.add_task(
                service.complete_prefetch_job,
                job_id=response.job_id,
                scene_id=scene_id,
                x=request.x,
                y=request.y,
                hotspot_id=request.hotspot_id,
                cache_key=cache_key,
            )
        return response

    return service.prefetch_click(
        scene_id=scene_id,
        x=request.x,
        y=request.y,
        hotspot_id=request.hotspot_id,
    )
