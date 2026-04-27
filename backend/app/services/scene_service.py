from backend.app.core.errors import not_found
from backend.app.db.repositories import Repository
from backend.app.schemas.job import JobResponse
from backend.app.schemas.scene import SceneResponse
from backend.app.schemas.transition import ClickReadyResponse
from backend.app.services.click_resolver import ClickResolver
from backend.app.services.hotspot_detector import HotspotDetector
from backend.app.services.image_generator import ImageGenerator
from backend.app.services.next_scene_planner import NextScenePlanner
from backend.app.services.scene_presenter import scene_to_response
from backend.app.services.transition_builder import TransitionBuilder


class SceneService:
    def __init__(
        self,
        *,
        repository: Repository,
        click_resolver: ClickResolver | None = None,
        next_scene_planner: NextScenePlanner | None = None,
        image_generator: ImageGenerator | None = None,
        hotspot_detector: HotspotDetector | None = None,
        transition_builder: TransitionBuilder | None = None,
    ) -> None:
        self._repository = repository
        self._click_resolver = click_resolver or ClickResolver()
        self._next_scene_planner = next_scene_planner or NextScenePlanner()
        self._image_generator = image_generator or ImageGenerator()
        self._hotspot_detector = hotspot_detector or HotspotDetector()
        self._transition_builder = transition_builder or TransitionBuilder()

    def get_scene(self, scene_id: str) -> SceneResponse:
        scene = self._repository.get_scene(scene_id)
        if scene is None:
            raise not_found(f"Scene not found: {scene_id}")

        hotspots = self._repository.list_hotspots(scene.id)
        return scene_to_response(scene, hotspots)

    def handle_click(
        self,
        *,
        scene_id: str,
        x: int,
        y: int,
        hotspot_id: str | None,
    ) -> ClickReadyResponse:
        """Resolve a click, using prefetch cache when available."""
        cache_key = self._cache_key(scene_id=scene_id, x=x, y=y, hotspot_id=hotspot_id)
        cached = self._repository.get_click_cache(cache_key)
        if cached is not None:
            return ClickReadyResponse.model_validate(cached)

        response = self._build_click_response(
            scene_id=scene_id,
            x=x,
            y=y,
            hotspot_id=hotspot_id,
        )
        self._repository.save_click_cache(cache_key, response.model_dump(mode="json"))
        return response

    def prefetch_click(
        self,
        *,
        scene_id: str,
        x: int,
        y: int,
        hotspot_id: str | None,
    ) -> JobResponse:
        """Build and cache the next scene for a likely click, returning a completed job."""
        cache_key = self._cache_key(scene_id=scene_id, x=x, y=y, hotspot_id=hotspot_id)
        cached = self._repository.get_click_cache(cache_key)
        if cached is None:
            response = self._build_click_response(
                scene_id=scene_id,
                x=x,
                y=y,
                hotspot_id=hotspot_id,
            )
            cached = response.model_dump(mode="json")
            self._repository.save_click_cache(cache_key, cached)

        job = self._repository.create_job(
            job_type="prefetch_click",
            input_json={
                "scene_id": scene_id,
                "x": x,
                "y": y,
                "hotspot_id": hotspot_id,
            },
        )
        completed = self._repository.complete_job(job.id, cached)
        return JobResponse(
            job_id=completed.id,
            status="done",
            result=completed.output_json,
            error=None,
        )

    def _build_click_response(
        self,
        *,
        scene_id: str,
        x: int,
        y: int,
        hotspot_id: str | None,
    ) -> ClickReadyResponse:
        scene = self._repository.get_scene(scene_id)
        if scene is None:
            raise not_found(f"Scene not found: {scene_id}")

        story = self._repository.get_story(scene.story_id)
        if story is None:
            raise not_found(f"Story not found: {scene.story_id}")

        hotspots = self._repository.list_hotspots(scene.id)
        click_target = self._click_resolver.resolve(
            scene=scene,
            hotspots=hotspots,
            x=x,
            y=y,
            hotspot_id=hotspot_id,
        )

        next_brief = self._next_scene_planner.plan(
            story=story,
            current_scene=scene,
            click_target=click_target,
        )
        next_image = self._image_generator.generate_next(
            prompt=next_brief.image_prompt,
            click_target=click_target,
        )
        next_scene = self._repository.create_scene(
            story_id=story.id,
            parent_scene_id=scene.id,
            parent_click_target=click_target,
            image_url=next_image.image_url,
            width=next_image.width,
            height=next_image.height,
            summary=next_brief.summary,
        )

        next_hotspots = self._hotspot_detector.detect_primary(
            scene_id=next_scene.id,
            primary_hotspots=next_brief.primary_hotspots,
        )
        self._repository.save_hotspots(next_scene.id, next_hotspots)
        self._repository.set_current_scene(story.id, next_scene.id)

        transition = self._transition_builder.build(
            from_scene=scene,
            to_scene=next_scene,
            click_target=click_target,
            next_hotspots=next_hotspots,
        )

        return ClickReadyResponse(
            status="ready",
            click_target=click_target,
            next_scene=scene_to_response(next_scene, next_hotspots),
            transition=transition,
        )

    def _cache_key(self, *, scene_id: str, x: int, y: int, hotspot_id: str | None) -> str:
        """Build a stable cache key for click or hover-prefetch results."""
        return f"click:{scene_id}:{hotspot_id or '-'}:{x}:{y}"
