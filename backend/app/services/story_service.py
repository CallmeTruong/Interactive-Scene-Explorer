from backend.app.db.repositories import Repository
from backend.app.schemas.story import StoryCreateResponse
from backend.app.services.hotspot_detector import HotspotDetector
from backend.app.services.image_generator import ImageGenerator
from backend.app.services.story_planner import StoryPlanner


class StoryService:
    def __init__(
        self,
        *,
        repository: Repository,
        story_planner: StoryPlanner | None = None,
        image_generator: ImageGenerator | None = None,
        hotspot_detector: HotspotDetector | None = None,
    ) -> None:
        self._repository = repository
        self._story_planner = story_planner or StoryPlanner()
        self._image_generator = image_generator or ImageGenerator()
        self._hotspot_detector = hotspot_detector or HotspotDetector()

    def create_story(self, *, prompt: str, style: str) -> StoryCreateResponse:
        story = self._repository.create_story(prompt=prompt, style_prompt=style)
        root_brief = self._story_planner.create_root_scene(prompt=prompt, style=style)
        image = self._image_generator.generate_root(prompt=root_brief.image_prompt)

        scene = self._repository.create_scene(
            story_id=story.id,
            image_url=image.image_url,
            width=image.width,
            height=image.height,
            summary=root_brief.summary,
        )

        hotspots = self._hotspot_detector.detect_primary(
            scene_id=scene.id,
            primary_hotspots=root_brief.primary_hotspots,
        )
        self._repository.save_hotspots(scene.id, hotspots)
        self._repository.set_current_scene(story.id, scene.id)

        return StoryCreateResponse(status="ready", story_id=story.id, root_scene_id=scene.id)
