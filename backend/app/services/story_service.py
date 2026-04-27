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
        root_scene_id = self._create_root_scene_for_story(
            story_id=story.id,
            prompt=prompt,
            style=style,
        )
        return StoryCreateResponse(status="ready", story_id=story.id, root_scene_id=root_scene_id)

    def start_create_story_job(self, *, prompt: str, style: str) -> StoryCreateResponse:
        """Create the story record now and generate its root scene in a background job."""
        story = self._repository.create_story(prompt=prompt, style_prompt=style)
        job = self._repository.create_job(
            job_type="create_story",
            input_json={"story_id": story.id, "prompt": prompt, "style": style},
        )
        return StoryCreateResponse(status="processing", story_id=story.id, job_id=job.id)

    def complete_create_story_job(
        self,
        *,
        job_id: str,
        story_id: str,
        prompt: str,
        style: str,
    ) -> None:
        """Generate a root scene and mark the create-story job complete."""
        try:
            root_scene_id = self._create_root_scene_for_story(
                story_id=story_id,
                prompt=prompt,
                style=style,
            )
            self._repository.complete_job(
                job_id,
                {"story_id": story_id, "root_scene_id": root_scene_id},
            )
        except Exception as exc:
            self._repository.fail_job(job_id, str(exc))

    def _create_root_scene_for_story(self, *, story_id: str, prompt: str, style: str) -> str:
        """Generate image and hotspot records for an existing story."""
        root_brief = self._story_planner.create_root_scene(prompt=prompt, style=style)
        image = self._image_generator.generate_root(prompt=root_brief.image_prompt)

        scene = self._repository.create_scene(
            story_id=story_id,
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
        self._repository.set_current_scene(story_id, scene.id)

        return scene.id
