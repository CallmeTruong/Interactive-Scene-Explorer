from backend.app.core.config import settings
from backend.app.core.errors import not_found
from backend.app.db.repositories import Repository
from backend.app.schemas.story import (
    StoryCreateResponse,
    StorySceneHistoryItem,
    StorySceneHistoryResponse,
)
from backend.app.services.generated_asset_cleaner import GeneratedAssetCleaner
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
        generated_asset_cleaner: GeneratedAssetCleaner | None = None,
    ) -> None:
        self._repository = repository
        self._story_planner = story_planner or StoryPlanner()
        self._image_generator = image_generator or ImageGenerator()
        self._hotspot_detector = hotspot_detector or HotspotDetector()
        self._generated_asset_cleaner = generated_asset_cleaner or GeneratedAssetCleaner()

    def create_story(self, *, prompt: str, style: str) -> StoryCreateResponse:
        self._clean_generated_assets_for_new_story()
        story = self._repository.create_story(prompt=prompt, style_prompt=style)
        root_scene_id = self._create_root_scene_for_story(
            story_id=story.id,
            prompt=prompt,
            style=style,
        )
        return StoryCreateResponse(status="ready", story_id=story.id, root_scene_id=root_scene_id)

    def start_create_story_job(self, *, prompt: str, style: str) -> StoryCreateResponse:
        """Create the story record now and generate its root scene in a background job."""
        self._clean_generated_assets_for_new_story()
        story = self._repository.create_story(prompt=prompt, style_prompt=style)
        job = self._repository.create_job(
            job_type="create_story",
            input_json={"story_id": story.id, "prompt": prompt, "style": style},
        )
        return StoryCreateResponse(status="processing", story_id=story.id, job_id=job.id)

    def get_scene_history(self, *, story_id: str) -> StorySceneHistoryResponse:
        """Return scenes for a story in creation order for frontend navigation."""
        story = self._repository.get_story(story_id)
        if story is None:
            raise not_found(f"Story not found: {story_id}")

        scenes = self._repository.list_scenes(story_id)
        root_scene = next((scene for scene in scenes if scene.parent_scene_id is None), None)
        return StorySceneHistoryResponse(
            story_id=story.id,
            root_scene_id=root_scene.id if root_scene is not None else None,
            current_scene_id=story.current_scene_id,
            scenes=[
                StorySceneHistoryItem(
                    scene_id=scene.id,
                    parent_scene_id=scene.parent_scene_id,
                    parent_click_target=scene.parent_click_target,
                    image_url=scene.image_url,
                    summary=scene.summary,
                    is_root=scene.parent_scene_id is None,
                    is_current=scene.id == story.current_scene_id,
                )
                for scene in scenes
            ],
        )

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

    def _clean_generated_assets_for_new_story(self) -> None:
        """Clear old generated scene files before starting a fresh local demo story."""
        if settings.cleanup_generated_assets_on_new_story:
            self._generated_asset_cleaner.clean()
