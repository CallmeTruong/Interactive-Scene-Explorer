import unittest

from backend.app.ai.base import GeneratedImage
from backend.app.db.models import HotspotRecord
from backend.app.db.repositories import InMemoryRepository
from backend.app.schemas.click_target import ClickTarget
from backend.app.services.scene_service import SceneService
from backend.app.services.story_service import StoryService


class ResetDuringRootGeneration:
    def __init__(self, repository: InMemoryRepository) -> None:
        self._repository = repository

    def generate_root(self, *, prompt: str, width: int = 1600, height: int = 900) -> GeneratedImage:
        self._repository.reset()
        return GeneratedImage(
            image_url="/static/assets/scenes/generated/stale_root.png",
            width=width,
            height=height,
        )


class ResetDuringNextGeneration:
    def __init__(self, repository: InMemoryRepository) -> None:
        self._repository = repository

    def generate_next(
        self,
        *,
        prompt: str,
        click_target: ClickTarget,
        current_image_url: str | None = None,
        width: int = 1600,
        height: int = 900,
    ) -> GeneratedImage:
        self._repository.reset()
        return GeneratedImage(
            image_url="/static/assets/scenes/generated/stale_next.png",
            width=width,
            height=height,
        )


class StaleBackgroundJobsTest(unittest.TestCase):
    def test_create_story_job_ignores_reset_during_generation(self) -> None:
        repository = InMemoryRepository()
        service = StoryService(
            repository=repository,
            image_generator=ResetDuringRootGeneration(repository),
        )
        response = service.start_create_story_job(prompt="Town", style="Style")

        service.complete_create_story_job(
            job_id=response.job_id,
            story_id=response.story_id,
            prompt="Town",
            style="Style",
        )

        self.assertEqual(repository.stories, {})
        self.assertEqual(repository.jobs, {})
        self.assertEqual(repository.scenes, {})

    def test_click_job_ignores_reset_during_generation(self) -> None:
        repository = InMemoryRepository()
        story = repository.create_story(prompt="Town", style_prompt="Style")
        scene = repository.create_scene(
            story_id=story.id,
            image_url="/static/assets/scenes/root_square.svg",
            width=1600,
            height=900,
            summary="Root scene",
        )
        repository.save_hotspots(
            scene.id,
            [
                HotspotRecord(
                    id="hotspot_1",
                    scene_id=scene.id,
                    label="cathedral",
                    query="cathedral",
                    bbox=[420, 110, 760, 520],
                )
            ],
        )
        service = SceneService(
            repository=repository,
            image_generator=ResetDuringNextGeneration(repository),
        )
        response = service.start_click_job(
            scene_id=scene.id,
            x=500,
            y=200,
            hotspot_id="hotspot_1",
        )

        service.complete_click_job(
            job_id=response.job_id,
            scene_id=scene.id,
            x=500,
            y=200,
            hotspot_id="hotspot_1",
            cache_key=service.click_cache_key(
                scene_id=scene.id,
                x=500,
                y=200,
                hotspot_id="hotspot_1",
            ),
        )

        self.assertEqual(repository.stories, {})
        self.assertEqual(repository.jobs, {})
        self.assertEqual(repository.scenes, {})


if __name__ == "__main__":
    unittest.main()
