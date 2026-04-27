import unittest

from backend.app.db.models import HotspotRecord
from backend.app.db.repositories import InMemoryRepository
from backend.app.services.generation_queue import GenerationQueue
from backend.app.services.scene_service import SceneService


class GenerationQueueTest(unittest.TestCase):
    def test_prefetch_skips_when_generation_queue_is_busy(self) -> None:
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
        queue = GenerationQueue()
        queue._lock.acquire()
        self.addCleanup(queue._lock.release)

        response = SceneService(
            repository=repository,
            generation_queue_instance=queue,
        ).start_prefetch_job(
            scene_id=scene.id,
            x=500,
            y=200,
            hotspot_id="hotspot_1",
        )

        self.assertEqual(response.status, "done")
        self.assertEqual(response.result["skipped"], True)
        self.assertEqual(response.result["reason"], "generation_busy")


if __name__ == "__main__":
    unittest.main()
