import tempfile
import unittest
from pathlib import Path

from backend.app.db.models import HotspotRecord
from backend.app.db.sqlite_repository import SQLiteRepository
from backend.app.schemas.click_target import ClickTarget


class SQLiteRepositoryTest(unittest.TestCase):
    def test_persists_story_scene_hotspot_job_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.sqlite3"
            repository = SQLiteRepository(str(db_path))

            story = repository.create_story(prompt="Prompt", style_prompt="Style")
            scene = repository.create_scene(
                story_id=story.id,
                image_url="/static/assets/scenes/root_square.svg",
                width=1600,
                height=900,
                summary="Root scene",
            )
            repository.set_current_scene(story.id, scene.id)
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

            click_target = ClickTarget(
                target_id="target_1",
                source="hotspot_id",
                label="cathedral",
                description="Selected hotspot: cathedral",
                bbox=[420, 110, 760, 520],
            )
            child_scene = repository.create_scene(
                story_id=story.id,
                parent_scene_id=scene.id,
                parent_click_target=click_target,
                image_url="/static/assets/scenes/cathedral_closeup.svg",
                width=1600,
                height=900,
                summary="Child scene",
            )
            job = repository.create_job("prefetch_click", {"scene_id": scene.id})
            repository.complete_job(job.id, {"next_scene_id": child_scene.id})
            repository.save_click_cache("cache_key", {"status": "ready"})

            reopened = SQLiteRepository(str(db_path))
            self.assertEqual(reopened.get_story(story.id).current_scene_id, scene.id)
            self.assertEqual(
                [record.id for record in reopened.list_scenes(story.id)],
                [scene.id, child_scene.id],
            )
            self.assertEqual(reopened.get_scene(child_scene.id).parent_click_target.label, "cathedral")
            self.assertEqual(reopened.list_hotspots(scene.id)[0].label, "cathedral")
            self.assertEqual(reopened.get_job(job.id).status, "done")
            self.assertEqual(reopened.get_click_cache("cache_key")["status"], "ready")


if __name__ == "__main__":
    unittest.main()
