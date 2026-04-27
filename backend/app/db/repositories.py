from threading import RLock
from typing import Protocol

from backend.app.core.config import settings
from backend.app.core.ids import new_id
from backend.app.db.models import HotspotRecord, JobRecord, SceneRecord, StoryRecord
from backend.app.schemas.click_target import ClickTarget


class Repository(Protocol):
    def reset(self) -> None: ...
    def create_story(self, prompt: str, style_prompt: str) -> StoryRecord: ...
    def set_current_scene(self, story_id: str, scene_id: str) -> None: ...
    def get_story(self, story_id: str) -> StoryRecord | None: ...
    def create_scene(
        self,
        *,
        story_id: str,
        image_url: str,
        width: int,
        height: int,
        summary: str,
        parent_scene_id: str | None = None,
        parent_click_target: ClickTarget | None = None,
    ) -> SceneRecord: ...
    def get_scene(self, scene_id: str) -> SceneRecord | None: ...
    def save_hotspots(self, scene_id: str, hotspots: list[HotspotRecord]) -> None: ...
    def get_hotspot(self, hotspot_id: str) -> HotspotRecord | None: ...
    def list_hotspots(self, scene_id: str) -> list[HotspotRecord]: ...
    def create_job(self, job_type: str, input_json: dict | None = None) -> JobRecord: ...
    def complete_job(self, job_id: str, output_json: dict) -> JobRecord: ...
    def fail_job(self, job_id: str, error: str) -> JobRecord: ...
    def get_job(self, job_id: str) -> JobRecord | None: ...
    def get_click_cache(self, cache_key: str) -> dict | None: ...
    def save_click_cache(self, cache_key: str, response_json: dict) -> None: ...


class InMemoryRepository:
    def __init__(self) -> None:
        self._lock = RLock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.stories: dict[str, StoryRecord] = {}
            self.scenes: dict[str, SceneRecord] = {}
            self.hotspots: dict[str, HotspotRecord] = {}
            self.jobs: dict[str, JobRecord] = {}
            self.click_cache: dict[str, dict] = {}

    def create_story(self, prompt: str, style_prompt: str) -> StoryRecord:
        with self._lock:
            story = StoryRecord(id=new_id("story"), prompt=prompt, style_prompt=style_prompt)
            self.stories[story.id] = story
            return story

    def set_current_scene(self, story_id: str, scene_id: str) -> None:
        with self._lock:
            self.stories[story_id].current_scene_id = scene_id

    def get_story(self, story_id: str) -> StoryRecord | None:
        return self.stories.get(story_id)

    def create_scene(
        self,
        *,
        story_id: str,
        image_url: str,
        width: int,
        height: int,
        summary: str,
        parent_scene_id: str | None = None,
        parent_click_target: ClickTarget | None = None,
    ) -> SceneRecord:
        with self._lock:
            scene = SceneRecord(
                id=new_id("scene"),
                story_id=story_id,
                image_url=image_url,
                width=width,
                height=height,
                summary=summary,
                parent_scene_id=parent_scene_id,
                parent_click_target=parent_click_target,
            )
            self.scenes[scene.id] = scene
            return scene

    def get_scene(self, scene_id: str) -> SceneRecord | None:
        return self.scenes.get(scene_id)

    def save_hotspots(self, scene_id: str, hotspots: list[HotspotRecord]) -> None:
        with self._lock:
            for hotspot in hotspots:
                if hotspot.scene_id != scene_id:
                    raise ValueError("Hotspot scene_id does not match target scene.")
                self.hotspots[hotspot.id] = hotspot

    def get_hotspot(self, hotspot_id: str) -> HotspotRecord | None:
        return self.hotspots.get(hotspot_id)

    def list_hotspots(self, scene_id: str) -> list[HotspotRecord]:
        return [hotspot for hotspot in self.hotspots.values() if hotspot.scene_id == scene_id]

    def get_job(self, job_id: str) -> JobRecord | None:
        return self.jobs.get(job_id)

    def create_job(self, job_type: str, input_json: dict | None = None) -> JobRecord:
        with self._lock:
            job = JobRecord(
                id=new_id("job"),
                type=job_type,
                status="processing",
                input_json=input_json,
            )
            self.jobs[job.id] = job
            return job

    def complete_job(self, job_id: str, output_json: dict) -> JobRecord:
        with self._lock:
            job = self.jobs[job_id]
            job.status = "done"
            job.output_json = output_json
            job.error = None
            return job

    def fail_job(self, job_id: str, error: str) -> JobRecord:
        with self._lock:
            job = self.jobs[job_id]
            job.status = "failed"
            job.error = error
            return job

    def get_click_cache(self, cache_key: str) -> dict | None:
        return self.click_cache.get(cache_key)

    def save_click_cache(self, cache_key: str, response_json: dict) -> None:
        with self._lock:
            self.click_cache[cache_key] = response_json


memory_repository = InMemoryRepository()
sqlite_repository: Repository | None = None


def get_repository() -> Repository:
    if settings.repository_backend == "sqlite":
        global sqlite_repository
        if sqlite_repository is None:
            from backend.app.db.sqlite_repository import SQLiteRepository

            sqlite_repository = SQLiteRepository(settings.sqlite_path)
        return sqlite_repository
    return memory_repository
