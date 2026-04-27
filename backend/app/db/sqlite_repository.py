import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from backend.app.core.ids import new_id
from backend.app.db.models import HotspotRecord, JobRecord, SceneRecord, StoryRecord
from backend.app.schemas.click_target import ClickTarget


class SQLiteRepository:
    """SQLite-backed repository with the same contract as the in-memory store."""

    def __init__(self, database_path: str) -> None:
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def reset(self) -> None:
        """Delete all persisted records. Intended for tests and local resets."""
        with self._connection() as conn:
            for table in ["click_cache", "jobs", "hotspots", "scenes", "stories"]:
                conn.execute(f"DELETE FROM {table}")

    def create_story(self, prompt: str, style_prompt: str) -> StoryRecord:
        story = StoryRecord(id=new_id("story"), prompt=prompt, style_prompt=style_prompt)
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO stories (id, prompt, style_prompt, current_scene_id)
                VALUES (?, ?, ?, ?)
                """,
                (story.id, story.prompt, story.style_prompt, story.current_scene_id),
            )
        return story

    def set_current_scene(self, story_id: str, scene_id: str) -> None:
        with self._connection() as conn:
            conn.execute(
                "UPDATE stories SET current_scene_id = ? WHERE id = ?",
                (scene_id, story_id),
            )

    def get_story(self, story_id: str) -> StoryRecord | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM stories WHERE id = ?", (story_id,)).fetchone()
        if row is None:
            return None
        return StoryRecord(
            id=row["id"],
            prompt=row["prompt"],
            style_prompt=row["style_prompt"],
            current_scene_id=row["current_scene_id"],
        )

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
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO scenes (
                  id, story_id, parent_scene_id, parent_click_target_json,
                  image_url, width, height, summary
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scene.id,
                    scene.story_id,
                    scene.parent_scene_id,
                    self._dump_click_target(scene.parent_click_target),
                    scene.image_url,
                    scene.width,
                    scene.height,
                    scene.summary,
                ),
            )
        return scene

    def get_scene(self, scene_id: str) -> SceneRecord | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM scenes WHERE id = ?", (scene_id,)).fetchone()
        if row is None:
            return None
        return self._scene_from_row(row)

    def save_hotspots(self, scene_id: str, hotspots: list[HotspotRecord]) -> None:
        with self._connection() as conn:
            for hotspot in hotspots:
                if hotspot.scene_id != scene_id:
                    raise ValueError("Hotspot scene_id does not match target scene.")
                conn.execute(
                    """
                    INSERT OR REPLACE INTO hotspots (
                      id, scene_id, label, query, bbox_json, mask_url, next_hint
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        hotspot.id,
                        hotspot.scene_id,
                        hotspot.label,
                        hotspot.query,
                        json.dumps(hotspot.bbox),
                        hotspot.mask_url,
                        hotspot.next_hint,
                    ),
                )

    def get_hotspot(self, hotspot_id: str) -> HotspotRecord | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM hotspots WHERE id = ?", (hotspot_id,)).fetchone()
        if row is None:
            return None
        return self._hotspot_from_row(row)

    def list_hotspots(self, scene_id: str) -> list[HotspotRecord]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM hotspots WHERE scene_id = ? ORDER BY id",
                (scene_id,),
            ).fetchall()
        return [self._hotspot_from_row(row) for row in rows]

    def create_job(self, job_type: str, input_json: dict | None = None) -> JobRecord:
        job = JobRecord(
            id=new_id("job"),
            type=job_type,
            status="processing",
            input_json=input_json,
        )
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO jobs (id, type, status, input_json, output_json, error)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.type,
                    job.status,
                    self._dump_json(job.input_json),
                    None,
                    None,
                ),
            )
        return job

    def complete_job(self, job_id: str, output_json: dict) -> JobRecord:
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, output_json = ?, error = NULL
                WHERE id = ?
                """,
                ("done", self._dump_json(output_json), job_id),
            )
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def fail_job(self, job_id: str, error: str) -> JobRecord:
        with self._connection() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ? WHERE id = ?",
                ("failed", error, job_id),
            )
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return JobRecord(
            id=row["id"],
            type=row["type"],
            status=row["status"],
            input_json=self._load_json(row["input_json"]),
            output_json=self._load_json(row["output_json"]),
            error=row["error"],
        )

    def get_click_cache(self, cache_key: str) -> dict | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT response_json FROM click_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return self._load_json(row["response_json"])

    def save_click_cache(self, cache_key: str, response_json: dict) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO click_cache (cache_key, response_json)
                VALUES (?, ?)
                """,
                (cache_key, self._dump_json(response_json)),
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._database_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS stories (
                  id TEXT PRIMARY KEY,
                  prompt TEXT NOT NULL,
                  style_prompt TEXT NOT NULL,
                  current_scene_id TEXT
                );

                CREATE TABLE IF NOT EXISTS scenes (
                  id TEXT PRIMARY KEY,
                  story_id TEXT NOT NULL,
                  parent_scene_id TEXT,
                  parent_click_target_json TEXT,
                  image_url TEXT NOT NULL,
                  width INTEGER NOT NULL,
                  height INTEGER NOT NULL,
                  summary TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS hotspots (
                  id TEXT PRIMARY KEY,
                  scene_id TEXT NOT NULL,
                  label TEXT NOT NULL,
                  query TEXT NOT NULL,
                  bbox_json TEXT NOT NULL,
                  mask_url TEXT,
                  next_hint TEXT
                );

                CREATE TABLE IF NOT EXISTS jobs (
                  id TEXT PRIMARY KEY,
                  type TEXT NOT NULL,
                  status TEXT NOT NULL,
                  input_json TEXT,
                  output_json TEXT,
                  error TEXT
                );

                CREATE TABLE IF NOT EXISTS click_cache (
                  cache_key TEXT PRIMARY KEY,
                  response_json TEXT NOT NULL
                );
                """
            )

    def _scene_from_row(self, row: sqlite3.Row) -> SceneRecord:
        parent_click_target = self._load_click_target(row["parent_click_target_json"])
        return SceneRecord(
            id=row["id"],
            story_id=row["story_id"],
            image_url=row["image_url"],
            width=row["width"],
            height=row["height"],
            summary=row["summary"],
            parent_scene_id=row["parent_scene_id"],
            parent_click_target=parent_click_target,
        )

    def _hotspot_from_row(self, row: sqlite3.Row) -> HotspotRecord:
        return HotspotRecord(
            id=row["id"],
            scene_id=row["scene_id"],
            label=row["label"],
            query=row["query"],
            bbox=json.loads(row["bbox_json"]),
            mask_url=row["mask_url"],
            next_hint=row["next_hint"],
        )

    def _dump_click_target(self, click_target: ClickTarget | None) -> str | None:
        if click_target is None:
            return None
        return click_target.model_dump_json()

    def _load_click_target(self, payload: str | None) -> ClickTarget | None:
        if payload is None:
            return None
        return ClickTarget.model_validate_json(payload)

    def _dump_json(self, payload: dict | None) -> str | None:
        if payload is None:
            return None
        return json.dumps(payload)

    def _load_json(self, payload: str | None) -> dict | None:
        if payload is None:
            return None
        return json.loads(payload)
