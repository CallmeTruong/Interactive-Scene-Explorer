from dataclasses import dataclass

from backend.app.schemas.click_target import ClickTarget


@dataclass(slots=True)
class StoryRecord:
    id: str
    prompt: str
    style_prompt: str
    current_scene_id: str | None = None


@dataclass(slots=True)
class SceneRecord:
    id: str
    story_id: str
    image_url: str
    width: int
    height: int
    summary: str
    parent_scene_id: str | None = None
    parent_click_target: ClickTarget | None = None


@dataclass(slots=True)
class HotspotRecord:
    id: str
    scene_id: str
    label: str
    query: str
    bbox: list[int]
    mask_url: str | None = None
    next_hint: str | None = None


@dataclass(slots=True)
class JobRecord:
    id: str
    type: str
    status: str
    input_json: dict | None = None
    output_json: dict | None = None
    error: str | None = None


@dataclass(slots=True)
class ClickCacheRecord:
    cache_key: str
    response_json: dict
