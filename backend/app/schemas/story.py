from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.app.schemas.click_target import ClickTarget


class StoryCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    style: str = "fixed LoRA style"


class StoryCreateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "processing"]
    story_id: str
    root_scene_id: str | None = None
    job_id: str | None = None


class StorySceneHistoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    parent_scene_id: str | None = None
    parent_click_target: ClickTarget | None = None
    image_url: str
    summary: str
    is_root: bool
    is_current: bool


class StorySceneHistoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    story_id: str
    root_scene_id: str | None = None
    current_scene_id: str | None = None
    scenes: list[StorySceneHistoryItem]
