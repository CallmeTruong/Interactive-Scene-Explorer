from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
