from pydantic import BaseModel, ConfigDict

from backend.app.schemas.click_target import ClickTarget
from backend.app.schemas.hotspot import Hotspot


class SceneResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    story_id: str
    parent_scene_id: str | None = None
    parent_click_target: ClickTarget | None = None
    image_url: str
    width: int
    height: int
    summary: str
    hotspots: list[Hotspot]


class ClickRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int
    y: int
    hotspot_id: str | None = None

