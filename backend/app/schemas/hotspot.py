from pydantic import BaseModel, ConfigDict

from backend.app.schemas.common import BBox


class Hotspot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hotspot_id: str
    scene_id: str
    label: str
    query: str
    bbox: BBox
    mask_url: str | None = None
    next_hint: str | None = None

