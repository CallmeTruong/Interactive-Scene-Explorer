from typing import Literal

from pydantic import BaseModel, ConfigDict

from backend.app.schemas.common import BBox

ClickTargetSource = Literal[
    "hotspot_id",
    "hotspot_mask",
    "on_demand_segmentation",
    "fallback_region",
]


class ClickTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: str
    source: ClickTargetSource
    label: str
    description: str
    bbox: BBox
    mask_url: str | None = None
    next_hint: str | None = None

