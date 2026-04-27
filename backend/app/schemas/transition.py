from typing import Literal

from pydantic import BaseModel, ConfigDict

from backend.app.schemas.click_target import ClickTarget
from backend.app.schemas.common import BBox
from backend.app.schemas.scene import SceneResponse

TransitionMode = Literal["zoom_crossfade", "zoom_mask_crossfade", "bbox_to_bbox_morph"]


class TransitionScene(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_url: str
    width: int
    height: int


class TransitionFocus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    from_bbox: BBox
    from_mask_url: str | None = None
    to_bbox: BBox | None = None
    to_mask_url: str | None = None


class TransitionPackage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["ppt_morph_like"]
    mode: TransitionMode
    duration_ms: int
    from_scene: TransitionScene
    to_scene: TransitionScene
    focus: TransitionFocus


class ClickReadyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ready"]
    click_target: ClickTarget
    next_scene: SceneResponse
    transition: TransitionPackage


class TransitionPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["zoom_hold"]
    from_bbox: BBox
    from_mask_url: str | None = None


class ClickProcessingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["processing"]
    job_id: str
    click_target: ClickTarget
    transition_preview: TransitionPreview
