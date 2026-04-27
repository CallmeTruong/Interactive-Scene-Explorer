from typing import Literal

from pydantic import BaseModel, ConfigDict


class DemoResetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    generated_assets_removed: int
