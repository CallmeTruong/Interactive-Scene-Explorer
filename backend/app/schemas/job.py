from typing import Literal

from pydantic import BaseModel, ConfigDict


class JobResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: Literal["processing", "done", "failed"]
    result: dict | None = None
    error: str | None = None

