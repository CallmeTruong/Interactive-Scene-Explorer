from dataclasses import dataclass
from typing import Protocol

from backend.app.schemas.click_target import ClickTarget


@dataclass(frozen=True)
class PlannedHotspot:
    label: str
    query: str
    next_hint: str


@dataclass(frozen=True)
class SceneBrief:
    summary: str
    image_prompt: str
    primary_hotspots: list[PlannedHotspot]


@dataclass(frozen=True)
class GeneratedImage:
    image_url: str
    width: int
    height: int


class RootScenePlannerProtocol(Protocol):
    def create_root_scene(self, prompt: str, style: str) -> SceneBrief: ...


class NextScenePlannerProtocol(Protocol):
    def plan_next_scene(
        self,
        *,
        original_prompt: str,
        style_prompt: str,
        current_summary: str,
        click_target: ClickTarget,
    ) -> SceneBrief: ...


class ImageGeneratorProtocol(Protocol):
    def generate_root(self, *, prompt: str, width: int, height: int) -> GeneratedImage: ...
    def generate_next(
        self,
        *,
        prompt: str,
        click_target: ClickTarget,
        current_image_url: str | None = None,
        width: int = 1600,
        height: int = 900,
    ) -> GeneratedImage: ...
