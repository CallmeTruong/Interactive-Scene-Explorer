from backend.app.ai.base import SceneBrief
from backend.app.ai.mock_models import MockStoryPlanner


class StoryPlanner:
    def __init__(self, planner: MockStoryPlanner | None = None) -> None:
        self._planner = planner or MockStoryPlanner()

    def create_root_scene(self, *, prompt: str, style: str) -> SceneBrief:
        return self._planner.create_root_scene(prompt=prompt, style=style)

