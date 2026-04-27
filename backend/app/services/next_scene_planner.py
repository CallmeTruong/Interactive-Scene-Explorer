from backend.app.ai.base import SceneBrief
from backend.app.ai.mock_models import MockNextScenePlanner
from backend.app.db.models import SceneRecord, StoryRecord
from backend.app.schemas.click_target import ClickTarget


class NextScenePlanner:
    def __init__(self, planner: MockNextScenePlanner | None = None) -> None:
        self._planner = planner or MockNextScenePlanner()

    def plan(
        self,
        *,
        story: StoryRecord,
        current_scene: SceneRecord,
        click_target: ClickTarget,
    ) -> SceneBrief:
        return self._planner.plan_next_scene(
            original_prompt=story.prompt,
            style_prompt=story.style_prompt,
            current_summary=current_scene.summary,
            click_target=click_target,
        )

