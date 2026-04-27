from backend.app.ai.base import GeneratedImage, PlannedHotspot, SceneBrief
from backend.app.schemas.click_target import ClickTarget


class MockStoryPlanner:
    def create_root_scene(self, prompt: str, style: str) -> SceneBrief:
        return SceneBrief(
            summary=(
                "An old European town square with a cathedral, cobblestone road, "
                "horse carriage, cafe, and stone fountain."
            ),
            image_prompt=(
                f"{prompt}. Style: {style}. Clear focal objects: cathedral, "
                "cobblestone road, horse carriage, small cafe, stone fountain. No text."
            ),
            primary_hotspots=[
                PlannedHotspot(
                    label="cathedral",
                    query="cathedral",
                    next_hint="Explore the cathedral exterior",
                ),
                PlannedHotspot(
                    label="cobblestone road",
                    query="cobblestone road",
                    next_hint="Follow the road to the side street",
                ),
                PlannedHotspot(
                    label="horse carriage",
                    query="horse carriage",
                    next_hint="Inspect the horse carriage",
                ),
                PlannedHotspot(
                    label="small cafe",
                    query="small cafe",
                    next_hint="Explore the cafe",
                ),
                PlannedHotspot(
                    label="stone fountain",
                    query="stone fountain",
                    next_hint="Explore the fountain",
                ),
            ],
        )


class MockNextScenePlanner:
    def plan_next_scene(
        self,
        *,
        original_prompt: str,
        style_prompt: str,
        current_summary: str,
        click_target: ClickTarget,
    ) -> SceneBrief:
        label = click_target.label
        return SceneBrief(
            summary=f"A closer illustrated view of {label} and nearby details.",
            image_prompt=(
                f"Create a closer view of {label}. Preserve the same world as: "
                f"{original_prompt}. Current scene: {current_summary}. "
                f"Style: {style_prompt}. No text."
            ),
            primary_hotspots=[
                PlannedHotspot(
                    label=f"{label} detail",
                    query=f"{label} detail",
                    next_hint=f"Inspect {label} detail",
                ),
                PlannedHotspot(
                    label="nearby path",
                    query="nearby path",
                    next_hint="Move along the nearby path",
                ),
                PlannedHotspot(
                    label="background architecture",
                    query="background architecture",
                    next_hint="Look toward the surrounding architecture",
                ),
            ],
        )


class MockImageGenerator:
    def generate_root(self, *, prompt: str, width: int, height: int) -> GeneratedImage:
        return GeneratedImage(
            image_url="/static/assets/scenes/root_square.svg",
            width=width,
            height=height,
        )

    def generate_next(
        self,
        *,
        prompt: str,
        click_target: ClickTarget,
        width: int,
        height: int,
    ) -> GeneratedImage:
        if "cathedral" in click_target.label:
            image_name = "cathedral_closeup.svg"
        elif "fountain" in click_target.label:
            image_name = "fountain_closeup.svg"
        else:
            image_name = "selected_region_closeup.svg"

        return GeneratedImage(
            image_url=f"/static/assets/scenes/{image_name}",
            width=width,
            height=height,
        )
