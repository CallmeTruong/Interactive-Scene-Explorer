import unittest

from backend.app.db.models import HotspotRecord, SceneRecord
from backend.app.schemas.click_target import ClickTarget
from backend.app.services.transition_builder import TransitionBuilder


class TransitionBuilderTest(unittest.TestCase):
    def test_builds_zoom_crossfade_transition(self) -> None:
        from_scene = SceneRecord(
            id="scene_from",
            story_id="story_1",
            image_url="/static/assets/scenes/root_square.svg",
            width=1600,
            height=900,
            summary="From scene",
        )
        to_scene = SceneRecord(
            id="scene_to",
            story_id="story_1",
            image_url="/static/assets/scenes/fountain_closeup.svg",
            width=1600,
            height=900,
            summary="To scene",
        )
        click_target = ClickTarget(
            target_id="target_1",
            source="hotspot_id",
            label="stone fountain",
            description="Selected hotspot: stone fountain",
            bbox=[705, 465, 910, 650],
            mask_url=None,
            next_hint="Explore the fountain",
        )

        transition = TransitionBuilder().build(
            from_scene=from_scene,
            to_scene=to_scene,
            click_target=click_target,
        )

        self.assertEqual(transition.type, "ppt_morph_like")
        self.assertEqual(transition.mode, "zoom_crossfade")
        self.assertEqual(transition.focus.from_bbox, click_target.bbox)
        self.assertEqual(transition.to_scene.image_url, to_scene.image_url)

    def test_builds_bbox_morph_when_next_scene_has_matching_hotspot(self) -> None:
        from_scene = SceneRecord(
            id="scene_from",
            story_id="story_1",
            image_url="/static/assets/scenes/root_square.svg",
            width=1600,
            height=900,
            summary="From scene",
        )
        to_scene = SceneRecord(
            id="scene_to",
            story_id="story_1",
            image_url="/static/assets/scenes/fountain_closeup.svg",
            width=1600,
            height=900,
            summary="To scene",
        )
        click_target = ClickTarget(
            target_id="target_1",
            source="hotspot_id",
            label="stone fountain",
            description="Selected hotspot: stone fountain",
            bbox=[705, 465, 910, 650],
        )
        next_hotspot = HotspotRecord(
            id="hotspot_1",
            scene_id="scene_to",
            label="stone fountain detail",
            query="stone fountain detail",
            bbox=[180, 180, 440, 370],
        )

        transition = TransitionBuilder().build(
            from_scene=from_scene,
            to_scene=to_scene,
            click_target=click_target,
            next_hotspots=[next_hotspot],
        )

        self.assertEqual(transition.mode, "bbox_to_bbox_morph")
        self.assertEqual(transition.focus.to_bbox, next_hotspot.bbox)


if __name__ == "__main__":
    unittest.main()
