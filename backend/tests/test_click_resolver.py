import unittest

from backend.app.db.models import HotspotRecord, SceneRecord
from backend.app.services.click_resolver import ClickResolver


class ClickResolverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = SceneRecord(
            id="scene_1",
            story_id="story_1",
            image_url="/static/assets/scenes/root_square.svg",
            width=1600,
            height=900,
            summary="Test scene",
        )
        self.hotspots = [
            HotspotRecord(
                id="hotspot_large",
                scene_id="scene_1",
                label="large object",
                query="large object",
                bbox=[100, 100, 500, 500],
            ),
            HotspotRecord(
                id="hotspot_small",
                scene_id="scene_1",
                label="small object",
                query="small object",
                bbox=[200, 200, 300, 300],
            ),
        ]
        self.resolver = ClickResolver()

    def test_valid_hotspot_id_wins(self) -> None:
        target = self.resolver.resolve(
            scene=self.scene,
            hotspots=self.hotspots,
            x=250,
            y=250,
            hotspot_id="hotspot_large",
        )

        self.assertEqual(target.source, "hotspot_id")
        self.assertEqual(target.label, "large object")

    def test_point_hit_chooses_smallest_bbox(self) -> None:
        target = self.resolver.resolve(
            scene=self.scene,
            hotspots=self.hotspots,
            x=250,
            y=250,
            hotspot_id=None,
        )

        self.assertEqual(target.source, "hotspot_mask")
        self.assertEqual(target.label, "small object")

    def test_fallback_region_stays_inside_image(self) -> None:
        target = self.resolver.resolve(
            scene=self.scene,
            hotspots=[],
            x=5,
            y=5,
            hotspot_id=None,
        )

        self.assertEqual(target.source, "fallback_region")
        self.assertGreaterEqual(target.bbox[0], 0)
        self.assertGreaterEqual(target.bbox[1], 0)
        self.assertLessEqual(target.bbox[2], self.scene.width)
        self.assertLessEqual(target.bbox[3], self.scene.height)


if __name__ == "__main__":
    unittest.main()

