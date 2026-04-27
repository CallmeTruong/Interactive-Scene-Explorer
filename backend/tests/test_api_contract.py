import unittest

from fastapi.testclient import TestClient

from backend.app.db.repositories import get_repository
from backend.app.main import create_app


class ApiContractTest(unittest.TestCase):
    def setUp(self) -> None:
        get_repository().reset()
        self.client = TestClient(create_app())

    def test_story_scene_and_click_happy_path(self) -> None:
        story_response = self.client.post(
            "/stories",
            json={"prompt": "Create an explorable old European town", "style": "illustrated"},
        )
        self.assertEqual(story_response.status_code, 200)
        story_payload = story_response.json()
        self.assertEqual(story_payload["status"], "ready")
        self.assertIsNotNone(story_payload["root_scene_id"])

        scene_response = self.client.get(f"/scenes/{story_payload['root_scene_id']}")
        self.assertEqual(scene_response.status_code, 200)
        scene_payload = scene_response.json()
        self.assertGreaterEqual(len(scene_payload["hotspots"]), 3)

        hotspot = scene_payload["hotspots"][0]
        x1, y1, x2, y2 = hotspot["bbox"]
        click_response = self.client.post(
            f"/scenes/{scene_payload['scene_id']}/click",
            json={
                "x": (x1 + x2) // 2,
                "y": (y1 + y2) // 2,
                "hotspot_id": hotspot["hotspot_id"],
            },
        )
        self.assertEqual(click_response.status_code, 200)
        click_payload = click_response.json()

        self.assertEqual(click_payload["status"], "ready")
        self.assertEqual(click_payload["click_target"]["source"], "hotspot_id")
        self.assertEqual(
            click_payload["next_scene"]["parent_scene_id"],
            scene_payload["scene_id"],
        )
        self.assertEqual(click_payload["transition"]["mode"], "bbox_to_bbox_morph")
        self.assertEqual(
            click_payload["transition"]["focus"]["from_bbox"],
            click_payload["click_target"]["bbox"],
        )
        self.assertIsNotNone(click_payload["transition"]["focus"]["to_bbox"])
        self.assertIn("hotspots", click_payload["next_scene"])

    def test_click_outside_hotspots_returns_fallback_region(self) -> None:
        story_response = self.client.post(
            "/stories",
            json={"prompt": "Create an explorable old European town"},
        )
        root_scene_id = story_response.json()["root_scene_id"]

        click_response = self.client.post(
            f"/scenes/{root_scene_id}/click",
            json={"x": 20, "y": 850, "hotspot_id": None},
        )
        self.assertEqual(click_response.status_code, 200)
        click_payload = click_response.json()
        self.assertEqual(click_payload["click_target"]["source"], "fallback_region")

    def test_unknown_scene_returns_404(self) -> None:
        response = self.client.get("/scenes/unknown_scene")

        self.assertEqual(response.status_code, 404)

    def test_invalid_hotspot_id_can_still_hit_bbox(self) -> None:
        story_response = self.client.post(
            "/stories",
            json={"prompt": "Create an explorable old European town"},
        )
        scene_response = self.client.get(f"/scenes/{story_response.json()['root_scene_id']}")
        scene_payload = scene_response.json()
        hotspot = scene_payload["hotspots"][0]
        x1, y1, x2, y2 = hotspot["bbox"]

        click_response = self.client.post(
            f"/scenes/{scene_payload['scene_id']}/click",
            json={
                "x": (x1 + x2) // 2,
                "y": (y1 + y2) // 2,
                "hotspot_id": "not_a_real_hotspot",
            },
        )

        self.assertEqual(click_response.status_code, 200)
        self.assertEqual(click_response.json()["click_target"]["source"], "hotspot_mask")

    def test_prefetch_returns_done_job_and_click_uses_cached_scene(self) -> None:
        story_response = self.client.post(
            "/stories",
            json={"prompt": "Create an explorable old European town"},
        )
        scene_response = self.client.get(f"/scenes/{story_response.json()['root_scene_id']}")
        scene_payload = scene_response.json()
        hotspot = scene_payload["hotspots"][0]
        x1, y1, x2, y2 = hotspot["bbox"]
        click_payload = {
            "x": (x1 + x2) // 2,
            "y": (y1 + y2) // 2,
            "hotspot_id": hotspot["hotspot_id"],
        }

        prefetch_response = self.client.post(
            f"/scenes/{scene_payload['scene_id']}/prefetch",
            json=click_payload,
        )
        self.assertEqual(prefetch_response.status_code, 200)
        prefetch_payload = prefetch_response.json()
        self.assertEqual(prefetch_payload["status"], "done")

        click_response = self.client.post(
            f"/scenes/{scene_payload['scene_id']}/click",
            json=click_payload,
        )
        self.assertEqual(click_response.status_code, 200)
        self.assertEqual(
            click_response.json()["next_scene"]["scene_id"],
            prefetch_payload["result"]["next_scene"]["scene_id"],
        )


if __name__ == "__main__":
    unittest.main()
