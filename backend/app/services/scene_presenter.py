from backend.app.db.models import HotspotRecord, SceneRecord
from backend.app.schemas.hotspot import Hotspot
from backend.app.schemas.scene import SceneResponse


def scene_to_response(scene: SceneRecord, hotspots: list[HotspotRecord]) -> SceneResponse:
    return SceneResponse(
        scene_id=scene.id,
        story_id=scene.story_id,
        parent_scene_id=scene.parent_scene_id,
        parent_click_target=scene.parent_click_target,
        image_url=scene.image_url,
        width=scene.width,
        height=scene.height,
        summary=scene.summary,
        hotspots=[
            Hotspot(
                hotspot_id=hotspot.id,
                scene_id=hotspot.scene_id,
                label=hotspot.label,
                query=hotspot.query,
                bbox=hotspot.bbox,
                mask_url=hotspot.mask_url,
                next_hint=hotspot.next_hint,
            )
            for hotspot in hotspots
        ],
    )

