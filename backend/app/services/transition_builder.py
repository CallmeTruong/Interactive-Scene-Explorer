from backend.app.db.models import HotspotRecord, SceneRecord
from backend.app.schemas.click_target import ClickTarget
from backend.app.schemas.transition import TransitionFocus, TransitionPackage, TransitionScene


class TransitionBuilder:
    def build(
        self,
        *,
        from_scene: SceneRecord,
        to_scene: SceneRecord,
        click_target: ClickTarget,
        next_hotspots: list[HotspotRecord] | None = None,
    ) -> TransitionPackage:
        """Build the best available PowerPoint-like transition package."""
        target_hotspot = self._find_target_hotspot(
            click_target=click_target,
            next_hotspots=next_hotspots or [],
        )
        mode = "zoom_crossfade"
        if target_hotspot is not None:
            mode = "bbox_to_bbox_morph"
        elif click_target.mask_url is not None:
            mode = "zoom_mask_crossfade"

        return TransitionPackage(
            type="ppt_morph_like",
            mode=mode,
            duration_ms=1050,
            from_scene=TransitionScene(
                image_url=from_scene.image_url,
                width=from_scene.width,
                height=from_scene.height,
            ),
            to_scene=TransitionScene(
                image_url=to_scene.image_url,
                width=to_scene.width,
                height=to_scene.height,
            ),
            focus=TransitionFocus(
                label=click_target.label,
                from_bbox=click_target.bbox,
                from_mask_url=click_target.mask_url,
                to_bbox=target_hotspot.bbox if target_hotspot else None,
                to_mask_url=target_hotspot.mask_url if target_hotspot else None,
            ),
        )

    def _find_target_hotspot(
        self,
        *,
        click_target: ClickTarget,
        next_hotspots: list[HotspotRecord],
    ) -> HotspotRecord | None:
        """Find a matching target in the next scene by simple label overlap."""
        target_label = self._normalize(click_target.label)
        for hotspot in next_hotspots:
            hotspot_label = self._normalize(hotspot.label)
            if target_label in hotspot_label or hotspot_label in target_label:
                return hotspot
        return next_hotspots[0] if next_hotspots else None

    def _normalize(self, value: str) -> str:
        return " ".join(value.lower().strip().split())
