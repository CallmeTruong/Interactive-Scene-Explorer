from backend.app.core.ids import new_id
from backend.app.db.models import HotspotRecord, SceneRecord
from backend.app.schemas.click_target import ClickTarget
from backend.app.services.hotspot_geometry import bbox_area, bbox_around_click, point_in_bbox


class ClickResolver:
    def resolve(
        self,
        *,
        scene: SceneRecord,
        hotspots: list[HotspotRecord],
        x: int,
        y: int,
        hotspot_id: str | None,
    ) -> ClickTarget:
        """Resolve any click into the normalized ClickTarget contract."""
        if hotspot_id:
            hotspot = next(
                (candidate for candidate in hotspots if candidate.id == hotspot_id),
                None,
            )
            if hotspot:
                return self._hotspot_to_click_target(hotspot, source="hotspot_id")

        hit = self._find_hotspot_by_point(hotspots=hotspots, x=x, y=y)
        if hit:
            return self._hotspot_to_click_target(hit, source="hotspot_mask")

        return ClickTarget(
            target_id=new_id("target"),
            source="fallback_region",
            label="selected region",
            description="the selected object or region",
            bbox=bbox_around_click(x=x, y=y, image_w=scene.width, image_h=scene.height),
            mask_url=None,
            next_hint="Explore the selected region",
        )

    def _find_hotspot_by_point(
        self,
        *,
        hotspots: list[HotspotRecord],
        x: int,
        y: int,
    ) -> HotspotRecord | None:
        """Return the smallest hotspot bbox containing the click point."""
        hits = [hotspot for hotspot in hotspots if point_in_bbox(hotspot.bbox, x, y)]
        if not hits:
            return None
        return sorted(hits, key=lambda hotspot: bbox_area(hotspot.bbox))[0]

    def _hotspot_to_click_target(self, hotspot: HotspotRecord, source: str) -> ClickTarget:
        """Convert a known hotspot into a ClickTarget response object."""
        return ClickTarget(
            target_id=new_id("target"),
            source=source,
            label=hotspot.label,
            description=f"Selected hotspot: {hotspot.label}",
            bbox=hotspot.bbox,
            mask_url=hotspot.mask_url,
            next_hint=hotspot.next_hint,
        )
