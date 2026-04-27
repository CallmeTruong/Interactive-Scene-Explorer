from backend.app.ai.base import PlannedHotspot
from backend.app.core.ids import new_id
from backend.app.db.models import HotspotRecord


class HotspotDetector:
    def detect_primary(
        self,
        *,
        scene_id: str,
        primary_hotspots: list[PlannedHotspot],
    ) -> list[HotspotRecord]:
        """Create deterministic mock hotspots for a scene brief."""
        hotspots: list[HotspotRecord] = []
        for index, planned in enumerate(primary_hotspots):
            bbox = self._bbox_for(planned.query, index)
            hotspots.append(
                HotspotRecord(
                    id=new_id("hotspot"),
                    scene_id=scene_id,
                    label=planned.label,
                    query=planned.query,
                    bbox=bbox,
                    mask_url=None,
                    next_hint=planned.next_hint,
                )
            )
        return hotspots

    def _bbox_for(self, query: str, index: int) -> list[int]:
        """Return a stable mock bbox for known or generated hotspot queries."""
        known_boxes = {
            "cathedral": [420, 110, 760, 520],
            "cobblestone road": [510, 610, 1120, 890],
            "horse carriage": [930, 485, 1245, 665],
            "small cafe": [1050, 230, 1460, 560],
            "stone fountain": [705, 465, 910, 650],
        }
        if query in known_boxes:
            return known_boxes[query]

        x1 = 180 + index * 260
        y1 = 180 + (index % 2) * 260
        return [x1, y1, min(x1 + 260, 1550), min(y1 + 190, 850)]
