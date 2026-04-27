def point_in_bbox(bbox: list[int], x: int, y: int) -> bool:
    """Check whether an image-coordinate point is inside a bbox."""
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def bbox_area(bbox: list[int]) -> int:
    """Compute bbox area, clamping invalid dimensions to zero."""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_around_click(x: int, y: int, image_w: int, image_h: int, ratio: float = 0.25) -> list[int]:
    """Build a fallback bbox around a click, clipped to image bounds."""
    x = min(max(x, 0), image_w)
    y = min(max(y, 0), image_h)
    side = int(min(image_w, image_h) * ratio)
    half = side // 2
    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(image_w, x + half)
    y2 = min(image_h, y + half)
    return [x1, y1, x2, y2]
