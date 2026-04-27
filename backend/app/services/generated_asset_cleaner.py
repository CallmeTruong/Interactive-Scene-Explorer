from pathlib import Path
from urllib.parse import urlparse

from backend.app.core.config import settings


class GeneratedAssetCleaner:
    """Remove generated scene images from the local demo runtime directory."""

    _allowed_suffixes = {".png", ".jpg", ".jpeg", ".webp"}

    def __init__(self, generated_dir: Path | None = None) -> None:
        self._generated_dir = generated_dir

    def clean(self) -> int:
        generated_dir = self._target_dir()
        if not generated_dir.exists():
            return 0

        removed = 0
        for path in generated_dir.iterdir():
            if path.is_file() and path.suffix.lower() in self._allowed_suffixes:
                path.unlink()
                removed += 1
        return removed

    def remove_url(self, image_url: str) -> bool:
        image_path = self._resolve_generated_image_path(image_url)
        if image_path is None or not image_path.exists():
            return False

        image_path.unlink()
        return True

    def _target_dir(self) -> Path:
        if self._generated_dir is not None:
            return self._generated_dir
        return settings.static_dir / "assets" / "scenes" / "generated"

    def _resolve_generated_image_path(self, image_url: str) -> Path | None:
        image_path = urlparse(image_url).path
        static_prefix = settings.static_url_prefix.rstrip("/")
        if not image_path.startswith(f"{static_prefix}/"):
            return None

        relative_path = image_path[len(static_prefix) :].lstrip("/")
        candidate = (settings.static_dir / relative_path).resolve()
        generated_dir = self._target_dir().resolve()
        if candidate.parent != generated_dir:
            return None
        if candidate.suffix.lower() not in self._allowed_suffixes:
            return None
        return candidate
