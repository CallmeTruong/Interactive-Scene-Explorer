from pathlib import Path

from backend.app.core.config import settings


class Storage:
    @property
    def assets_dir(self) -> Path:
        return settings.static_dir / "assets"

    def scene_url(self, filename: str) -> str:
        return f"{settings.static_url_prefix}/assets/scenes/{filename}"

