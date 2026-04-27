from pathlib import Path

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

    def _target_dir(self) -> Path:
        if self._generated_dir is not None:
            return self._generated_dir
        return settings.static_dir / "assets" / "scenes" / "generated"
