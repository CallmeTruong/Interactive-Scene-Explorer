from backend.app.ai.base import GeneratedImage
from backend.app.ai.mock_models import MockImageGenerator
from backend.app.core.config import settings
from backend.app.schemas.click_target import ClickTarget


class ImageGenerator:
    def __init__(self, generator: object | None = None) -> None:
        self._generator = generator or self._build_generator()

    def generate_root(
        self,
        *,
        prompt: str,
        width: int = 1600,
        height: int = 900,
    ) -> GeneratedImage:
        return self._generator.generate_root(prompt=prompt, width=width, height=height)

    def generate_next(
        self,
        *,
        prompt: str,
        click_target: ClickTarget,
        current_image_url: str | None = None,
        width: int = 1600,
        height: int = 900,
    ) -> GeneratedImage:
        return self._generator.generate_next(
            prompt=prompt,
            click_target=click_target,
            current_image_url=current_image_url,
            width=width,
            height=height,
        )

    def _build_generator(self) -> object:
        if settings.image_generator_backend in {"sdxl", "diffusion"}:
            from backend.app.ai.diffusion_image_generator import DiffusionImageGenerator

            return DiffusionImageGenerator()
        return MockImageGenerator()
