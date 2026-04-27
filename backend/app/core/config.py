import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "Interactive AI Scene Explorer"
    app_env: str = "development"
    static_url_prefix: str = "/static"
    repository_backend: str = "memory"
    sqlite_path: str = "backend/dev.sqlite3"
    async_jobs_enabled: bool = True
    cleanup_generated_assets_on_new_story: bool = True
    image_generator_backend: str = "mock"
    diffusion_model_family: str = "sd15"
    diffusion_checkpoint_path: str = "model/Base_model/dreamshaper_8.safetensors"
    diffusion_lora_path: str | None = "model/Lora/Edward_Hopper-000001.safetensors"
    diffusion_lora_scale: float = 0.8
    diffusion_output_width: int = 768
    diffusion_output_height: int = 432
    diffusion_steps: int = 24
    diffusion_guidance_scale: float = 7.0
    diffusion_img2img_strength: float = 0.62
    diffusion_focus_crop_padding: float = 1.6
    diffusion_cpu_offload: bool = False
    diffusion_negative_prompt: str = (
        "text, watermark, logo, blurry, low quality, distorted, duplicate objects, cropped"
    )
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    @property
    def static_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "static"


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", "Interactive AI Scene Explorer"),
        app_env=os.getenv("APP_ENV", "development"),
        static_url_prefix=os.getenv("STATIC_URL_PREFIX", "/static"),
        repository_backend=os.getenv("REPOSITORY_BACKEND", "memory"),
        sqlite_path=os.getenv("SQLITE_PATH", "backend/dev.sqlite3"),
        async_jobs_enabled=os.getenv("ASYNC_JOBS_ENABLED", "true").lower()
        in {"1", "true", "yes"},
        cleanup_generated_assets_on_new_story=os.getenv(
            "CLEANUP_GENERATED_ASSETS_ON_NEW_STORY",
            "true",
        ).lower()
        in {"1", "true", "yes"},
        image_generator_backend=os.getenv("IMAGE_GENERATOR_BACKEND", "mock"),
        diffusion_model_family=os.getenv(
            "DIFFUSION_MODEL_FAMILY",
            os.getenv("SDXL_MODEL_FAMILY", "sd15"),
        ),
        diffusion_checkpoint_path=os.getenv(
            "DIFFUSION_CHECKPOINT_PATH",
            os.getenv("SDXL_CHECKPOINT_PATH", "model/Base_model/dreamshaper_8.safetensors"),
        ),
        diffusion_lora_path=os.getenv(
            "DIFFUSION_LORA_PATH",
            os.getenv("SDXL_LORA_PATH", "model/Lora/Edward_Hopper-000001.safetensors"),
        )
        or None,
        diffusion_lora_scale=float(
            os.getenv("DIFFUSION_LORA_SCALE", os.getenv("SDXL_LORA_SCALE", "0.8"))
        ),
        diffusion_output_width=int(
            os.getenv("DIFFUSION_OUTPUT_WIDTH", os.getenv("SDXL_OUTPUT_WIDTH", "768"))
        ),
        diffusion_output_height=int(
            os.getenv("DIFFUSION_OUTPUT_HEIGHT", os.getenv("SDXL_OUTPUT_HEIGHT", "432"))
        ),
        diffusion_steps=int(os.getenv("DIFFUSION_STEPS", os.getenv("SDXL_STEPS", "24"))),
        diffusion_guidance_scale=float(
            os.getenv("DIFFUSION_GUIDANCE_SCALE", os.getenv("SDXL_GUIDANCE_SCALE", "7.0"))
        ),
        diffusion_img2img_strength=float(os.getenv("DIFFUSION_IMG2IMG_STRENGTH", "0.62")),
        diffusion_focus_crop_padding=float(os.getenv("DIFFUSION_FOCUS_CROP_PADDING", "1.6")),
        diffusion_cpu_offload=os.getenv(
            "DIFFUSION_CPU_OFFLOAD",
            os.getenv("SDXL_CPU_OFFLOAD", "false"),
        ).lower()
        in {"1", "true", "yes"},
        diffusion_negative_prompt=os.getenv(
            "DIFFUSION_NEGATIVE_PROMPT",
            os.getenv(
                "SDXL_NEGATIVE_PROMPT",
                "text, watermark, logo, blurry, low quality, distorted, duplicate objects, cropped",
            ),
        ),
    )


settings = get_settings()
