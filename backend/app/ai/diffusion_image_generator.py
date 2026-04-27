from pathlib import Path
from threading import Lock

from backend.app.ai.base import GeneratedImage
from backend.app.core.config import settings
from backend.app.core.ids import new_id
from backend.app.schemas.click_target import ClickTarget


class DiffusionImageGenerator:
    """Generate scene images from a local SD 1.5 or SDXL checkpoint and optional LoRA."""

    _pipeline = None
    _lock = Lock()

    def generate_root(self, *, prompt: str, width: int, height: int) -> GeneratedImage:
        return self._generate(prompt=prompt, width=width, height=height)

    def generate_next(
        self,
        *,
        prompt: str,
        click_target: ClickTarget,
        width: int,
        height: int,
    ) -> GeneratedImage:
        focused_prompt = (
            f"{prompt}. Focus the composition on {click_target.label}. "
            "Keep the same visual world and LoRA style."
        )
        return self._generate(prompt=focused_prompt, width=width, height=height)

    def _generate(self, *, prompt: str, width: int, height: int) -> GeneratedImage:
        pipe = self._get_pipeline()
        output_width = settings.diffusion_output_width
        output_height = settings.diffusion_output_height

        image = pipe(
            prompt=prompt,
            negative_prompt=settings.diffusion_negative_prompt,
            width=output_width,
            height=output_height,
            num_inference_steps=settings.diffusion_steps,
            guidance_scale=settings.diffusion_guidance_scale,
        ).images[0]

        if (width, height) != (output_width, output_height):
            image = image.resize((width, height))

        filename = f"{new_id('scene')}.png"
        output_dir = settings.static_dir / "assets" / "scenes" / "generated"
        output_dir.mkdir(parents=True, exist_ok=True)
        image.save(output_dir / filename)

        return GeneratedImage(
            image_url=f"{settings.static_url_prefix}/assets/scenes/generated/{filename}",
            width=width,
            height=height,
        )

    def _get_pipeline(self):
        with self._lock:
            if self.__class__._pipeline is not None:
                return self.__class__._pipeline

            try:
                import torch
                from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
            except ImportError as exc:
                raise RuntimeError(
                    "Diffusion image generation requires diffusers and torch. "
                    "Install with: python -m pip install -e \".[sdxl]\""
                ) from exc

            checkpoint_path = Path(settings.diffusion_checkpoint_path)
            if not checkpoint_path.exists():
                raise RuntimeError(f"Diffusion checkpoint not found: {checkpoint_path}")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            pipeline_cls = self._pipeline_class(
                family=settings.diffusion_model_family,
                sd15_cls=StableDiffusionPipeline,
                sdxl_cls=StableDiffusionXLPipeline,
            )

            pipe = pipeline_cls.from_single_file(
                str(checkpoint_path),
                torch_dtype=dtype,
                use_safetensors=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )

            lora_path = Path(settings.diffusion_lora_path) if settings.diffusion_lora_path else None
            if lora_path is not None:
                if not lora_path.exists():
                    raise RuntimeError(f"Diffusion LoRA not found: {lora_path}")
                pipe.load_lora_weights(str(lora_path), adapter_name="style_lora")
                pipe.set_adapters(["style_lora"], adapter_weights=[settings.diffusion_lora_scale])

            pipe.enable_attention_slicing()
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()

            if device == "cuda" and settings.diffusion_cpu_offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(device)

            self.__class__._pipeline = pipe
            return pipe

    def _pipeline_class(self, *, family: str, sd15_cls, sdxl_cls):
        normalized = family.lower().strip()
        if normalized in {"sd15", "sd1.5", "stable-diffusion-v1"}:
            return sd15_cls
        if normalized == "sdxl":
            return sdxl_cls
        raise RuntimeError(f"Unsupported diffusion model family: {family}")
