from pathlib import Path
from threading import RLock
from urllib.parse import urlparse

from backend.app.ai.base import GeneratedImage
from backend.app.core.config import settings
from backend.app.core.ids import new_id
from backend.app.schemas.click_target import ClickTarget


class DiffusionImageGenerator:
    """Generate scene images from a local SD 1.5 or SDXL checkpoint and optional LoRA."""

    _text_pipeline = None
    _img2img_pipeline = None
    _lock = RLock()

    def generate_root(self, *, prompt: str, width: int, height: int) -> GeneratedImage:
        return self._generate_text2img(prompt=prompt, width=width, height=height)

    def generate_next(
        self,
        *,
        prompt: str,
        click_target: ClickTarget,
        current_image_url: str | None = None,
        width: int = 1600,
        height: int = 900,
    ) -> GeneratedImage:
        subject = self._subject_label(click_target)
        focused_prompt = (
            f"{prompt}. The clicked target is {subject}. Create a closer contextual "
            f"view of {subject} in the same place. Make {subject} larger and more "
            "detailed, but keep the surrounding architecture, materials, lighting, "
            "weather, color palette, and time of day consistent with the reference. "
            "Preserve nearby visible landmarks when possible. Do not invent a new "
            "location, unrelated background, or different scene."
        )
        reference_image = self._load_reference_image(current_image_url)
        if reference_image is not None:
            reference_image = self._focus_reference_image(
                image=reference_image,
                click_target=click_target,
            )
            return self._generate_img2img(
                prompt=focused_prompt,
                reference_image=reference_image,
                width=width,
                height=height,
            )

        return self._generate_text2img(prompt=focused_prompt, width=width, height=height)

    def _generate_text2img(self, *, prompt: str, width: int, height: int) -> GeneratedImage:
        pipe = self._get_text_pipeline()
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

        return self._save_image(image=image, width=width, height=height)

    def _generate_img2img(
        self,
        *,
        prompt: str,
        reference_image: object,
        width: int,
        height: int,
    ) -> GeneratedImage:
        pipe = self._get_img2img_pipeline()
        output_width = settings.diffusion_output_width
        output_height = settings.diffusion_output_height

        if reference_image.size != (output_width, output_height):
            reference_image = reference_image.resize((output_width, output_height))

        strength = self._img2img_strength()
        image = pipe(
            prompt=prompt,
            negative_prompt=settings.diffusion_negative_prompt,
            image=reference_image,
            strength=strength,
            num_inference_steps=self._img2img_steps(strength),
            guidance_scale=settings.diffusion_guidance_scale,
        ).images[0]

        return self._save_image(image=image, width=width, height=height)

    def _save_image(self, *, image: object, width: int, height: int) -> GeneratedImage:
        output_width = settings.diffusion_output_width
        output_height = settings.diffusion_output_height

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

    def _img2img_strength(self) -> float:
        return min(max(settings.diffusion_img2img_strength, 0.05), 1.0)

    def _img2img_steps(self, strength: float) -> int:
        minimum_steps = int(1 / strength) + 1
        return max(settings.diffusion_steps, minimum_steps)

    def _subject_label(self, click_target: ClickTarget) -> str:
        if click_target.label == "selected region":
            return "the selected central object or region"
        return click_target.label

    def _focus_reference_image(self, *, image: object, click_target: ClickTarget) -> object:
        crop_box = self._focus_crop_box(
            bbox=click_target.bbox,
            image_width=image.size[0],
            image_height=image.size[1],
        )
        return image.crop(crop_box)

    def _focus_crop_box(
        self,
        *,
        bbox: list[int],
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        x1 = min(max(x1, 0), image_width)
        x2 = min(max(x2, 0), image_width)
        y1 = min(max(y1, 0), image_height)
        y2 = min(max(y2, 0), image_height)

        bbox_width = max(1, x2 - x1)
        bbox_height = max(1, y2 - y1)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        target_aspect = settings.diffusion_output_width / settings.diffusion_output_height
        padding = max(1.0, settings.diffusion_focus_crop_padding)

        crop_width = bbox_width * padding
        crop_height = bbox_height * padding
        if crop_width / crop_height < target_aspect:
            crop_width = crop_height * target_aspect
        else:
            crop_height = crop_width / target_aspect

        crop_width = min(crop_width, image_width)
        crop_height = min(crop_height, image_height)
        left = min(max(center_x - crop_width / 2, 0), image_width - crop_width)
        top = min(max(center_y - crop_height / 2, 0), image_height - crop_height)
        right = left + crop_width
        bottom = top + crop_height

        return (
            int(round(left)),
            int(round(top)),
            int(round(right)),
            int(round(bottom)),
        )

    def _get_text_pipeline(self):
        with self._lock:
            if self.__class__._text_pipeline is not None:
                return self.__class__._text_pipeline

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

            self._load_lora(pipe)
            self._prepare_pipeline_for_runtime(pipe, device=device, allow_cpu_offload=True)

            self.__class__._text_pipeline = pipe
            return pipe

    def _get_img2img_pipeline(self):
        with self._lock:
            if self.__class__._img2img_pipeline is not None:
                return self.__class__._img2img_pipeline

            try:
                import torch
                from diffusers import (
                    StableDiffusionImg2ImgPipeline,
                    StableDiffusionXLImg2ImgPipeline,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Diffusion image-to-image generation requires diffusers and torch. "
                    "Install with: python -m pip install -e \".[sdxl]\""
                ) from exc

            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipeline_cls = self._pipeline_class(
                family=settings.diffusion_model_family,
                sd15_cls=StableDiffusionImg2ImgPipeline,
                sdxl_cls=StableDiffusionXLImg2ImgPipeline,
            )
            if settings.diffusion_cpu_offload:
                checkpoint_path = Path(settings.diffusion_checkpoint_path)
                if not checkpoint_path.exists():
                    raise RuntimeError(f"Diffusion checkpoint not found: {checkpoint_path}")

                dtype = torch.float16 if device == "cuda" else torch.float32
                pipe = pipeline_cls.from_single_file(
                    str(checkpoint_path),
                    torch_dtype=dtype,
                    use_safetensors=True,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                )
                self._load_lora(pipe)
                self._prepare_pipeline_for_runtime(pipe, device=device, allow_cpu_offload=True)
            else:
                text_pipe = self._get_text_pipeline()
                pipe = pipeline_cls(**text_pipe.components)
                self._prepare_pipeline_for_runtime(pipe, device=device, allow_cpu_offload=False)

            self.__class__._img2img_pipeline = pipe
            return pipe

    def _load_lora(self, pipe: object) -> None:
        lora_path = Path(settings.diffusion_lora_path) if settings.diffusion_lora_path else None
        if lora_path is None:
            return
        if not lora_path.exists():
            raise RuntimeError(f"Diffusion LoRA not found: {lora_path}")
        pipe.load_lora_weights(str(lora_path), adapter_name="style_lora")
        pipe.set_adapters(["style_lora"], adapter_weights=[settings.diffusion_lora_scale])

    def _prepare_pipeline_for_runtime(
        self,
        pipe: object,
        *,
        device: str,
        allow_cpu_offload: bool,
    ) -> None:
        pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

        if device == "cuda" and settings.diffusion_cpu_offload and allow_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

    def _load_reference_image(self, image_url: str | None) -> object | None:
        image_path = self._resolve_static_image_path(image_url)
        if image_path is None or not image_path.exists():
            return None

        try:
            from PIL import Image

            with Image.open(image_path) as image:
                return image.convert("RGB")
        except OSError:
            return None

    def _resolve_static_image_path(self, image_url: str | None) -> Path | None:
        if not image_url:
            return None

        image_path = urlparse(image_url).path
        static_prefix = settings.static_url_prefix.rstrip("/")
        if not image_path.startswith(f"{static_prefix}/"):
            return None

        relative_path = image_path[len(static_prefix) :].lstrip("/")
        candidate = (settings.static_dir / relative_path).resolve()
        static_root = settings.static_dir.resolve()
        if candidate == static_root or static_root in candidate.parents:
            return candidate
        return None

    def _pipeline_class(self, *, family: str, sd15_cls, sdxl_cls):
        normalized = family.lower().strip()
        if normalized in {"sd15", "sd1.5", "stable-diffusion-v1"}:
            return sd15_cls
        if normalized == "sdxl":
            return sdxl_cls
        raise RuntimeError(f"Unsupported diffusion model family: {family}")
