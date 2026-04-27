# Interactive AI Scene Explorer Backend

This backend starts with a deterministic mock pipeline so the API contract,
click behavior, and transition package can be tested before wiring in local AI
models.

## Run

Install dependencies from `pyproject.toml`, then run:

```powershell
python -m uvicorn backend.app.main:app --reload
```

By default the backend uses the in-memory repository. To persist data locally:

```powershell
$env:REPOSITORY_BACKEND="sqlite"
$env:SQLITE_PATH="backend/dev.sqlite3"
python -m uvicorn backend.app.main:app --reload
```

## Test

The current tests use the standard library `unittest` runner:

```powershell
python -m unittest discover backend/tests
```

## MVP Flow

1. `POST /stories` creates a story, root scene, and mock hotspots.
2. `GET /stories/{story_id}/scenes` returns scene history for Back/Root/tree navigation.
3. `GET /scenes/{scene_id}` returns the scene and hotspots.
4. `POST /scenes/{scene_id}/click` resolves the click into a `ClickTarget`,
   creates a mock next scene, and returns a `zoom_crossfade` transition.
5. `POST /scenes/{scene_id}/prefetch` creates and caches the click result behind
   a completed mock job. A later matching click reuses that cached scene.
6. `POST /demo/reset` clears local repository records and generated image files
   for a fresh demo session.

## Repository Options

- `memory`: fastest for tests and throwaway demos.
- `sqlite`: persists stories, scenes, hotspots, jobs, and click cache in a local
  `.sqlite3` file.

## Real Image Generation

The default image generator is `mock`. To use the local SD 1.5 checkpoint and
LoRA:

```powershell
python -m pip install -e ".[sdxl]"

$env:IMAGE_GENERATOR_BACKEND="diffusion"
$env:CLEANUP_GENERATED_ASSETS_ON_NEW_STORY="true"
$env:DIFFUSION_MODEL_FAMILY="sd15"
$env:DIFFUSION_CHECKPOINT_PATH="model/Base_model/dreamshaper_8.safetensors"
$env:DIFFUSION_LORA_PATH="model/Lora/Edward_Hopper-000001.safetensors"
$env:DIFFUSION_LORA_SCALE="0.8"
$env:DIFFUSION_OUTPUT_WIDTH="768"
$env:DIFFUSION_OUTPUT_HEIGHT="432"
$env:DIFFUSION_STEPS="24"
$env:DIFFUSION_GUIDANCE_SCALE="7.0"
$env:DIFFUSION_IMG2IMG_STRENGTH="0.62"
$env:DIFFUSION_FOCUS_CROP_PADDING="1.6"
$env:DIFFUSION_CPU_OFFLOAD="false"

python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Generated images are saved under:

```text
backend/app/static/assets/scenes/generated/
```

When `CLEANUP_GENERATED_ASSETS_ON_NEW_STORY` is `true`, starting a new story
removes old generated image files from that directory.

The frontend contract is unchanged. Hotspots are still mock bboxes until
GroundingDINO is added.

Root scenes use text-to-image. Next scenes use image-to-image with a 16:9 crop
around the clicked target as the reference image, so the next scene explores the
selected object instead of repeating the whole root composition. Lowering
`DIFFUSION_IMG2IMG_STRENGTH` preserves more of that crop and raising it allows
larger changes. Raising `DIFFUSION_FOCUS_CROP_PADDING` includes more surrounding
context; lowering it makes the next scene tighter on the clicked target.

When `IMAGE_GENERATOR_BACKEND` is not `mock`, story creation and click handling
return `processing` responses with a `job_id`. The frontend polls
`GET /jobs/{job_id}` until the result is ready.

Diffusion generation is guarded by a single-process queue lock. Create-story
and click jobs wait for the GPU. Hover prefetch is opportunistic: if the GPU is
busy, it completes with a skipped result instead of starting another generation.

For SDXL checkpoints, set:

```powershell
$env:DIFFUSION_MODEL_FAMILY="sdxl"
$env:DIFFUSION_CPU_OFFLOAD="true"
```

SDXL needs substantially more RAM/VRAM than SD 1.5.
