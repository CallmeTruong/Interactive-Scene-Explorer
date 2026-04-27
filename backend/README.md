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
2. `GET /scenes/{scene_id}` returns the scene and hotspots.
3. `POST /scenes/{scene_id}/click` resolves the click into a `ClickTarget`,
   creates a mock next scene, and returns a `zoom_crossfade` transition.
4. `POST /scenes/{scene_id}/prefetch` creates and caches the click result behind
   a completed mock job. A later matching click reuses that cached scene.

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
$env:DIFFUSION_MODEL_FAMILY="sd15"
$env:DIFFUSION_CHECKPOINT_PATH="model/Base_model/dreamshaper_8.safetensors"
$env:DIFFUSION_LORA_PATH="model/Lora/Edward_Hopper-000001.safetensors"
$env:DIFFUSION_LORA_SCALE="0.8"
$env:DIFFUSION_OUTPUT_WIDTH="768"
$env:DIFFUSION_OUTPUT_HEIGHT="432"
$env:DIFFUSION_STEPS="24"
$env:DIFFUSION_GUIDANCE_SCALE="7.0"
$env:DIFFUSION_CPU_OFFLOAD="false"

python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Generated images are saved under:

```text
backend/app/static/assets/scenes/generated/
```

The frontend contract is unchanged. Hotspots are still mock bboxes until
GroundingDINO is added.

For SDXL checkpoints, set:

```powershell
$env:DIFFUSION_MODEL_FAMILY="sdxl"
$env:DIFFUSION_CPU_OFFLOAD="true"
```

SDXL needs substantially more RAM/VRAM than SD 1.5.
