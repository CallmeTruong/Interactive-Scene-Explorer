import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline, DPMSolverSinglestepScheduler

BASE_MODEL = Path(r"D:\interactive-scene-explorer\model\Base_model\epiCRealism_Natural Sin RC1 VAE.safetensors")
LORA_DIR = Path(r"D:\interactive-scene-explorer\model\Lora")
LORA_NAME = "【Worldwide Premiere】 Luban - Wooden Product Image_V1.0.safetensors"
LORA_SCALE = 0.65

PROMPT = """
architectural concept art, hand-drawn ink sketch with light watercolor illustration,
loose expressive linework, single unified scene showing roman architectural complex,
colosseum arena, triumphal arch gate, aqueduct bridge, temple facade,
arranged in natural spatial depth, foreground midground background layers,
isometric perspective with vanishing point, cast shadows and ambient occlusion,
volumetric soft lighting,
soft muted earth tones, sepia ink lines, subtle pastel color,
professional architectural illustration, editorial sketch style
"""

NEGATIVE_PROMPT = """
flat diagram, technical blueprint, orthographic projection, no depth,
multiple disconnected views on white paper, infographic layout, callout lines,
dimension arrows, text labels, annotations, watermark, signature,
photorealistic render, 3d cgi, plastic, sterile, cold lighting,
crowded overlapping composition, blurry, low quality, distorted
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


pipe = StableDiffusionPipeline.from_single_file(
    str(BASE_MODEL),
    torch_dtype=dtype,
    safety_checker=None
).to(device)

# set đúng sampler gần với DPM++ SDE Karras
pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
    pipe.scheduler.config,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
)

pipe.load_lora_weights(
    str(LORA_DIR),
    weight_name=LORA_NAME,
    adapter_name="my_lora"
)

pipe.set_adapters("my_lora", adapter_weights=LORA_SCALE)

generator = torch.Generator(device=device)

image = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    num_inference_steps=30,   # match Shakker
    guidance_scale=7.0,       # match Shakker
    width=1024,
    height=576,
    generator=generator,
).images[0]

image.save("output_match_shakker_no_hiresfix.png")
print("Done: output_match_shakker_no_hiresfix.png")