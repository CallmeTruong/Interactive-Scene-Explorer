import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Patch CLIP trước khi import bất kỳ thứ gì khác
import transformers.models.clip.modeling_clip as _clip
class _PatchedCLIP(_clip.CLIPTextModel):
    @property
    def text_model(self):
        return self
_clip.CLIPTextModel = _PatchedCLIP

import time, json, torch
from PIL import Image
from transformers import CLIPTextModel
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from safetensors.torch import load_file

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH  = "model/Base_model/epiCRealism_Natural Sin RC1 VAE.safetensors"
LORA_PATH   = "model\Lora\IsoPixelV2-SD15.safetensors"
RESULT_JSON = "click_views/result.json"

INIT_IMAGE_MAP = {
    "wide_crop":  "click_views/03_wide_crop.jpg",
    "tight_crop": "click_views/02_tight_crop.jpg",
    "full":       "click_views/01_full_marked.jpg",
}

W, H = 1024, 576

def p(msg): print(msg, flush=True)

# ─────────────────────────────────────────────
# PROMPT TRUNCATION — CLIP hard limit 77 tokens
# ─────────────────────────────────────────────

def truncate_prompt(prompt, tokenizer, max_tokens=75):
    tokens = tokenizer.encode(prompt)
    if len(tokens) <= max_tokens:
        return prompt
    truncated = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
    p(f"[WARN] Prompt truncated: {len(tokens)} → 75 tokens")
    return truncated

# ─────────────────────────────────────────────
# LOAD PIPELINE — một lần duy nhất
# ─────────────────────────────────────────────

p("Loading SD pipeline...")
t0 = time.time()

text_encoder = CLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14",
    torch_dtype=torch.float16,
)

pipe = StableDiffusionImg2ImgPipeline.from_single_file(
    MODEL_PATH,
    text_encoder=text_encoder,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
).to("cuda")

pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

# LoRA — chỉ load unet weights
lora_state_dict = load_file(LORA_PATH)
unet_lora = {k: v for k, v in lora_state_dict.items() if "lora_te" not in k}
pipe.load_lora_weights(unet_lora, adapter_name="lora")

pipe.set_adapters(["lora"], adapter_weights=[0.75])

# Scheduler: DPM++ SDE Karras — chạy tốt với 15-20 steps
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
)

# IP-Adapter
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe.set_ip_adapter_scale(0.3)
pipe.image_encoder = pipe.image_encoder.to("cuda", dtype=torch.float16)

tokenizer = pipe.tokenizer
p(f"Pipeline ready in {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────
# LOAD RESULT
# ─────────────────────────────────────────────

with open(RESULT_JSON, encoding="utf-8") as f:
    data = json.load(f)

obj       = data["target"]
gen_prompts = data["generation_prompts"]

# Style anchor: wide_crop — giữ màu sắc và style gốc qua IP-Adapter
style_anchor = Image.open(INIT_IMAGE_MAP["wide_crop"]).convert("RGB").resize((W, H))

p(f"\nObject  : {obj['name']}")
p(f"Material: {obj['material']}")
p(f"Generating {len(gen_prompts)} views...\n")

# ─────────────────────────────────────────────
# GENERATE — tuần tự, mỗi view một lần
# ─────────────────────────────────────────────

for view_name, cfg in gen_prompts.items():
    p(f"=== {view_name} ===")

    # Fix: thêm space giữa prefix và prompt
    raw_prompt = f"best quality, {cfg['prompt']}"
    prompt          = truncate_prompt(raw_prompt,          tokenizer)
    negative_prompt = truncate_prompt(cfg["negative_prompt"], tokenizer)

    # Hard cap strength theo từng loại view
    strength_cap = {"faithful_zoom": 0.6, "side_view": 0.7, "low_angle": 0.8}
    strength = max(cfg["strength"], strength_cap.get(view_name, 0.50))

    p(f"Strength: {strength} | Steps: {cfg['num_inference_steps']} | Guidance: {cfg['guidance_scale']}")
    p(f"Prompt: {prompt[:120]}...")

    init_image = Image.open(
        INIT_IMAGE_MAP.get(cfg["init_image"], INIT_IMAGE_MAP["wide_crop"])
    ).convert("RGB").resize((W, H))

    torch.cuda.empty_cache()

    t0 = time.time()
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            ip_adapter_image=style_anchor,
            strength=strength,
            guidance_scale=cfg["guidance_scale"],
            num_inference_steps=cfg["num_inference_steps"] + 30,
            width=W,
            height=H,
            generator=torch.Generator("cuda").manual_seed(42),  # reproducible
        ).images[0]

    out_path = f"output_{view_name}.png"
    result.save(out_path)
    p(f"Saved: {out_path} ({time.time()-t0:.1f}s)\n")

    torch.cuda.empty_cache()

p("Done! All views generated.")