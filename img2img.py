import os
import json
import torch
from PIL import Image
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    EulerAncestralDiscreteScheduler,
)


# =========================================================
# CONFIG — chỉ đổi ở đây, không cần chạm vào logic bên dưới
# =========================================================

BASE_DIR = r"D:\interactive-scene-explorer"

SD_MODEL_PATH = os.path.join(
    BASE_DIR,
    r"model\Base_model\epiCRealism_Natural Sin RC1 VAE.safetensors"
)

VLM_JSON_PATH    = os.path.join(BASE_DIR, r"click_views\result.json")
TIGHT_IMAGE_PATH = os.path.join(BASE_DIR, r"click_views\02_tight_crop.jpg")
WIDE_IMAGE_PATH  = os.path.join(BASE_DIR, r"click_views\03_wide_crop.jpg")
OUTPUT_DIR       = os.path.join(BASE_DIR, "click_views")

# Chọn mode — không cần đổi gì khác
# faithful_zoom | plausible_view | creative_view
MODE = "plausible_view"

# Override góc nhìn nếu muốn thử khác với recommended_view từ VLM.
# Set None để dùng góc VLM đề xuất (khuyến nghị).
# Các giá trị hợp lệ: "front" | "left_3_4" | "right_3_4" | "side" | "isometric" | "top_3_4" | "low_angle"
VIEW_HINT_OVERRIDE = None

# Override strength nếu muốn kiểm soát thủ công.
# Set None để dùng giá trị VLM đề xuất (khuyến nghị).
STRENGTH_OVERRIDE = None   # ví dụ: 0.55

# LoRA
USE_LORA   = True
LORA_PATH  = os.path.join(
    BASE_DIR,
    r"model\Lora\【Worldwide Premiere】 Luban - Wooden Product Image_V1.0.safetensors"
)
LORA_SCALE = 0.65

WIDTH           = 1024
HEIGHT          = 576
SEED            = 12345
USE_CPU_OFFLOAD = False


# =========================================================
# VIEW HINT MAP
# =========================================================

VIEW_HINT_MAP = {
    "front":     "front view, straight-on camera, symmetrical composition",
    "left_3_4":  "left three-quarter view, 3D perspective showing depth",
    "right_3_4": "right three-quarter view, 3D perspective showing depth",
    "side":      "side profile view, lateral camera angle",
    "isometric": "isometric orthographic view, equal-angle perspective, no vanishing point distortion",
    "top_3_4":   "top three-quarter aerial view, looking slightly down",
    "low_angle": "dramatic low-angle upward shot, camera near ground level looking up",
}


# =========================================================
# LOAD & PARSE VLM JSON
# =========================================================

def load_vlm_result(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy file JSON: {path}\n"
            "Hãy chạy vlm_analyze.py trước."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_mode_config(data: dict, mode: str) -> dict:
    """
    Đọc toàn bộ config (prompt, negative, strength, steps, init_image)
    từ JSON VLM. Không hardcode gì thêm.
    """
    gen = data.get("generation_prompts", {})

    if mode not in gen:
        available = list(gen.keys())
        raise ValueError(
            f"Mode '{mode}' không có trong JSON. "
            f"Các mode hiện có: {available}"
        )

    m = gen[mode]

    # Resolve init_image key → actual path
    init_image_key  = m.get("init_image", "tight_crop")
    init_image_path = WIDE_IMAGE_PATH if init_image_key == "wide_crop" else TIGHT_IMAGE_PATH

    return {
        "prompt":              m.get("prompt", "").strip(),
        "negative_prompt":     m.get("negative_prompt", "").strip(),
        "strength":            float(m.get("strength", 0.55)),
        "guidance_scale":      float(m.get("guidance_scale", 7.5)),
        "num_inference_steps": int(m.get("num_inference_steps", 30)),
        "init_image_path":     init_image_path,
        "recommended_view":    m.get("recommended_view", ""),
        "output_name":         f"generated_{mode}.png",
    }


def apply_overrides(cfg: dict) -> dict:
    """Áp dụng override từ CONFIG nếu có."""
    if STRENGTH_OVERRIDE is not None:
        cfg["strength"] = float(STRENGTH_OVERRIDE)

    if VIEW_HINT_OVERRIDE is not None:
        view_text = VIEW_HINT_MAP.get(VIEW_HINT_OVERRIDE, VIEW_HINT_OVERRIDE)
        cfg["prompt"] = f"{cfg['prompt']}, {view_text}"
        cfg["recommended_view"] = VIEW_HINT_OVERRIDE + " (manual override)"

    return cfg


# =========================================================
# PIPELINE
# =========================================================

def load_pipe() -> StableDiffusionImg2ImgPipeline:
    if not os.path.exists(SD_MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy model: {SD_MODEL_PATH}")

    pipe = StableDiffusionImg2ImgPipeline.from_single_file(
        SD_MODEL_PATH,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        use_safetensors=True,
    )

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config
    )
    pipe.enable_attention_slicing()

    if USE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")

    if USE_LORA:
        if not os.path.exists(LORA_PATH):
            print(f"[WARNING] LoRA không tìm thấy, bỏ qua: {LORA_PATH}")
        else:
            pipe.load_lora_weights(LORA_PATH)
            pipe.fuse_lora(lora_scale=LORA_SCALE)
            print(f"LoRA loaded (scale={LORA_SCALE})")

    return pipe


def prepare_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy ảnh init: {path}")
    return Image.open(path).convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS)


def run_generation(pipe: StableDiffusionImg2ImgPipeline, cfg: dict) -> Image.Image:
    generator  = torch.Generator(device="cuda").manual_seed(SEED)
    init_image = prepare_image(cfg["init_image_path"])

    return pipe(
        prompt=cfg["prompt"],
        negative_prompt=cfg["negative_prompt"],
        image=init_image,
        strength=cfg["strength"],
        guidance_scale=cfg["guidance_scale"],
        num_inference_steps=cfg["num_inference_steps"],
        generator=generator,
    ).images[0]


# =========================================================
# DEBUG PRINT
# =========================================================

def print_summary(data: dict, cfg: dict):
    sep = "=" * 60

    target = data.get("target", {})
    scene  = data.get("scene_context", {})
    camera = data.get("camera_analysis", {})
    intent = data.get("user_intent", {})

    print(f"\n{sep}")
    print(f"  TARGET")
    print(f"{sep}")
    print(f"  Name        : {target.get('name', 'N/A')}")
    print(f"  Material    : {target.get('material', 'N/A')}")
    print(f"  Condition   : {target.get('condition', 'N/A')}")
    print(f"  Scale       : {target.get('approximate_scale', 'N/A')}")
    print(f"  Confidence  : {target.get('confidence', 'N/A')} "
          f"({'LOW' if target.get('low_confidence') else 'OK'})")

    print(f"\n{sep}")
    print(f"  SCENE")
    print(f"{sep}")
    print(f"  Lighting    : {scene.get('lighting', 'N/A')}")
    print(f"  Atmosphere  : {scene.get('atmosphere', 'N/A')}")
    print(f"  Art style   : {scene.get('art_style', 'N/A')}")

    print(f"\n{sep}")
    print(f"  CAMERA")
    print(f"{sep}")
    print(f"  Original    : {camera.get('original_angle', 'N/A')}")
    print(f"  Focal len   : {camera.get('focal_length_estimate', 'N/A')}")
    print(f"  Suggested   : {camera.get('suggested_angles', [])}")

    print(f"\n{sep}")
    print(f"  GENERATION CONFIG  (mode={MODE})")
    print(f"{sep}")
    print(f"  View        : {cfg['recommended_view']}")
    print(f"  Init image  : {cfg['init_image_path']}")
    print(f"  Strength    : {cfg['strength']}")
    print(f"  CFG scale   : {cfg['guidance_scale']}")
    print(f"  Steps       : {cfg['num_inference_steps']}")

    print(f"\n{sep}")
    print(f"  PROMPT")
    print(f"{sep}")
    print(cfg["prompt"])

    print(f"\n{sep}")
    print(f"  NEGATIVE PROMPT")
    print(f"{sep}")
    print(cfg["negative_prompt"])

    note = data.get("truthfulness_note", "")
    if note and note.lower() != "all details clearly visible":
        print(f"\n[VLM NOTE] {note}")

    print(f"\n{sep}")
    print(f"  USER INTENT : {intent.get('likely_intent', 'N/A')}")
    print(f"{sep}\n")


# =========================================================
# MAIN
# =========================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA không khả dụng. Kiểm tra PyTorch CUDA.")

    print("GPU  :", torch.cuda.get_device_name(0))
    print("MODE :", MODE)

    # 1. Load VLM result
    data = load_vlm_result(VLM_JSON_PATH)

    # 2. Build config từ JSON
    cfg = get_mode_config(data, MODE)

    # 3. Áp dụng override nếu có
    cfg = apply_overrides(cfg)

    # 4. In summary để kiểm tra trước khi generate
    print_summary(data, cfg)

    # 5. Load pipeline
    print("Loading SD pipeline...")
    pipe = load_pipe()

    # 6. Generate
    print("Generating...")
    result = run_generation(pipe, cfg)

    # 7. Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, cfg["output_name"])
    result.save(out_path)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()