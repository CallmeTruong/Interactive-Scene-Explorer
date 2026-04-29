import os
import json
import torch
from PIL import Image, ImageDraw
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info


# =========================
# CONFIG
# =========================

MODEL_PATH = r"model\vlm\Qwen2.5-VL-3B-Instruct"
IMAGE_PATH = r"output_match_shakker_no_hiresfix.png"
OUT_DIR    = r"D:\interactive-scene-explorer\click_views"

CLICK_X = 420
CLICK_Y = 200

MAX_IMAGE_SIZE  = 1024
MAX_NEW_TOKENS  = 2500   # tăng vì schema mới lớn hơn


# =========================
# UTILS
# =========================

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def resize_image_if_needed(image_path, max_size=1024):
    image = Image.open(image_path).convert("RGB")
    w, h  = image.size

    if max(w, h) <= max_size:
        return image_path

    scale = max_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = image.resize((new_w, new_h), Image.LANCZOS)

    root, ext    = os.path.splitext(image_path)
    resized_path = f"{root}_resized{ext}"
    resized.save(resized_path, quality=95)
    return resized_path


def make_click_views(image_path, x, y, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    image   = Image.open(image_path).convert("RGB")
    w, h    = image.size

    x = clamp(x, 0, w - 1)
    y = clamp(y, 0, h - 1)

    # --- 1. Full image with marker ---
    marked  = image.copy()
    draw    = ImageDraw.Draw(marked)
    radius  = max(14, int(min(w, h) * 0.018))
    lw      = max(3, int(radius * 0.22))

    draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline="red", width=lw)
    draw.line([x - radius, y, x + radius, y], fill="red", width=lw)
    draw.line([x, y - radius, x, y + radius], fill="red", width=lw)

    marked_path = os.path.join(out_dir, "01_full_marked.jpg")
    marked.save(marked_path, quality=95)

    # --- 2. Tight crop ---
    tight_size = max(224, int(min(w, h) * 0.28))
    tx1 = clamp(x - tight_size // 2, 0, w)
    ty1 = clamp(y - tight_size // 2, 0, h)
    tx2 = clamp(x + tight_size // 2, 0, w)
    ty2 = clamp(y + tight_size // 2, 0, h)

    tight_path = os.path.join(out_dir, "02_tight_crop.jpg")
    image.crop((tx1, ty1, tx2, ty2)).save(tight_path, quality=95)

    # --- 3. Wide crop ---
    wide_size = max(448, int(min(w, h) * 0.62))
    wx1 = clamp(x - wide_size // 2, 0, w)
    wy1 = clamp(y - wide_size // 2, 0, h)
    wx2 = clamp(x + wide_size // 2, 0, w)
    wy2 = clamp(y + wide_size // 2, 0, h)

    wide_path = os.path.join(out_dir, "03_wide_crop.jpg")
    image.crop((wx1, wy1, wx2, wy2)).save(wide_path, quality=95)

    return {
        "full_marked": marked_path,
        "tight_crop":  tight_path,
        "wide_crop":   wide_path,
        "image_size":  [w, h],
        "click":       [x, y],
    }


# =========================
# MODEL
# =========================

def load_model():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
        local_files_only=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        use_fast=False,
    )

    return model, processor


# =========================
# VLM PROMPT
# =========================

VLM_PROMPT = """
You are a click-to-explore vision system that analyzes where a user clicked in an image.

You receive 3 images:
- Image 1: full scene with a RED CIRCLE marker showing the click position
- Image 2: tight crop around the click point
- Image 3: wider crop showing surrounding context

CRITICAL RULES:
- The red circle/crosshair is a UI marker ONLY. It is NOT a real object in the scene.
- Analyze the object or region at the CENTER of the red circle, not the circle itself.
- If the target is unclear, describe the surrounding area and set "low_confidence": true.
- Never invent logos, brand names, text, or details you cannot clearly see.

Your output will be fed DIRECTLY into a Stable Diffusion img2img pipeline.
No additional prompt engineering will happen after you.
Your prompts must be COMPLETE, SPECIFIC, and READY TO USE as-is.

Return ONLY valid JSON. No markdown fences, no explanation outside the JSON.

{
  "target": {
    "name": "short descriptive name of the clicked object, e.g. 'ancient stone arch gate'",
    "description": "detailed visual description of the object itself, shape, form, structure",
    "material": "specific material(s), e.g. 'weathered limestone blocks with mossy mortar joints'",
    "color_palette": ["#hex1", "#hex2", "#hex3"],
    "texture_detail": "surface texture description ready for SD prompt, e.g. 'rough chiseled stone surface, deep shadow crevices, patina staining'",
    "approximate_scale": "e.g. 'monumental arch approximately 8 meters tall' or 'small decorative wall bracket'",
    "condition": "e.g. 'ancient, heavily eroded, partially restored' or 'modern, pristine, sharp edges'",
    "confidence": 0.0,
    "low_confidence": false
  },
  "scene_context": {
    "background": "describe what is behind and around the object in the scene",
    "nearby_elements": ["element1 with brief description", "element2"],
    "lighting": "full lighting description, e.g. 'warm golden hour sunlight from upper-left, long shadows'",
    "lighting_direction": "left",
    "atmosphere": "overall mood and atmosphere, e.g. 'hazy warm afternoon, dusty Mediterranean feel'",
    "art_style": "e.g. 'photorealistic architectural photography' or 'painterly digital render'",
    "render_keywords": "comma-separated SD style tokens ready to append to any prompt, e.g. 'hyperrealistic, 8k uhd, professional photography, sharp focus, high detail, cinematic'"
  },
  "camera_analysis": {
    "original_angle": "describe the current camera angle precisely, e.g. 'slight low-angle front elevation, eye level at base of arch'",
    "focal_length_estimate": "e.g. '28mm wide' or '50mm standard'",
    "suggested_angles": [
      "best alternative angle 1 as a complete camera description, e.g. 'left three-quarter view at 45 degrees, slight upward tilt showing full arch span'",
      "best alternative angle 2, e.g. 'dramatic low-angle looking up through arch opening toward sky'",
      "best alternative angle 3, e.g. 'isometric top-left view showing depth and side wall thickness'"
    ],
    "dof_style": "e.g. 'deep focus f/8, everything sharp from foreground to background' or 'moderate shallow dof, soft background bokeh'"
  },
  "generation_prompts": {
    "faithful_zoom": {
      "prompt": "WRITE A COMPLETE DETAILED SD PROMPT HERE. Must include in order: (1) object name and full physical description, (2) material and texture from target, (3) exact lighting from scene_context, (4) color palette description, (5) atmosphere, (6) same camera angle as original with closer framing, (7) render_keywords. Example: 'close-up detailed view of ancient stone arch gate, weathered limestone blocks with mossy mortar joints, rough chiseled stone surface with deep shadow crevices, warm golden hour sunlight from upper-left casting long shadows, muted earth tones of sandy beige and warm gray, hazy warm Mediterranean atmosphere, slight low-angle front elevation zoomed closer, hyperrealistic, 8k uhd, professional architectural photography, sharp focus'",
      "negative_prompt": "WRITE A COMPLETE NEGATIVE PROMPT. Must start with: 'red circle, red marker, crosshair, ui overlay, cursor, watermark, text, labels, annotation, logo, signature,' then add: 'blurry, out of focus, low quality, jpeg artifacts, deformed, bad geometry, distorted perspective, melted surfaces, impossible architecture,' then add any style-specific negatives relevant to this object",
      "strength": 0.30,
      "guidance_scale": 7.0,
      "num_inference_steps": 25,
      "init_image": "wide_crop",
      "recommended_view": "same framing as original, slightly closer"
    },
    "plausible_view": {
      "prompt": "WRITE A COMPLETE DETAILED SD PROMPT for a NEW CAMERA ANGLE. Must include: (1) same object with identical materials/textures/colors as faithful_zoom, (2) camera angle = your best pick from suggested_angles that best reveals the 3D form, (3) same lighting and atmosphere, (4) render_keywords. The object identity must remain completely faithful to the original.",
      "negative_prompt": "WRITE A COMPLETE NEGATIVE PROMPT. Same base as faithful_zoom negative. Add: 'flat front elevation only, exact same crop as original, boring zoom,'",
      "strength": 0.60,
      "guidance_scale": 8.0,
      "num_inference_steps": 36,
      "init_image": "tight_crop",
      "recommended_view": "pick the best angle from suggested_angles that shows 3D form clearly, write it here"
    },
    "creative_view": {
      "prompt": "WRITE A COMPLETE DETAILED SD PROMPT for a DRAMATIC CREATIVE REINTERPRETATION. Must include: (1) same object identity, (2) your most dramatic suggested_angle, (3) enhanced cinematic lighting (golden hour, dramatic shadows, volumetric light rays if appropriate), (4) enhanced atmosphere, (5) same material and texture, (6) render_keywords + 'cinematic, dramatic lighting, volumetric, epic'. No people, no fantasy elements, no unrecognizable changes to the object form.",
      "negative_prompt": "WRITE A COMPLETE NEGATIVE PROMPT. Same base as faithful_zoom negative. Add: 'person, human, face, portrait, figure, statue, fantasy, magic, unrealistic, kitsch, lowbrow, boring, flat lighting,'",
      "strength": 0.65,
      "guidance_scale": 8.5,
      "num_inference_steps": 42,
      "init_image": "tight_crop",
      "recommended_view": "pick the most dramatic angle from suggested_angles, write it here"
    }
  },
  "user_intent": {
    "likely_intent": "what the user probably wants to explore or understand about this object",
    "reason": "brief explanation of why you inferred this intent from the click position and object type"
  },
  "truthfulness_note": "list any details you were uncertain about or had to infer, or write 'all details clearly visible'"
}
""".strip()


def ask_click_intent(model, processor, full_marked, tight_crop, wide_crop):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": full_marked},
                {"type": "image", "image": tight_crop},
                {"type": "image", "image": wide_crop},
                {"type": "text",  "text": VLM_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]

    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


# =========================
# JSON PARSE
# =========================

def try_parse_json(text):
    text = text.strip()

    # strip markdown fences nếu có
    for fence in ["```json", "```"]:
        if text.startswith(fence):
            text = text[len(fence):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    try:
        return json.loads(text)
    except Exception:
        # thử tìm {} ngoài cùng
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass
        return None


# =========================
# MAIN
# =========================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA không khả dụng. Kiểm tra lại PyTorch CUDA.")

    print("CUDA:", torch.cuda.get_device_name(0))

    image_path = resize_image_if_needed(IMAGE_PATH, max_size=MAX_IMAGE_SIZE)

    views = make_click_views(
        image_path=image_path,
        x=CLICK_X,
        y=CLICK_Y,
        out_dir=OUT_DIR,
    )

    print("Đã tạo 3 ảnh đầu vào:")
    print("  1. Full marked :", views["full_marked"])
    print("  2. Tight crop  :", views["tight_crop"])
    print("  3. Wide crop   :", views["wide_crop"])

    print("\nĐang load model VLM...")
    model, processor = load_model()

    print("Đang phân tích ảnh...")
    result_text = ask_click_intent(
        model=model,
        processor=processor,
        full_marked=views["full_marked"],
        tight_crop=views["tight_crop"],
        wide_crop=views["wide_crop"],
    )

    print("\n===== RAW VLM OUTPUT =====")
    print(result_text)

    parsed = try_parse_json(result_text)

    if parsed is not None:
        output_json_path = os.path.join(OUT_DIR, "result.json")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)

        print("\n===== JSON ĐÃ LƯU =====")
        print(f"  {output_json_path}")

        # Preview nhanh
        target = parsed.get("target", {})
        camera = parsed.get("camera_analysis", {})
        print(f"\nTarget   : {target.get('name')}")
        print(f"Material : {target.get('material')}")
        print(f"Angle    : {camera.get('original_angle')}")
        print(f"Confident: {not target.get('low_confidence', False)}")

    else:
        output_text_path = os.path.join(OUT_DIR, "result_raw.txt")
        with open(output_text_path, "w", encoding="utf-8") as f:
            f.write(result_text)

        print("\n[WARNING] Không parse được JSON. Đã lưu raw text tại:")
        print(f"  {output_text_path}")
        print("Gợi ý: tăng MAX_NEW_TOKENS hoặc kiểm tra model output.")


if __name__ == "__main__":
    main()