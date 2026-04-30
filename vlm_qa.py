import os, json, re, time, torch
from PIL import Image, ImageDraw
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

MODEL_PATH = r"D:\interactive-scene-explorer\model\vlm\Qwen2.5-VL-3B-Instruct"
IMAGE_PATH = r"output_low_angle.png"
OUT_DIR    = r"D:\interactive-scene-explorer\click_views"
CLICK_X    = 420
CLICK_Y    = 240

def p(msg): print(msg, flush=True)
def clamp(v, mn, mx): return max(mn, min(v, mx))

# ─────────────────────────────────────────────
# IMAGE PREP
# ─────────────────────────────────────────────

def resize_image_if_needed(image_path, max_size=768):
    image = Image.open(image_path).convert("RGB")
    w, h  = image.size
    if max(w, h) <= max_size:
        return image_path
    scale   = max_size / max(w, h)
    resized = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    root, ext = os.path.splitext(image_path)
    path = f"{root}_resized{ext}"
    resized.save(path, quality=90)
    return path

def make_click_views(image_path, x, y, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    w, h  = image.size
    x, y  = clamp(x, 0, w-1), clamp(y, 0, h-1)

    # View 1: full image với crosshair đỏ — VLM dùng để hiểu vị trí click
    marked = image.copy()
    draw   = ImageDraw.Draw(marked)
    r  = max(16, int(min(w, h) * 0.022))
    lw = max(3, int(r * 0.25))
    draw.ellipse([x-r, y-r, x+r, y+r], outline="red", width=lw)
    draw.line([x - r*2, y, x + r*2, y], fill="red", width=lw)
    draw.line([x, y - r*2, x, y + r*2], fill="red", width=lw)
    marked_path = os.path.join(out_dir, "01_full_marked.jpg")
    marked.save(marked_path, quality=90)

    # View 2: tight crop — object rõ nét, ít noise xung quanh
    tight_size = max(256, int(min(w, h) * 0.32))
    tight_path = os.path.join(out_dir, "02_tight_crop.jpg")
    image.crop((
        clamp(x - tight_size//2, 0, w), clamp(y - tight_size//2, 0, h),
        clamp(x + tight_size//2, 0, w), clamp(y + tight_size//2, 0, h),
    )).save(tight_path, quality=90)

    # View 3: wide crop — context để SD dùng làm init_image
    wide_size = max(480, int(min(w, h) * 0.60))
    wide_path = os.path.join(out_dir, "03_wide_crop.jpg")
    image.crop((
        clamp(x - wide_size//2, 0, w), clamp(y - wide_size//2, 0, h),
        clamp(x + wide_size//2, 0, w), clamp(y + wide_size//2, 0, h),
    )).save(wide_path, quality=90)

    return {
        "full_marked": marked_path,
        "tight_crop":  tight_path,
        "wide_crop":   wide_path,
        "image_size":  [w, h],
        "click":       [x, y],
    }

# ─────────────────────────────────────────────
# MODEL LOAD
# ─────────────────────────────────────────────

def load_model():
    if torch.cuda.is_available():
        props   = torch.cuda.get_device_properties(0)
        free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
        p(f"GPU: {props.name} | VRAM free: {free_gb:.1f} GB")

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=quant,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    p(f"Model loaded in {time.time()-t0:.1f}s")
    return model, processor

# ─────────────────────────────────────────────
# PROMPT — VLM chỉ nhận diện object, KHÔNG viết SD prompt
# Python sẽ build SD prompt từ kết quả này → giảm max_new_tokens 4000 → 200
# ─────────────────────────────────────────────

VLM_PROMPT = """The red crosshair marks the clicked point. Identify what object is under it.
Return ONLY this JSON, nothing else, no markdown:

{
  "name": "concise object name",
  "material": "specific material",
  "colors": ["#hex1", "#hex2", "#hex3"],
  "texture": "surface texture description for image generation",
  "condition": "physical state, e.g. polished, worn, weathered",
  "lighting": "current lighting in the image",
  "lighting_dir": "left|right|top|front|back",
  "style": "photorealistic|illustration|3d_render|painting",
  "confidence": 0.95
}"""

def ask_vlm(model, processor, full_marked, tight_crop):
    # Chỉ gửi 2 ảnh: full (để biết vị trí click) + tight (để nhìn rõ object)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": full_marked},
            {"type": "image", "image": tight_crop},
            {"type": "text",  "text": VLM_PROMPT},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to("cuda")

    t0 = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=220,      # JSON này chỉ cần ~150 tokens
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.05,
        )
    n_tokens = generated_ids.shape[-1] - inputs.input_ids.shape[-1]
    p(f"VLM inference: {time.time()-t0:.1f}s | {n_tokens} tokens generated")

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# ─────────────────────────────────────────────
# PARSE JSON
# ─────────────────────────────────────────────

def try_parse_json(text):
    text = re.sub(r'^```(?:json)?\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Tìm JSON object đầu tiên hợp lệ
    for m in re.finditer(r'\{', text):
        depth, s = 0, m.start()
        for i, ch in enumerate(text[s:]):
            depth += (ch == '{') - (ch == '}')
            if depth == 0:
                try:
                    return json.loads(text[s:s+i+1])
                except Exception:
                    break
    return None

# ─────────────────────────────────────────────
# BUILD SD PROMPTS từ VLM output — nhanh hơn, controllable hơn
# ─────────────────────────────────────────────

NEG_BASE = (
    "red circle, crosshair, marker, ui element, watermark, text, logo, "
    "blurry, deformed, distorted, lowres, artifacts, noise, overexposed"
)
NEG_STYLE = "different style, different era, different material, different color, inconsistent"
NEG_PEOPLE = "people, person, human, face, hands"

def build_sd_prompts(obj: dict) -> dict:
    name      = obj["name"]
    mat       = obj["material"]
    tex       = obj["texture"]
    col       = ", ".join(obj["colors"][:2])   # 2 màu chính
    cond      = obj["condition"]
    light     = obj["lighting"]
    ldir      = obj["lighting_dir"]
    style     = obj["style"]
    rk        = f"{style}, hyperrealistic, high detail, sharp focus, 8k"

    base_desc = f"{name}, {mat}, {tex}, {cond}, tones {col}"

    return {
        # Zoom vào chính object — strength thấp để giữ hình gốc
        "faithful_zoom": {
            "prompt": (
                f"close-up detail of {base_desc}, "
                f"{light}, light from {ldir}, "
                f"tighter framing same angle, macro detail, {rk}"
            ),
            "negative_prompt": f"{NEG_BASE}, {NEG_STYLE}",
            "strength": 0.28,
            "guidance_scale": 6.5,
            "num_inference_steps": 18,
            "init_image": "wide_crop",
        },
        # Góc nhìn 3/4 — thấy chiều sâu object
        "side_view": {
            "prompt": (
                f"{base_desc}, three-quarter view from the side, "
                f"reveals depth and side profile, "
                f"{light}, light from {ldir}, {rk}"
            ),
            "negative_prompt": f"{NEG_BASE}, {NEG_STYLE}, {NEG_PEOPLE}",
            "strength": 0.40,
            "guidance_scale": 7.0,
            "num_inference_steps": 20,
            "init_image": "tight_crop",
        },
        # Góc thấp ngước lên — thấy tỉ lệ và hình khối
        "low_angle": {
            "prompt": (
                f"{base_desc}, low angle shot looking up, "
                f"dramatic scale, cinematic composition, "
                f"dramatic {light}, volumetric light, {rk}"
            ),
            "negative_prompt": f"{NEG_BASE}, {NEG_STYLE}, {NEG_PEOPLE}, fantasy",
            "strength": 0.45,
            "guidance_scale": 7.5,
            "num_inference_steps": 22,
            "init_image": "tight_crop",
        },
    }

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    t_total = time.time()

    image_path = resize_image_if_needed(IMAGE_PATH)
    views = make_click_views(image_path, CLICK_X, CLICK_Y, OUT_DIR)
    p(f"Views created: {list(views.keys())}")

    torch.cuda.empty_cache()
    p("Loading VLM...")
    model, processor = load_model()

    p("Analysing click...")
    raw = ask_vlm(model, processor, views["full_marked"], views["tight_crop"])

    p(f"[RAW] {raw[:300]}")   # luôn in ra 300 ký tự đầu để debug dễ
    obj = try_parse_json(raw)

    if obj is None:
        raw_path = os.path.join(OUT_DIR, "result_raw.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw)
        p(f"[WARN] Parse failed — raw saved to {raw_path}")
        return

    p(f"Object  : {obj['name']}")
    p(f"Material: {obj['material']}")
    p(f"Confidence: {obj['confidence']}")

    # Build SD prompts bằng Python, không phụ thuộc VLM
    sd_prompts = build_sd_prompts(obj)

    result = {
        "target": obj,
        "views":  views,
        "generation_prompts": sd_prompts,
    }

    out_path = os.path.join(OUT_DIR, "result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    p(f"Saved: {out_path}")
    p(f"Total VLM time: {time.time()-t_total:.1f}s")
    p("Run rotate.py to generate images.")

if __name__ == "__main__":
    main()