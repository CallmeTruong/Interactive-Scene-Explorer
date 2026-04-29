"""
config.py — Toàn bộ config tập trung ở đây.
Chỉnh file này, không cần đụng vào analyze.py hay generate.py.
"""

from __future__ import annotations
from typing import Literal

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR         = r"D:\interactive-scene-explorer"
IMAGE_PATH       = r"D:\interactive-scene-explorer\output_match_shakker_no_hiresfix.png"
OUTPUT_DIR       = r"D:\interactive-scene-explorer\click_views"

SD_MODEL_PATH    = rf"{BASE_DIR}\model\Base_model\epiCRealism_Natural Sin RC1 VAE.safetensors"
VLM_MODEL_PATH   = rf"{BASE_DIR}\model\vlm\Qwen2.5-VL-3B-Instruct"

# LoRA — đặt USE_LORA=False nếu không có
USE_LORA         = False
LORA_PATH        = rf"{BASE_DIR}\model\Lora\【Worldwide Premiere】 Luban - Wooden Product Image_V1.0.safetensors"
LORA_SCALE       = 0.20

# ── Click point ───────────────────────────────────────────────────────────────

CLICK_X: int = 420
CLICK_Y: int = 200

# ── Step 1 — Analysis ────────────────────────────────────────────────────────

FLORENCE_MODEL_ID = "microsoft/Florence-2-base"   # hoặc Florence-2-large
VLM_MAX_TOKENS    = 512     # đủ cho JSON ngắn gọn
MAX_IMAGE_SIZE    = 1024    # resize input nếu lớn hơn

# ── Step 2 — Generation ──────────────────────────────────────────────────────

# "faithful_zoom"  → refine detail, giữ góc gốc   (img2img + depth)
# "plausible_view" → góc nhìn mới hợp lý           (txt2img)
# "creative_view"  → góc dramatic, cinematic       (txt2img)
MODE: Literal["faithful_zoom", "plausible_view", "creative_view"] = "plausible_view"

# "sketch"    → ink + watercolor, background trắng sạch
# "realistic" → hyperrealistic photo, background neutral
# "concept"   → concept art / matte painting
OUTPUT_STYLE: Literal["sketch", "realistic", "concept"] = "sketch"

# Góc nhìn muốn generate — chỉ áp dụng cho plausible_view / creative_view
# "front" | "left_3_4" | "right_3_4" | "isometric"
# "low_angle" | "top_3_4" | "side" | "close_up" | "wide"
VIEW_HINT: str = "left_3_4"

WIDTH:  int = 1024
HEIGHT: int = 576
SEED:   int = 42
USE_CPU_OFFLOAD: bool = False

DEPTH_MODEL_ID      = "depth-anything/Depth-Anything-V2-Small-hf"
CONTROLNET_MODEL_ID = "lllyasviel/control_v11f1p_sd15_depth"