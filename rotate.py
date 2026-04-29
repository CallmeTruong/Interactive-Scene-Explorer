import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Load model đúng cách
pipe = StableDiffusionPipeline.from_single_file(
    "model/Base_model/epiCRealism_Natural Sin RC1 VAE.safetensors",
    torch_dtype=torch.float16
).to("cuda")

# Load IP-Adapter
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter-plus_sd15.bin"
)

pipe.set_ip_adapter_scale(0.7)

# Load ảnh
image = Image.open("click_views/02_tight_crop.jpg").convert("RGB")

# Generate
result = pipe(
    prompt = """
same object, low_angle view,
different angle, new perspective,
""",
    ip_adapter_image=image,
    num_inference_steps=40,
    width=1024,
    height=576,
    guidance_scale=8.0
).images[0]

result.save("output.png")