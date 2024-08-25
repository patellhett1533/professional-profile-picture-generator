import torch
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import os

prompt = 'professional headshot with a professional hairstyle and wearing a casual cloths with little smile'

prj_path = "username/repo_name"
model = os.environ.get('MODEL_NAME')
pipe = DiffusionPipeline.from_pretrained(
    model,
    torch_dtype=torch.float16,
)
# pipe.enable_model_cpu_offload()
pipe.to("cuda")
pipe.load_lora_weights(
    prj_path, weight_name="pytorch_lora_weights.safetensors")
pipe.load_lora_weights(
    prj_path, weight_name="pytorch_lora_weights.safetensors")

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
)
refiner.to("cuda")

prompt = "generate a professional headshot with a professional hairstyle and wearing a casual cloths with little smile"

seed = 42
generator = torch.Generator("cuda").manual_seed(seed)
image = pipe(prompt=prompt, generator=generator).images[0]
image = refiner(prompt=prompt, generator=generator, image=image).images[0]
image.save(f"generated_image.png")
