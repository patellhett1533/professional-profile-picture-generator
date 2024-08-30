from diffusers import StableDiffusionPipeline
import torch
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)


prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

image.save("generated_image.png")
