from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    #"Qwen/Qwen-Image",
    torch_dtype=torch.float16).to("cuda")

init_image = Image.open("Picture/stabilityai_stable-diffusion-xl-base-1.0.png").convert("RGB").resize((512, 512))

prompt = "Angela Merkel with a sunset as a background, photorealistic, 8k"
negative_prompt = "blurry, low quality, deformed, bad anatomy"

result = pipe(
    prompt=prompt,
    image=init_image,
    strength=1.0,  # 0.0 = kaum Veränderung, 1.0 = stark verändert
    guidance_scale=7.5,
    negative_prompt=negative_prompt
)

result.images[0].save("output_img2img.png")
