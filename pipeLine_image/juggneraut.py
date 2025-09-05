from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "/run/media/riccardodandrea/Ricca_Data/hub/Juggernaut-XL-v9",
    torch_dtype=torch.float16
).to("cuda")

prompt = """
Dead people"""
image = pipe(prompt).images[0]
image.save("astronaut.png")

print("✅ Bild gespeichert: astronaut.png")
