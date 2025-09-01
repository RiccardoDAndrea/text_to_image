import torch
from diffusers import StableDiffusionXLPipeline
import os
os.environ["HF_HOME"] = "/run/media/riccardodandrea/Ricca_Data"

pipe = StableDiffusionXLPipeline.from_pretrained(
    "RunDiffusion/Juggernaut-XL-v9",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    cache_dir="./models"  # nur der Oberordner!
)

pipe = pipe.to("cuda")  # GPU nutzen

# --- Prompt
prompt = " (Moonlit lighting:1.1) A close-up of a mysterious man's face, with soft shadows highlighting his sharp features, the light reflecting in his intense eyes, cool tones creating a calm yet enigmatic atmosphere"
negative_prompt = "fake eyes, deformed eyes, bad eyes, cgi, 3D, digital, airbrushed"

# --- Bild erzeugen
image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
    height=1024,
    width=1024
).images[0]

# --- Speichern
image.save("Picture/juggernaut_test.png")
print("✅ Bild gespeichert: juggernaut_test.png")
