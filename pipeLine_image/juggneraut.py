from diffusers import DiffusionPipeline
import torch

<<<<<<< HEAD
=======
# 1️⃣ Modell lokal herunterladen
local_dir = "/run/media/riccardodandrea/Ricca_Data/hub/Juggernaut-XL-v9"


# 2️⃣ Pipeline aus lokalem Ordner laden
>>>>>>> path
pipe = DiffusionPipeline.from_pretrained(
    "/run/media/riccardodandrea/Ricca_Data/hub/Juggernaut-XL-v9",
    torch_dtype=torch.float16
).to("cuda")

<<<<<<< HEAD
prompt = """
Dead people"""
=======
# 3️⃣ Bild generieren
prompt = """The image is a 3D render of a green dinosaur named Yoshi from the Mario series. Yoshi is standing on a brick street in a town and is holding a sign that says "Feed me please!" in capital white letters. Yoshi has a white belly, orange shoes, and a brown shell with orange spots. He is looking at the camera with a hopeful expression on his face. The background of the image is slightly blurred and shows a building with large windows behind Yoshi. The image is well-lit, and the colors are vibrant, <lora:dalle-3-xl-lora-v2:0.8>
"""
>>>>>>> path
image = pipe(prompt).images[0]
image.save("astronaut.png")

print("✅ Bild gespeichert: astronaut.png")
