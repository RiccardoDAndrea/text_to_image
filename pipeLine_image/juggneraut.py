from huggingface_hub import snapshot_download
from diffusers import DiffusionPipeline
import torch

# 1️⃣ Modell lokal herunterladen
local_dir = "/run/media/riccardodandrea/Ricca_Data/Juggernaut-XL-v9"
snapshot_download(
    repo_id="RunDiffusion/Juggernaut-XL-v9",
    local_dir=local_dir,
    local_dir_use_symlinks=False  # stellt sicher, dass alles physisch kopiert wird
)

# 2️⃣ Pipeline aus lokalem Ordner laden
pipe = DiffusionPipeline.from_pretrained(
    local_dir,
    torch_dtype=torch.float16
).to("cuda")

# 3️⃣ Bild generieren
prompt = "Astronaut in a jungle, detailed, 8k"
image = pipe(prompt).images[0]

# 4️⃣ Bild speichern
image.save("astronaut.png")
print("✅ Bild gespeichert!")
