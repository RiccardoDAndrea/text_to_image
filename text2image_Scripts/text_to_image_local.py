# --- Set custom Hugging Face cache path before loading anything
import os
os.environ["HF_HOME"] = "/run/media/riccardodandrea/Ricca_Data"
# --- Imports
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
)
from transformers import CLIPTokenizer

# --- Modellliste (verschiedene Diffusion-Modelle)
models_ids = [
    "stabilityai/sdxl-turbo",                         # SDXL
    "runwayml/stable-diffusion-v1-5",                 # Klassisch
    "stabilityai/stable-diffusion-xl-base-1.0"
]

# --- Prompt & negativer Prompt
prompt = """
A stunning futuristic cityscape at sunset, seen from a high balcony. Neon lights reflect on wet streets, flying cars move between skyscrapers. A stylish woman in a red leather jacket and cyberpunk visor stands in the foreground, glowing holographic tattoos on her arm. The atmosphere is cinematic, ultra detailed, photorealistic yet artistic, masterpiece, award-winning photography, sharp focus, 8K, dramatic lighting, high dynamic range.

"""
negative_prompt = "blurry, cartoon, distorted, extra limbs, bad anatomy, watermark, text, logo, low detail, lowres"
# --- Kürze Prompt auf 77 Tokens
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
tokens = tokenizer(prompt)["input_ids"]

if len(tokens) > 77:
    print(f"⚠️ Prompt zu lang ({len(tokens)} Tokens). Kürze auf 77 Tokens.")
    prompt = tokenizer.decode(tokens[:77], skip_special_tokens=True)

# --- Bilder-Ordner anlegen
os.makedirs("Picture", exist_ok=True)

# --- Durch alle Modelle iterieren
for model_id in models_ids:
    print(f"\n🔄 Lade Modell: {model_id}")

    try:
        # Wähle richtige Pipeline basierend auf Modellart
        if "xl" in model_id or "sdxl" in model_id:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                variant="fp16"
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16
            )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_attention_slicing()
        pipe = pipe.to("cuda")

        # Bild generieren
        result = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            height=512,
            width=512
        )

        # Speichern
        image = result.images[0]
        model_name = model_id.replace("/", "_")
        image.save(f"Picture/{model_name}.png", quality=85)
        print(f"✅ Bild gespeichert: Picture/{model_name}.png")

        # Speicher freigeben
        del pipe
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ Fehler mit Modell {model_id}: {str(e)}")
