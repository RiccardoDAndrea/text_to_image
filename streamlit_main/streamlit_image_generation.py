import os
import torch
HF_HOME = "/run/media/riccardodandrea/Ricca_Data"
os.environ["HF_HOME"] = HF_HOME
from transformers import CLIPTokenizer
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
)

os.makedirs("Picture", exist_ok=True)

MODELS = [
    "Qwen/Qwen-Image",
    "stabilityai/sdxl-turbo",
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "dreamlike-art/dreamlike-photoreal-2.0",
]


# ----------------------------
# Funktionen mit Rückgaben
# ----------------------------

def shorten_prompt(prompt: str, max_tokens=77) -> str:
    """Kürzt Prompt auf max. Token-Anzahl für CLIP"""
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    tokens = tokenizer(prompt)["input_ids"]
    if len(tokens) > max_tokens:
        print(f"⚠️ Prompt zu lang ({len(tokens)} Tokens). Kürze auf {max_tokens}.")
        prompt = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
    return prompt


def load_pipeline(model_id: str):
    """Lädt Modell und gibt die Pipeline zurück"""
    try:
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
        return pipe
    except Exception as e:
        print(f"❌ Fehler beim Laden von {model_id}: {e}")
        return None


def generate_and_save_image(pipe, prompt: str, model_id: str) -> str:
    """Erzeugt ein Bild und speichert es – Rückgabe: Dateiname"""
    try:
        result = pipe(
            prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            height=512,
            width=512
        )
        image = result.images[0]
        model_name = model_id.replace("/", "_")
        filepath = f"Picture/{model_name}.png"
        image.save(filepath, quality=85)
        print(f"✅ Bild gespeichert: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Fehler beim Generieren von {model_id}: {e}")
        return ""


def process_model(model_id: str, prompt: str) -> bool:
    """Komplettprozess: Pipeline laden, Bild generieren, Speicher freigeben"""
    print(f"\n🔄 Verarbeite Modell: {model_id}")
    pipe = load_pipeline(model_id)
    if pipe is None:
        return False

    filepath = generate_and_save_image(pipe, prompt, model_id)
    del pipe
    torch.cuda.empty_cache()
    return bool(filepath)


def main() -> list:
    """Startet die Generierung für alle Modelle – Rückgabe: Liste generierter Dateien"""
    prompt = "A cat drawn in a Picasso look"
    prompt = shorten_prompt(prompt)
    saved_files = []

    for model_id in MODELS:
        success = process_model(model_id, prompt)
        if success:
            saved_files.append(model_id.replace("/", "_") + ".png")

    return saved_files


# ----------------------------
# Hauptprogramm
# ----------------------------
if __name__ == "__main__":
    files = main()
    print(f"\n📁 Fertig. {len(files)} Bild(er) generiert:")
    for f in files:
        print(" -", f)
