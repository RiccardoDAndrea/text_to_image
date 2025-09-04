import os 

import altair as alt
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

st.set_page_config(page_title="Text → Image Generator", page_icon="🎨", layout="centered")

st.title("🎨 Text → Image Generator")

# -----------------------
# Modell-Auswahl
# -----------------------
MODEL_OPTIONS = {
    "dreamlike-art/dreamlike-photoreal-2.0": "Dreamlike Photoreal 2.0 (fotorealistisch)",
    "UnfilteredAI/NSFW-gen-v2": "NSFW Gen v2 (unzensierte Inhalte)",
}

model_choice = st.selectbox("Wähle ein Modell:", options=list(MODEL_OPTIONS.keys()), format_func=lambda x: MODEL_OPTIONS[x])

# -----------------------
# Prompt Eingabe
# -----------------------
prompt = st.text_area("Gib deinen Prompt ein:", placeholder="z. B. A futuristic cityscape at sunset, ultra detailed")

negative_prompt = st.text_input("Optional: Negative Prompt (was ausgeschlossen werden soll)")

# -----------------------
# Pipeline laden (gecached)
# -----------------------
def load_pipeline(model_name: str):
    pipe = StableDiffusionPipeline.from_pretrained(model_name, 
                                                   torch_dtype=torch.float16,
                                                       cache_dir="/run/media/riccardodandrea/Ricca_Data"
)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
    return pipe

# -----------------------
# Bild generieren
# -----------------------
if st.button("Bild generieren"):
    if not prompt.strip():
        st.warning("Bitte zuerst einen Prompt eingeben!")
    else:
        with st.spinner(f"Generiere Bild mit {model_choice} ..."):
            pipe = load_pipeline(model_choice)
            image = pipe(prompt, negative_prompt=negative_prompt, guidance_scale=7.5).images[0]
            st.image(image, caption=f"Generiert mit {model_choice}", use_column_width=True)
