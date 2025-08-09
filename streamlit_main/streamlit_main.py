import os
import torch
import streamlit as st
from transformers import CLIPTokenizer

from streamlit_image_generation import (
    shorten_prompt,
    load_pipeline,
    generate_and_save_image,
)

# Setze HuggingFace-Cache-Verzeichnis
HF_HOME = "/run/media/riccardodandrea/Ricca_Data"
os.environ["HF_HOME"] = HF_HOME

# Liste verfügbarer Modelle
MODELS = [
    "stabilityai/sdxl-turbo",
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "dreamlike-art/dreamlike-photoreal-2.0",
]

# UI: Prompt-Eingabe
user_input = st.text_input("📝 Describe your image prompt:")
prompt = shorten_prompt(prompt=user_input, max_tokens=77)

# UI: Model-Auswahl
user_model_selection = st.multiselect(
    label="🧠 Choose model(s):", 
    options=MODELS
)

# UI: Knopf zum Generieren
if st.button("🎨 Generate Image(s)") and prompt and user_model_selection:
    for model_id in user_model_selection:
        with st.spinner(f"Loading model: {model_id}"):
            try:
                pipe = load_pipeline(model_id=model_id)

                image_path = generate_and_save_image(
                    pipe=pipe, 
                    prompt=prompt, 
                    model_id=model_id
                )

                st.image(image_path, caption=f"Model: {model_id}")

                # Speicher aufräumen
                del pipe
                torch.cuda.empty_cache()

            except Exception as e:
                st.error(f"❌ Error with model {model_id}: {str(e)}")
