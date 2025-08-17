import os
import torch
import streamlit as st
from diffusers import StableDiffusionXLPipeline

# Speicherpfad für HF-Modelle (anpassen!)

# Lade nur die SDXL Pipeline
@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    ).to("cuda")
    return pipe

pipe = load_pipeline()

# Streamlit UI
st.title("🎨 SDXL Image Generator")

prompt = st.text_input("Schreib deinen Prompt hier:")

if st.button("Bild generieren"):
    if prompt.strip():
        with st.spinner("Generiere Bild... ⏳"):
            image = pipe(prompt).images[0]
            st.image(image, caption="Generiertes Bild", use_container_width=True)
    else:
        st.warning("Bitte einen Prompt eingeben!")
