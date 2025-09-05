import os
import streamlit as st
import torch
from diffusers import DiffusionPipeline


st.set_page_config(page_title="Text-to-Image Generator", page_icon="🎨", layout="centered")

st.title("🎨 Text-to-Image mit Juggernaut-XL-v9")

# Modell nur einmal laden
@st.cache_resource
def load_pipeline(model_path):
    pipe = DiffusionPipeline.from_pretrained(
        "/run/media/riccardodandrea/Ricca_Data/hub/Juggernaut-XL-v9",
        torch_dtype=torch.float16
    ).to("cuda")
    return pipe

pipe = load_pipeline()

# Prompt-Eingabe
prompt = st.text_input(
    label="🖊️ Schreibe deinen Prompt:",
    placeholder="z.B. Astronaut in a jungle, cold color palette, detailed, 8k"
)

if st.button("✨ Bild generieren"):

    if not prompt.strip():
        st.warning("Bitte gib einen Prompt ein!")
    else:
        with st.spinner("Generiere Bild... bitte warten ⏳"):
            image = pipe(prompt).images[0]

        st.image(image, caption=f"🖼️ Ergebnis für: {prompt}", use_container_width=True)
