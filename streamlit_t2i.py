# "/run/media/riccardodandrea/Ricca_Data"

import streamlit as st
import os
from diffusers import AutoPipelineForText2Image, DEISMultistepScheduler
import torch

# SessionState für Button merken
if "change_path" not in st.session_state:
    st.session_state.change_path = False

if st.button("Do you want to change path?"):
    st.session_state.change_path = True

if st.session_state.change_path:
    Path_to_models = st.text_input(
        label="Enter your Dir where the models are stored",
        key="path_input",
        value="/run/media/riccardodandrea/Ricca_Data/UnfilteredAI/NSFW-gen-v2"  # optional Default
    )

    if Path_to_models:
        # Nur für HF_CACHE setzen, wenn du willst
        os.environ["HF_HOME"] = Path_to_models
        st.success(f"HF_HOME set to: {Path_to_models}")

        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                Path_to_models,   # <-- Lokaler Pfad statt Repo-Name
                torch_dtype=torch.float16,
                variant="fp16",
                local_files_only=True  # 🔒 verhindert Download-Versuche
            )
            pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to("cuda")
            st.success("Pipeline geladen ✅")
        except Exception as e:
            st.error(f"Konnte Modell nicht laden: {e}")
    else:
        st.warning("No dir entered")
