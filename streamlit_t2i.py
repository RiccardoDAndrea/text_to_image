# "/run/media/riccardodandrea/Ricca_Data"

import streamlit as st
import os
from diffusers import AutoPipelineForText2Image, DEISMultistepScheduler
import torch
# Button, um den Pfad zu ändern
if "change_path" not in st.session_state:
    st.session_state.change_path = False

if st.button("Do you want to change path?"):
    st.session_state.change_path = True

if st.session_state.change_path:
    Path_to_models = st.text_input(
        label="Enter your Dir where the models are stored",
        key="path_input"
    )

    if Path_to_models:
        os.environ["HF_HOME"] = Path_to_models
        st.success(f"HF_HOME set to: {Path_to_models}")
    else:
        st.warning("No dir entered")

    pipe = AutoPipelineForText2Image.from_pretrained('UnfilteredAI/NSFW-gen-v2', torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")


