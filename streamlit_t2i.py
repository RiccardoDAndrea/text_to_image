import streamlit as st
import torch
from diffusers import DiffusionPipeline

st.set_page_config(page_title="Text-to-Image Generator",
                   page_icon="🎨",
                   layout="centered")

st.title("🎨 Text-to-Image mit Juggernaut-XL-v9")

welcome_page_tab1, Chat_tab2, info_for_models_tab3 = st.tabs(["Welcome_Page", "Chat", "Owl"])

with welcome_page_tab1:
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

with Chat_tab2:
    # Sidebar Parameter
    user_num_inference_steps = st.sidebar.slider(
        label="inference steps",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
    )

    # Pipeline laden
    def load_pipeline(model_path: str):
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        )
        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()
        return pipe

    @st.cache_resource
    def safe_load_pipeline(model_path: str):
        try:
            return load_pipeline(model_path)
        except OSError as e:
            st.info(""" **Beispielpfade (Mac):**
            - `/Users/<USERNAME>/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1`
            - `/Volumes/ExternalDrive/models/Juggernaut-XL-v9`""")
            st.info(""" **Beispielpfade (Windows):**
            - `C:\\Users\\<USERNAME>\\.cache\\huggingface\\hub\\models--stabilityai--stable-diffusion-2-1`
            - `D:\\models\\Juggernaut-XL-v9`""")
            st.info(""" **Beispielpfade (Linux):**
            - `~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1`
            - `/run/media/<USERNAME>/hard_drive/hub/Juggernaut-XL-v9`
            - `/home/<USERNAME>/models/Juggernaut-XL-v9`""")
            return f"❌ Modell konnte nicht geladen werden.\n\nFehler: {e}"
        except EnvironmentError as e:
            return f"❌ Problem beim Laden oder Herunterladen des Modells.\n\nFehler: {e}"
        except ValueError as e:
            return f"❌ Ungültiger Modellpfad: {e}"
        return None

    # Session-State initialisieren
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipe" not in st.session_state:
        st.session_state.pipe = None
    if "model_path" not in st.session_state:
        st.session_state.model_path = None

    # --- Container für Chat-Verlauf ---
    chat_container = st.container()

    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["type"] == "text":
                    st.write(msg["content"])
                elif msg["type"] == "image":
                    st.image(msg["content"], caption=msg.get("caption", ""), use_container_width=True)

    # --- Eingabe ganz unten, außerhalb des Containers ---
    user_input = st.chat_input("👉 Modellpfad zuerst eingeben, danach Prompts...")

    if user_input:
        st.session_state.messages.append({"role": "user", "type": "text", "content": user_input})
        with chat_container:
            with st.chat_message("user"):
                st.write(user_input)

        if st.session_state.pipe is None:
            st.session_state.model_path = user_input
            with chat_container:
                with st.chat_message("assistant"):
                    st.write("🔄 Lade Modell...")
            pipe = safe_load_pipeline(st.session_state.model_path)

            if isinstance(pipe, str):
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": pipe})
                with chat_container:
                    with st.chat_message("assistant"):
                        st.write(pipe)
            elif pipe is not None:
                st.session_state.pipe = pipe
                msg = "✅ Modell erfolgreich geladen! Gib jetzt deinen Prompt ein ✨"
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": msg})
                with chat_container:
                    with st.chat_message("assistant"):
                        st.write(msg)

        else:
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Generiere Bild... bitte warten ⏳"):
                        try:
                            image = st.session_state.pipe(
                                user_input,
                                num_inference_steps=user_num_inference_steps
                            ).images[0]
                            st.image(image, caption=f"🖼️ Ergebnis für: {user_input}", use_container_width=True)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "type": "image",
                                "content": image,
                                "caption": f"🖼️ Ergebnis für: {user_input}"
                            })
                        except Exception as e:
                            err_msg = f"❌ Fehler bei der Bildgenerierung: {e}"
                            st.session_state.messages.append({"role": "assistant", "type": "text", "content": err_msg})
                            st.error(err_msg)

with info_for_models_tab3:
    st.header("An owl")
    st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
