import streamlit as st
import torch
from diffusers import DiffusionPipeline

st.set_page_config(page_title="Text-to-Image Generator",
                   page_icon="🎨",
                   layout="centered")

st.title("🎨 Text-to-Image mit Juggernaut-XL-v9")

welcome_page_tab1, Chat_tab2, info_for_models_tab3 = st.tabs([" :house: Home", 
                                                              " :statue_of_liberty: Text2ImageChat", 
                                                              " 🤗 hugging face models"])

with welcome_page_tab1:
    # Introduction on what is text 2 Image
    st.title("Welcome my name is")
    st.write("""
            :blue[Riccardo D'Andrea] Welcome to my text-to-image journey!!! 🚀

            a data scientist and deep learning enthusiast from Germany 😊. 
            I look forward to taking you on an exciting journey into the world 
            of machine learning.

            In this field, the unknown has often become tangible—I will help 
            you understand this even better and enable you to take action yourself. 💡

            Let's discover it together! 🤔 Will we create new image spaces 
            through text...?
            """)
    

    st.divider()
    st.subheader("What is Text 2 Image?")

    st.write("""
        Imagine that text-to-image models allow us to visually manifest our desires. 
        These models use techniques from machine learning research, 
        in particular computer vision and neural networks, to create complex, 
        high-resolution image descriptions from simple text.
            
        It works as follows: The input text is first encoded into a numerical format, 
        which is then fed into a neural network as an input data stream. 
        This network consists of several layers of neural units (neurons) that 
        derive more complex information about the text and ultimately generate 
        an image description.
            
        The image is generated through a process called “diffusion,” in which the 
        neural network slowly builds up an image piece by piece. This process is 
        divided into several steps: 
        1) Preliminary information is generated, which serves as the starting point 
        for the diffusion process.
        2) The information is then spread and updated across the neural network.
        3) The process is repeated until a stable image is produced.

        Text-to-image models have the ability to derive complex visual content 
        from simple text descriptions. 
        They can therefore be used as a tool to create innovative and creative 
        images!
            """)



    st.markdown("")
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
            
            # Mac File path explanation
            st.info(""" **Beispielpfade (Mac):**
            - `/Users/<USERNAME>/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1`
            - `/Volumes/ExternalDrive/models/Juggernaut-XL-v9`""")
            
            # Windows File path explanation
            st.info(""" **Beispielpfade (Windows):**
            - `C:\\Users\\<USERNAME>\\.cache\\huggingface\\hub\\models--stabilityai--stable-diffusion-2-1`
            - `D:\\models\\Juggernaut-XL-v9`""")
            
            # Linux File path explanation
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
                st.session_state.messages.append({"role": "assistant", 
                                                  "type": "text", 
                                                  "content": pipe})
                with chat_container:
                    with st.chat_message("assistant"):
                        st.write(pipe)

            elif pipe is not None:
                st.session_state.pipe = pipe
                msg = "✅ Modell erfolgreich geladen! Gib jetzt deinen Prompt ein ✨"
                st.session_state.messages.append({"role": "assistant", 
                                                  "type": "text", 
                                                  "content": msg})
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
                            st.session_state.messages.append({"role": "assistant", 
                                                              "type": "text", 
                                                              "content": err_msg})
                            st.error(err_msg)

with info_for_models_tab3:
    st.header("Which Model can you use 🤗? ")
    st.info(""" 
            Models for Text 2 Image [Huggingface](https://huggingface.co/models?pipeline_tag=text-to-image&sort=trending)!
            """)
    st.write("""
            **Introduction:**

            This tool helps you select a suitable text-to-image model.

            Each model was trained on different images, so your model selection 
            is crucial.

            While one model was trained on photorealistic image diffusion, 
            another was trained on cartoon-like images.

            HuggingFace models have a comprehensive description of the model's purpose. 
            However, there are other models that do not have a description, 
            in which case an internet search is always helpful.

            """)

    # Button to open Model Catalog
    st.markdown("[Find your Models](https://huggingface.co/models?pipeline_tag=text-to-image&sort=trending)")
    # Select text to image model

    models = ["Stable Diffusion", "DALL-E 2", "Midjourney"]
    selected_model = st.selectbox("Choose a Model:", models)

    if selected_model == 'Stable Diffusion':
        # Info about the chosen model
        st.write("""
                **Model Details:**

                * Purpose: Text-to-image tasks, such as generating photorealistic images from text descriptions.

                * Training Data: Photorealistic images

                Important Source: [Stable Diffusion](https://huggingface.co/docs/transformers/model_doc/stablediffusion) 
        """)

    elif selected_model == 'DALL-E 2':
        # Info about the chosen model
        st.write("""
                **Model Details:**

                * Purpose: Text-to-image tasks, such as generating photorealistic images from text descriptions.

                * Training Data: Cartoon-like images

                Important Source: [DALL-E 2](https://huggingface.co/ehristoforu/dalle-3-xl-v2) 
        """)

    elif selected_model == 'Midjourney':
        # Info about the chosen model
        st.write("""
                **Model Details:**

                    * Purpose: Text-to-image tasks, such as generating photorealistic images from text descriptions.

                    * Training Data: Photographic images

                Important Source: [Midjourney](https://huggingface.co/strangerzonehf/Flux-Midjourney-Mix2-LoRA) 
        """)