import streamlit as st
import torch
from diffusers import DiffusionPipeline
from streamlit_lottie import st_lottie

from utils import load_lottieurl
st.set_page_config(page_title="Text-to-Image Generator",
                   page_icon="🎨",
                   layout="centered")

st.title("🎨 Text-to-Image Locally")
creative_man = load_lottieurl('https://lottie.host/a2786f75-598c-457d-83b8-da7d5c45b91f/g06V88qWpk.json')
empty_chat_lottie = load_lottieurl('https://lottie.host/75cbdcf3-0356-4c8d-bfe3-5a278a262bb1/ShbsF9Vyu4.json')
welcome_page_tab1, Chat_tab2, info_for_models_tab3 = st.tabs([" :house: Home", 
                                                              " :statue_of_liberty: Text2ImageChat", 
                                                              " 🤗 hugging face models"])

with welcome_page_tab1:
    creative_man_lottie_tab, welcome_text_tab = st.columns(2)

    # Introduction on what is text 2 Image
    with creative_man_lottie_tab:
        st_lottie(creative_man)
    with welcome_text_tab:
        st.markdown("""
                    # Welcome! 🚀

                    Hi, my name is **Riccardo D'Andrea** – welcome to my **Text-to-Image Journey**! 😊  

                    I'm a **data scientist** and **deep learning enthusiast** from Germany 🇩🇪, and I'm excited to take you on an adventure into the world of **machine learning**.  

                    In this field, the unknown often becomes tangible 🔍 – I’ll help you understand it better and empower you to take action yourself 💡.  

                    Let's explore together! 🤔 Will we create new **image spaces** just from text…? 🌈✨
                    """)
    

    st.divider()

    st.markdown("""
            # What is Text 2 Image? 🎨✨

            Imagine text-to-image models are like magical potions 🧙‍♂️: you give them a few words, and *voilà* – they turn them into a picture! 🖼️💫 These models use clever tricks from **artificial intelligence**, especially **computer vision** and **neural networks**, to transform simple text into complex, high-resolution images.

            **Here’s how it works:**

            1. Your text is first turned into numbers 🔢 – a kind of “secret language” the computer understands.  
            2. These numbers are fed into a neural network 🧠, made up of layers of neurons. Each neuron thinks a little, mixes information, and wonders: “Hmm, what could this image look like?” 🤔  
            3. Now comes the fun part: **diffusion** 🌫️. The network builds the image step by step, like an artist painting layer by layer:  
            - First, a rough sketch is conjured ✏️  
            - Then details are spread, adjusted, and polished 🎨  
            - Step by step, until a stable, finished image emerges 🖼️✨

            The result? An image born from just a few words, almost like you poured your imagination into color! 🌈😄

            **Why it’s awesome:**  
            Text-to-image models can bring super creative ideas to life – whether you want a cartoon, a scenic landscape, or a wacky fantasy creature. They’re like tiny AI artists who never get tired 🎨🤖💖.
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
    if bool(user_input)==False:
        st_lottie(empty_chat_lottie,
                  width=600, height=300)
        st.info("Hey I am your Prime i am here to help you")
        st.info("Please give me your file path were you have saved your Models")
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
    st.divider()
    models = ["RunDiffusion/Juggernaut-XL-v9", "Lykon/dreamshaper-7", "UnfilteredAI/NSFW-gen-v2"]
    selected_model = st.selectbox("Choose a Model:", models)

    if selected_model == 'RunDiffusion/Juggernaut-XL-v9':
        st.write("""
                **Model Details:**

                * Purpose: Dein persönlicher Foto-Zauberer 🧙‍♂️📸 – erschafft fotorealistische Bilder von Landschaften, Menschen oder Objekten, als hätte er selbst eine Kamera in der Hand. Perfekt, wenn du realitätsnahe Szenen brauchst oder ein Bild generieren willst, das fast wie ein Foto aussieht.

                * Training Data: Hochwertige, fotorealistische Bilder aus allen möglichen Quellen. Alles wurde trainiert, damit das Modell Licht, Schatten und Perspektiven wie ein echter Fotograf versteht 🌄💡.

                * Besonderheit: Ideal für Projekte, bei denen die Realität täuschend echt aussehen soll – z. B. Produktbilder, Architekturszenen oder Portraits.  

                Important Source: [Juggernaut-XL-v9](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9) 🖼️
        """)

    elif selected_model == 'Lykon/dreamshaper-7':
        st.write("""
                **Model Details:**

                * Purpose: Der kreative Traum-Maler 🎨💭 – perfekt für farbenfrohe, fantasievolle Kunstwerke. Anders als Juggernaut-XL zaubert Dreamshaper-7 eher Cartoon-, Manga- oder Illustrations-Stil, weniger fotorealistisch, dafür voller Charakter und Stimmung.

                * Training Data: Cartoon- und Kunstbilder, Concept Art und Illustrationen. Die Bilder sind stilisiert und bunt – als würdest du durch ein Kaleidoskop schauen 🌈✨.  

                * Besonderheit: Ideal für Charakterdesigns, Storyboards, Comic-Kunst oder kreative Szenen, bei denen Realismus nicht das Ziel ist. Ein echtes Spielzeug für die Fantasie! 🧸🎭  

                Important Source: [Lykon/dreamshaper-7](https://huggingface.co/Lykon/dreamshaper-7) 🧩
        """)

    elif selected_model == 'UnfilteredAI/NSFW-gen-v2':
        st.write("""
                **Model Details:**

                * Purpose: Der rebellische Fotograf 📷🔥 – generiert realistische Bilder, oft mit erwachsenen Inhalten. Dieses Modell ist sehr leistungsstark, aber auch riskant, da es Inhalte erzeugen kann, die nicht jugendfrei oder rechtlich problematisch sein können.  

                * Training Data: Rohes, fotografisches Material aus verschiedenen Quellen. Das Modell hat keine strengen Filter, deshalb entstehen Inhalte, die verstörend oder heikel sein können ⚠️.  

                * Besonderheit: NSFW-gen-v2 eignet sich nur für verantwortungsbewusste Nutzung in sicheren, rechtlich zulässigen Umgebungen. Achtung: Es kann leicht zu problematischen Inhalten führen, also unbedingt Vorsicht walten lassen 🚨.  

                Important Source: [UnfilteredAI/NSFW-gen-v2](https://huggingface.co/UnfilteredAI/NSFW-gen-v2?not-for-all-audiences=true) 🚀
        """)

