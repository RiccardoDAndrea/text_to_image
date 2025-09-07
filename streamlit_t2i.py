import streamlit as st
import torch
from diffusers import DiffusionPipeline

st.set_page_config(page_title="Text-to-Image Generator", 
                   page_icon="🎨", 
                   layout="centered")

st.title("🎨 Text-to-Image mit Juggernaut-XL-v9")

user_num_inference_steps = st.sidebar.slider(
    label="inference steps",
    min_value=1,
    max_value=50,
    value=5,
    step=1,
)

# Pipeline laden (nur Modell, kein num_inference_steps)
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


# Bisherige Nachrichten anzeigen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["type"] == "text":
            st.write(msg["content"])
        elif msg["type"] == "image":
            st.image(msg["content"], caption=msg.get("caption", ""), use_container_width=True)


# Eingabe vom User
user_input = st.chat_input("👉 Modellpfad zuerst eingeben, danach Prompts...")

if user_input:
    # Nachricht speichern & anzeigen
    st.session_state.messages.append({"role": "user", "type": "text", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Wenn noch kein Modell geladen ist → erster Input = Pfad
    if st.session_state.pipe is None:
        st.session_state.model_path = user_input
        with st.chat_message("assistant"):
            st.write("🔄 Lade Modell...")
        pipe = safe_load_pipeline(st.session_state.model_path)

        if isinstance(pipe, str):  # Fehler-Text zurück
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": pipe})
            with st.chat_message("assistant"):
                st.write(pipe)
        elif pipe is not None:
            st.session_state.pipe = pipe
            msg = "✅ Modell erfolgreich geladen! Gib jetzt deinen Prompt ein ✨"
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": msg})
            with st.chat_message("assistant"):
                st.write(msg)

    # Wenn Modell geladen ist → Prompt → Bild generieren
    else:
        with st.chat_message("assistant"):
            with st.spinner("Generiere Bild... bitte warten ⏳"):
                try:
                    image = st.session_state.pipe(user_input, num_inference_steps=user_num_inference_steps).images[0]
                    st.image(image, caption=f"🖼️ Ergebnis für: {user_input}", use_container_width=True)
                    # Nachricht im Verlauf speichern
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
