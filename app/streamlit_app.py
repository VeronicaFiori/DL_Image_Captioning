import os
import streamlit as st
from PIL import Image

from src.prompts import load_styles, build_style_prompt
from src.captioner_model import Blip2Captioner, CaptionConfig

st.set_page_config(page_title="BLIP2 Style Captioning", layout="centered")

st.title("BLIP-2 Image Captioning con Stile")
st.caption("Modello: BLIP-2 (Transformers) • Stili: prompt-based • Output: 1 caption")

@st.cache_resource
def load_captioner(model_id: str, max_new_tokens: int, num_beams: int, temperature: float):
    return Blip2Captioner(
        CaptionConfig(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
        )
    )

styles = load_styles()
style_names = list(styles.keys()) if isinstance(styles, dict) else ["factual", "romantic", "funny", "scientific"]

with st.sidebar:
    st.header("⚙️ Impostazioni")
    model_id = st.text_input("HuggingFace model_id", value="Salesforce/blip2-flan-t5-xl")
    max_new_tokens = st.slider("max_new_tokens", 10, 120, 40, 1)
    num_beams = st.slider("num_beams", 1, 8, 3, 1)
    temperature = st.slider("temperature", 0.1, 2.0, 1.0, 0.1)

    st.divider()
    style = st.selectbox("Stile", style_names, index=0)
    extra = st.text_input("Extra istruzioni (opzionale)", value="")

uploaded = st.file_uploader("Carica un'immagine", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input", use_container_width=True)

    instruction = build_style_prompt(style, styles)
    prompt = "Write ONE concise caption describing the image. Do not invent objects not visible. " + instruction
    if extra.strip():
        prompt = prompt + " " + extra.strip()

    if st.button(" Genera caption", type="primary"):
        with st.spinner("Genero..."):
            captioner = load_captioner(model_id, max_new_tokens, num_beams, temperature)
            cap = captioner.caption(image=image, user_prompt=prompt)
        st.subheader("Caption")
        st.write(cap)

else:
    st.info("Carica un'immagine per iniziare.")
