import streamlit as st
from src.utils_image import load_pil_image
from src.prompts import load_styles, build_style_prompt
from src.captioner_lavis import LavisCaptioner
from src.evaluator_qwen2vl import QwenFidelityEvaluator

st.set_page_config(page_title="Stylized Captioning (Flickr8k)", layout="wide")
st.title("Stylized Image Captioning (LAVIS) + Fidelity Eval (Qwen2-VL)")

styles = load_styles()

@st.cache_resource
def load_models():
    captioner = LavisCaptioner(model_name="instructblip")
    evaluator = QwenFidelityEvaluator(model_name="Qwen/Qwen2-VL-7B-Instruct")
    return captioner, evaluator

captioner, evaluator = load_models()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    img_file = st.file_uploader("Carica un'immagine", type=["png", "jpg", "jpeg"])
    style_key = st.selectbox("Stile", list(styles.keys()), index=0)

    max_new_tokens = st.slider("max_new_tokens", 10, 120, 40)
    temperature = st.slider("temperature", 0.0, 1.5, 0.7)
    top_p = st.slider("top_p", 0.1, 1.0, 0.9)

    run = st.button("Genera caption")

with col2:
    st.subheader("Output")
    if img_file and run:
        image = load_pil_image(img_file)
        st.image(image, caption="Immagine caricata", use_container_width=True)

        instruction = build_style_prompt(style_key, styles)
        caption = captioner.generate(
            image,
            instruction=instruction,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        st.markdown("### Caption")
        st.write(caption)

        st.markdown("### Qwen2-VL Fidelity Report")
        report = evaluator.evaluate(image, caption)
        st.write(report)
