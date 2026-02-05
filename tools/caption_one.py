
import argparse
from PIL import Image
import re
import torch
from src.prompts import load_styles, build_style_prompt
from src.captioner_model import Blip2Captioner, CaptionConfig
from src.style_rewriter import StyleRewriter, RewriteConfig

#Genera una caption su un'immagine
def build_prompt(style_text: str, extra: str = "") -> str:
    prompt = f"""
        Write ONE caption describing the image.

        Constraints:
        - Use only information visible in the image.
        - Do not invent objects, actions, text, logos, or counts.
        - Use exactly ONE sentence (max 20 words).

        Style requirement:
        {style_text}
        """.strip()

    if extra.strip():
        prompt += "\n\nExtra requirement:\n" + extra.strip()

    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to an image (jpg/png)")
    ap.add_argument("--style", default="factual", help="Style key from configs/caption_styles.yaml")
    ap.add_argument("--model_id", default="Salesforce/blip2-flan-t5-xl")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--num_beams", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    ap.add_argument("--dtype", default=None, help="float16 or float32 (default: auto)")
    ap.add_argument("--extra", default="", help="Extra instructions (optional)")
    ap.add_argument("--facts_first", action="store_true", help="Two-step safer captioning (anti-hallucination)")
    ap.add_argument("--style_from_base", action="store_true", help="Style transfer from base caption")

    args = ap.parse_args()

    # 1) stile
    styles = load_styles()
    style_text = build_style_prompt(args.style, styles)

    # 2) captioner (qui si settano i parametri)
    captioner = Blip2Captioner(
        CaptionConfig(
            model_id=args.model_id,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
            dtype=args.dtype,
        )
    )

    # 3) image
    image = Image.open(args.image).convert("RGB")

    #Genera caption base fattuale
    base_prompt = (
        "Write ONE factual caption describing the image. "
        "Use only what is visible. Do not invent objects. "
        "One sentence, max 20 words."
    )
    base = captioner.caption(image=image, user_prompt=base_prompt)

    #se stile non factual, riscrivi con stile
    if args.style != "factual":
        rewriter = StyleRewriter(RewriteConfig())
        cap = rewriter.rewrite(base_caption=base, style_text=style_text, extra=args.extra)
        prompt_used=base_prompt
    else:
        user_prompt = (
            "Write ONE caption describing the image.\n"
            "Constraints:\n"
            "- Use only information visible in the image.\n"
            "- Do not invent objects, actions, text, logos, or counts.\n"
            "- Use exactly ONE sentence (max 20 words).\n\n"
            f"Style requirement:\n{style_text}"
        )
        cap = captioner.caption(image=image, user_prompt=user_prompt)
        prompt_used = user_prompt

    

    print("\n========== PROMPT ==========")
    print(prompt_used)
    print("========== /PROMPT =========\n")

    print("========== CAPTION ==========")
    print(cap)
    print("========== /CAPTION =========\n")


if __name__ == "__main__":
    main()



