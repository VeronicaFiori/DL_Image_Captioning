import argparse
from PIL import Image

from src.captioner_model import Blip2Captioner, CaptionConfig
from src.prompts import load_styles, build_style_prompt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--style", default="factual")
    ap.add_argument("--model_id", default="Salesforce/blip2-flan-t5-xl")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--num_beams", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    styles = load_styles()
    instruction = build_style_prompt(args.style, styles)

    prompt = (
        "Write ONE concise caption describing the image. "
        "Do not invent objects not visible. "
        + instruction
    )

    captioner = Blip2Captioner(
        CaptionConfig(
            model_id=args.model_id,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            temperature=args.temperature,
        )
    )

    image = Image.open(args.image).convert("RGB")
    cap = captioner.caption(image=image, user_prompt=prompt)

    print("PROMPT:", prompt)
    print("CAPTION:", cap)

if __name__ == "__main__":
    main()
