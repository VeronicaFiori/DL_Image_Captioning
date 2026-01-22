import os
import argparse
import time
from PIL import Image

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from src.prompts import load_styles, build_style_prompt



def pick_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype(dtype_arg: str | None, device: str) -> torch.dtype:
    if dtype_arg:
        return getattr(torch, dtype_arg)
    return torch.float16 if device == "cuda" else torch.float32


@torch.inference_mode()
def generate_caption(
    model_id: str,
    image_path: str,
    user_prompt: str,
    max_new_tokens: int,
    num_beams: int,
    temperature: float,
    top_p: float,
    device: str,
    dtype: torch.dtype,
):
    print(f"[INFO] model_id = {model_id}")
    print(f"[INFO] device   = {device}")
    print(f"[INFO] dtype    = {dtype}")

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"[INFO] GPU: {props.name} | VRAM: {props.total_memory/1024**3:.2f} GB")
        torch.cuda.empty_cache()

    t0 = time.time()
    print("[INFO] Loading processor...")
    processor = Blip2Processor.from_pretrained(model_id)
    print(f"[INFO] Processor loaded in {time.time()-t0:.1f}s")

    t1 = time.time()
    print("[INFO] Loading model (this can take a while on first run)...")
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    print(f"[INFO] Model loaded in {time.time()-t1:.1f}s")

    print("[INFO] Loading image...")
    image = Image.open(image_path).convert("RGB")

    # prompt "Question/Answer" (come il tuo)
    prompt = f"Question: {user_prompt}\nAnswer:"

    print("[INFO] Preprocessing inputs...")
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )

    # sampling solo se beams=1 e temperature>0
    if num_beams == 1 and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)

    print("[INFO] Generating caption...")
    t2 = time.time()
    output_ids = model.generate(**inputs, **gen_kwargs)
    print(f"[INFO] Generation done in {time.time()-t2:.1f}s")

    text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    if text.lower().startswith("answer:"):
        text = text[len("answer:"):].strip()

    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to image")
    ap.add_argument("--style", default="factual", help="just a label (optional)")
    ap.add_argument("--model_id", default="Salesforce/blip2-flan-t5-xl")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--num_beams", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--device", default=None, help="cuda or cpu")
    ap.add_argument("--dtype", default=None, help="float16 or float32 (torch dtype name)")
    ap.add_argument("--extra", default="", help="extra instructions")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)

    # prompt “solido” anti-hallucination
   # user_prompt = (
   #     "Write ONE concise caption describing the image. "
   #     "Do not invent objects not visible. "
   #     f"Style: {args.style}. "
   # )
    
    styles = load_styles()
    style_text = build_style_prompt(args.style, styles)

    user_prompt = f"""
    Write ONE caption describing the image.

    Constraints:
    - Use only information visible in the image.
    - Do not invent objects, actions, text, logos, or counts.
    - Use exactly ONE sentence (max 20 words).

    Style requirement:
    {style_text}
    """.strip()

   # if args.extra.strip():
   #     user_prompt += args.extra.strip()

    print("\n========== PROMPT ==========")
    print(user_prompt)
    print("========== /PROMPT =========\n")

    cap = generate_caption(
        model_id=args.model_id,
        image_path=args.image,
        user_prompt=user_prompt,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
        dtype=dtype,
    )

    print("\n========== CAPTION ==========")
    print(cap)
    print("========== /CAPTION =========\n")


if __name__ == "__main__":
    main()
