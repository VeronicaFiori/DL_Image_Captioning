import os
import json
import argparse
from typing import Dict, Any, List

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


JUDGE_SYSTEM = (
    "You are a strict judge for image caption faithfulness.\n"
    "Given an image and a caption, decide whether the caption is supported by the image.\n"
    "Return ONLY valid JSON with keys: faithful (true/false), score (1-5), hallucinations (list of strings), rationale (string).\n"
    "Faithful means: no invented objects/attributes/actions not visible.\n"
)

def try_parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if "{" in text and "}" in text:
        text = text[text.find("{"): text.rfind("}") + 1]
    return json.loads(text)

@torch.inference_mode()
def judge_one(model, processor, image: Image.Image, caption: str, device: str) -> Dict[str, Any]:
    user_prompt = (
        f"CAPTION: {caption}\n\n"
        "Evaluate faithfulness. If there are hallucinations, list them as short phrases.\n"
        "Output JSON only."
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    out_ids = model.generate(**inputs, max_new_tokens=256)
    out_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

    # prova a parsare JSON
    try:
        j = try_parse_json(out_text)
    except Exception:
        # fallback soft
        j = {
            "faithful": False,
            "score": 1,
            "hallucinations": ["parse_error"],
            "rationale": out_text[:500],
        }
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root (contains images/)")
    ap.add_argument("--preds", required=True, help="preds json (list of {image_id, caption})")
    ap.add_argument("--out", default="results/qwen_judgements.json")
    ap.add_argument("--model_id", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    with open(args.preds, "r", encoding="utf-8") as f:
        preds = json.load(f)

    n = len(preds) if args.limit == 0 else min(args.limit, len(preds))

    results: List[Dict[str, Any]] = []
    faithful_count = 0
    score_sum = 0.0

    for i in range(n):
        item = preds[i]
        img_id = item["image_id"]
        cap = item["caption"]

    
        file_name = item.get("file_name", None)
        if file_name is None:
            raise ValueError("Nel preds json manca 'file_name'. Modifica flickr8k_to_coco.py per salvarlo.")

        img_path = os.path.join(args.root, "images", file_name)
        image = Image.open(img_path).convert("RGB")

        j = judge_one(model, processor, image, cap, device)
        j["image_id"] = img_id
        j["file_name"] = file_name
        j["caption"] = cap
        results.append(j)

        if bool(j.get("faithful", False)):
            faithful_count += 1
        try:
            score_sum += float(j.get("score", 0))
        except Exception:
            pass

    summary = {
        "n": n,
        "faithful_rate": faithful_count / max(1, n),
        "avg_score": score_sum / max(1, n),
    }

    payload = {"summary": summary, "results": results}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Saved:", args.out)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
