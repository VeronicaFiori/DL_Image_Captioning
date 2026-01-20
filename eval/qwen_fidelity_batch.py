import os
import json
import argparse
from PIL import Image
from tqdm import tqdm

from src.datasets.flickr8k_dataset import Flickr8kDataset
from src.prompts import load_styles, build_style_prompt
from src.captioner_lavis import LavisCaptioner
from src.evaluator_qwen2vl import QwenFidelityEvaluator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/flickr8k")
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--out", default="results/qwen_fidelity_test.json")
    ap.add_argument("--style", default="factual")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    styles = load_styles()
    instruction = build_style_prompt(args.style, styles)

    dataset = Flickr8kDataset(root=args.root, split=args.split)
    captioner = LavisCaptioner(model_name="instructblip")
    evaluator = QwenFidelityEvaluator(model_name="Qwen/Qwen2-VL-7B-Instruct")

    n = len(dataset) if args.limit == 0 else min(args.limit, len(dataset))

    rows = []
    for i in tqdm(range(n), desc="Qwen fidelity"):
        sample = dataset[i]
        img_path = os.path.join(args.root, "images", sample["image_id"])
        image = Image.open(img_path).convert("RGB")

        pred = captioner.generate(image, instruction=instruction)
        report = evaluator.evaluate(image, pred)

        rows.append({
            "image_id": sample["image_id"],
            "pred_caption": pred,
            "qwen_report": report,
            "gt_captions": sample["captions"]
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("Saved:", args.out)

if __name__ == "__main__":
    main()
