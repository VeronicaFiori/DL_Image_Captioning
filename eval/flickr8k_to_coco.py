import os
import json
import argparse
from PIL import Image
from tqdm import tqdm

from src.prompts import load_styles, build_style_prompt
from src.captioner_lavis import LavisCaptioner
from src.datasets.flickr8k_dataset import Flickr8kDataset

def build_coco_refs(dataset: Flickr8kDataset):
    images = []
    annotations = []
    ann_id = 1

    for i in range(len(dataset)):
        sample = dataset[i]
        img_name = sample["image_id"]
        captions = sample["captions"]

        images.append({"id": i, "file_name": img_name})

        for cap in captions:
            annotations.append({
                "id": ann_id,
                "image_id": i,
                "caption": cap
            })
            ann_id += 1

    coco = {"images": images, "annotations": annotations}
    return coco

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/flickr8k", help="dataset root")
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--style", default="factual")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    styles = load_styles()
    instruction = build_style_prompt(args.style, styles)

    dataset = Flickr8kDataset(root=args.root, split=args.split, transform=None)
    refs = build_coco_refs(dataset)

    # salva refs
    refs_path = os.path.join(args.out_dir, f"refs_{args.split}.json")
    with open(refs_path, "w", encoding="utf-8") as f:
        json.dump(refs, f, ensure_ascii=False, indent=2)

    captioner = LavisCaptioner(model_name="instructblip")

    preds = []
    for i in tqdm(range(len(dataset)), desc=f"Generating preds ({args.split})"):
        sample = dataset[i]
        img_path = os.path.join(args.root, "images", sample["image_id"])
        image = Image.open(img_path).convert("RGB")

        cap = captioner.generate(
            image,
            instruction=instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        preds.append({
            "image_id": i,
            "caption": cap
        })

    preds_path = os.path.join(args.out_dir, f"preds_{args.split}.json")
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)

    print("Saved:")
    print(" -", refs_path)
    print(" -", preds_path)

if __name__ == "__main__":
    main()
