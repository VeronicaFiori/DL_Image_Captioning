import os
import json
import argparse
from PIL import Image
from tqdm import tqdm

from src.prompts import load_styles, build_style_prompt
from src.datasets.flickr8k_dataset import Flickr8kDataset
from src.captioner_model import Blip2Captioner, CaptionConfig


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
            annotations.append({"id": ann_id, "image_id": i, "caption": cap})
            ann_id += 1

    return {"images": images, "annotations": annotations}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/flickr8k", help="dataset root (contains images/ and captions/)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out_dir", default="results")

    ap.add_argument("--style", default="factual")
    ap.add_argument("--model_id", default="Salesforce/blip2-flan-t5-xl")  # <-- ESISTE
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--num_beams", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)

    ap.add_argument("--limit", type=int, default=0, help="0=all, otherwise only N samples")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # stile -> istruzione
    styles = load_styles()
    instruction = build_style_prompt(args.style, styles)

    # dataset
    dataset = Flickr8kDataset(root=args.root, split=args.split, transform=None)

    # refs COCO
    refs = build_coco_refs(dataset)
    refs_path = os.path.join(args.out_dir, f"refs_{args.split}.json")
    with open(refs_path, "w", encoding="utf-8") as f:
        json.dump(refs, f, ensure_ascii=False, indent=2)

    # captioner
    captioner = Blip2Captioner(
        CaptionConfig(
            model_id=args.model_id,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            temperature=args.temperature,
        )
    )

    preds = []
    n = len(dataset) if args.limit == 0 else min(args.limit, len(dataset))

    # prompt “solido” per captioning (1 frase)
    base_prompt = (
        "Write ONE concise caption describing the image. "
        "Do not invent objects not visible. "
        f"{instruction}"
    )

    for i in tqdm(range(n), desc=f"Generating preds ({args.split})"):
        sample = dataset[i]
        img_path = os.path.join(args.root, "images", sample["image_id"])
        image = Image.open(img_path).convert("RGB")

        cap = captioner.caption(image=image, user_prompt=base_prompt)

        preds.append({"image_id": i, "file_name": sample["image_id"], "caption": cap})

    preds_path = os.path.join(args.out_dir, f"preds_{args.split}.json")
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)

    print("Saved:")
    print(" -", refs_path)
    print(" -", preds_path)


if __name__ == "__main__":
    main()
