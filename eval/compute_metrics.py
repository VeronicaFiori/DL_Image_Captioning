import argparse
import json
from pycocotools.coco import COCO
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def coco_refs_to_gts(refs_path: str):
    coco = COCO(refs_path)
    gts = {}
    img_ids = coco.getImgIds()

    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        # Lista di stringhe
        gts[img_id] = [a["caption"] for a in anns]

    return gts


def preds_to_res(preds_path: str):
    with open(preds_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    res = {}
    for p in preds:
        img_id = int(p["image_id"])
        # Lista di stringhe (1 caption per immagine)
        res[img_id] = [p["caption"]]

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    gts = coco_refs_to_gts(args.refs)
    res = preds_to_res(args.preds)

    # solo immagini comuni
    common = sorted(set(gts.keys()) & set(res.keys()))
    gts = {k: gts[k] for k in common}
    res = {k: res[k] for k in common}

    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        # NIENTE METEOR / SPICE
    ]

    results = {}

    for scorer, method in scorers:
        score, _ = scorer.compute_score(gts, res)

        if isinstance(method, (list, tuple)):
            for m, s in zip(method, score):
                results[m] = float(s)
        else:
            results[method] = float(score)

    print("\n=== METRICS (NO METEOR / NO SPICE) ===")
    for k in sorted(results.keys()):
        print(f"{k}: {results[k]:.4f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("Saved:", args.out)


if __name__ == "__main__":
    main()

