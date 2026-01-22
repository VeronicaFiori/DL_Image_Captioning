import argparse
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, help="refs json (coco format)")
    ap.add_argument("--preds", required=True, help="preds json (list of {image_id, caption})")
    args = ap.parse_args()

    coco = COCO(args.refs)
    cocoRes = coco.loadRes(args.preds)

    img_ids = coco.getImgIds()

    gts = {}
    res = {}

    for img_id in img_ids:
        gts[img_id] = coco.imgToAnns[img_id]
        res[img_id] = cocoRes.imgToAnns[img_id]

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]

    print("\n=== METRICS ===")
    for scorer, method in scorers:
        score, _ = scorer.compute_score(gts, res)

        if isinstance(method, list):
            for m, s in zip(method, score):
                print(f"{m}: {s:.4f}")
        else:
            print(f"{method}: {score:.4f}")

if __name__ == "__main__":
    main()

"""
    cocoEval = COCOEvalCap(coco, cocoRes)

    #try:
    #    cocoEval.evaluate()
    #except Exception as e:
    #    print("SPICE skipped:", e)
    cocoEval.scorers = [
        (scorer, method)
        for (scorer, method) in cocoEval.scorers
        if method not in ["SPICE", "METEOR"]
    ]

    cocoEval.params["image_id"] = coco.getImgIds()
    cocoEval.evaluate()

    print("\n=== METRICS ===")
    for metric, score in cocoEval.eval.items():
        print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    main()
"""