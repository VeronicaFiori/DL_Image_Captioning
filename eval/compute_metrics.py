import argparse
import json
from pycocotools.coco import COCO

# scorers "safe"
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def coco_to_gts(coco: COCO):
    """
    Ritorna dict: {image_id: [ {"caption": "..."} , ... ]}
    dove image_id è l'id intero usato nel COCO json refs.
    """
    gts = {}
    img_ids = coco.getImgIds()
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        gts[img_id] = [{"caption": a["caption"]} for a in anns]
    return gts


def preds_to_res(preds_path: str):
    """
    preds json: list of {image_id:int, caption:str, ...}
    Ritorna dict: {image_id: [ {"caption": "..."} ]}
    """
    with open(preds_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    res = {}
    for p in preds:
        img_id = int(p["image_id"])
        res[img_id] = [{"caption": p["caption"]}]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, help="refs json in COCO format")
    ap.add_argument("--preds", required=True, help="preds json list of {image_id, caption}")
    ap.add_argument("--out", default="", help="optional: save metrics to json")
    args = ap.parse_args()

    coco = COCO(args.refs)
    gts = coco_to_gts(coco)
    res = preds_to_res(args.preds)

    # allinea: tieni solo image_id presenti in entrambi
    common_ids = sorted(set(gts.keys()) & set(res.keys()))
    gts = {k: gts[k] for k in common_ids}
    res = {k: res[k] for k in common_ids}

    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]

    results = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts, res)
        if isinstance(method, (list, tuple)):
            # Bleu ritorna 4 valori
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
        print("\nSaved metrics to:", args.out)


if __name__ == "__main__":
    main()

"""
import argparse
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def _keep_method(method, banned):
    # method può essere stringa ("CIDEr") o lista (["Bleu_1",...])
    if isinstance(method, (list, tuple)):
        return all(m not in banned for m in method)
    return method not in banned

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, help="refs json (coco format)")
    ap.add_argument("--preds", required=True, help="preds json (list of {image_id, caption})")
    ap.add_argument("--skip", default="SPICE,METEOR", help="comma-separated metrics to skip")
    args = ap.parse_args()

    skip_set = {m.strip() for m in args.skip.split(",") if m.strip()}

    coco = COCO(args.refs)
    cocoRes = coco.loadRes(args.preds)
    cocoEval = COCOEvalCap(coco, cocoRes)

    # 1) Filtra gli scorer prima di valutare
    cocoEval.scorers = [
        (scorer, method)
        for (scorer, method) in cocoEval.scorers
        if _keep_method(method, skip_set)
    ]

    cocoEval.params["image_id"] = coco.getImgIds()

    # 2) Valuta (se qualcosa crasha comunque, non muore tutto)
    try:
        cocoEval.evaluate()
    except Exception as e:
        print("Evaluation error (some metric crashed):", e)
        # prova a stampare quello che è riuscito a calcolare
        if hasattr(cocoEval, "eval") and cocoEval.eval:
            pass
        else:
            raise

    print("\n=== METRICS ===")
    for metric, score in cocoEval.eval.items():
        try:
            print(f"{metric}: {float(score):.4f}")
        except Exception:
            print(f"{metric}: {score}")

if __name__ == "__main__":
    main()
"""

"""
import argparse
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, help="refs json (coco format)")
    ap.add_argument("--preds", required=True, help="preds json (list of {image_id, caption})")
    args = ap.parse_args()

    coco = COCO(args.refs)
    cocoRes = coco.loadRes(args.preds)


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