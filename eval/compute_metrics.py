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