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
    cocoEval.params["image_id"] = coco.getImgIds()
    cocoEval.evaluate()

    print("\n=== METRICS ===")
    for metric, score in cocoEval.eval.items():
        print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    main()
