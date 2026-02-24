import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import json

def load_coco(coco_json):
    with open(coco_json, "r") as f:
        coco = json.load(f)
    # index
    imgs = {im["id"]: im for im in coco["images"]}
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    return imgs, anns_by_img, cats

def draw_one(image_path, anns, cats, out_path=None, max_boxes=None):
    import matplotlib.image as mpimg

    img = mpimg.imread(image_path)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.axis("off")

    if max_boxes is not None:
        anns = anns[:max_boxes]

    for ann in anns:
        x, y, w, h = ann["bbox"]
        cat = cats.get(ann["category_id"], str(ann["category_id"]))

        rect = Rectangle((x, y), w, h, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, cat, fontsize=10, verticalalignment="bottom")

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="Path to COCO JSON (train.json or val.json)")
    ap.add_argument("--imgdir", required=True, help="Directory where images live (data/images)")
    ap.add_argument("--outdir", default="outputs/coco_viz", help="Where to save overlays")
    ap.add_argument("--n", type=int, default=12, help="Number of random images to visualize")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_boxes", type=int, default=None, help="Optional cap boxes drawn per image")
    args = ap.parse_args()

    coco_json = Path(args.coco)
    imgdir = Path(args.imgdir)
    outdir = Path(args.outdir)

    imgs, anns_by_img, cats = load_coco(coco_json)

    img_ids = list(imgs.keys())
    random.Random(args.seed).shuffle(img_ids)

    # choose only images that actually exist
    chosen = []
    for iid in img_ids:
        fname = imgs[iid]["file_name"]
        if (imgdir / fname).exists():
            chosen.append(iid)
        if len(chosen) >= args.n:
            break

    if not chosen:
        raise SystemExit("No images found. Check --imgdir and file_name values in COCO JSON.")

    for iid in chosen:
        im = imgs[iid]
        fname = im["file_name"]
        image_path = imgdir / fname
        anns = anns_by_img.get(iid, [])

        out_path = outdir / coco_json.stem / f"{Path(fname).stem}.png"
        draw_one(image_path, anns, cats, out_path=out_path, max_boxes=args.max_boxes)

    print(f"Saved {len(chosen)} overlays to: {outdir / coco_json.stem}")

if __name__ == "__main__":
    main()