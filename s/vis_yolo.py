import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg


IMG_EXTS = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]


def load_class_names(classes_path: Path | None):
    """
    Optional: a text file with one class name per line (YOLO convention).
    If not provided or missing, we fall back to str(class_id).
    """
    if not classes_path:
        return None
    if not classes_path.exists():
        print(f"[WARN] classes file not found: {classes_path} (will label by class id)")
        return None
    names = []
    for line in classes_path.read_text().splitlines():
        line = line.strip()
        if line:
            names.append(line)
    return names if names else None


def find_image_for_stem(imgdir: Path, stem: str) -> Path | None:
    """
    YOLO label filename usually matches image stem.
    Search for any supported image extension.
    """
    for ext in IMG_EXTS:
        p = imgdir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def parse_yolo_label_file(label_path: Path):
    """
    Returns list of dicts: [{class_id, xc, yc, w, h}, ...] (all normalized floats)
    Skips malformed lines.
    """
    anns = []
    txt = label_path.read_text().strip()
    if not txt:
        return anns

    for i, line in enumerate(txt.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            print(f"[WARN] {label_path.name}:{i} malformed (expected 5 cols): {line}")
            continue
        try:
            class_id = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            print(f"[WARN] {label_path.name}:{i} parse error: {line}")
            continue
        anns.append({"class_id": class_id, "xc": xc, "yc": yc, "w": w, "h": h})
    return anns


def yolo_to_xywh_pixels(ann, img_w: int, img_h: int):
    """
    YOLO normalized center format -> top-left pixel + pixel width/height.
    """
    xc = ann["xc"] * img_w
    yc = ann["yc"] * img_h
    bw = ann["w"] * img_w
    bh = ann["h"] * img_h
    x = xc - bw / 2.0
    y = yc - bh / 2.0
    return x, y, bw, bh


def clamp_box(x, y, w, h, img_w, img_h):
    """
    Clamp to image bounds (handles tiny numeric drift outside [0,1]).
    """
    x2 = x + w
    y2 = y + h
    x = max(0, min(img_w - 1, x))
    y = max(0, min(img_h - 1, y))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    w = max(1, x2 - x)
    h = max(1, y2 - y)
    return x, y, w, h


def draw_one(image_path: Path, anns, class_names=None, out_path: Path | None = None, max_boxes=None):
    img = mpimg.imread(str(image_path))
    img_h, img_w = img.shape[0], img.shape[1]

    if max_boxes is not None:
        anns = anns[:max_boxes]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.axis("off")

    for ann in anns:
        x, y, w, h = yolo_to_xywh_pixels(ann, img_w, img_h)
        x, y, w, h = clamp_box(x, y, w, h, img_w, img_h)

        cid = ann["class_id"]
        label = class_names[cid] if (class_names and 0 <= cid < len(class_names)) else str(cid)

        rect = Rectangle((x, y), w, h, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, label, fontsize=10, verticalalignment="bottom")

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", required=True, help="Directory of YOLO .txt label files (e.g., data/labels/train)")
    ap.add_argument("--imgdir", required=True, help="Directory where images live (e.g., /Volumes/PortableSSD/saltnoaa/images/2022)")
    ap.add_argument("--outdir", default="data/processed/yolo_viz", help="Where to save overlays")
    ap.add_argument("--n", type=int, default=12, help="Number of random label files to visualize")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_boxes", type=int, default=None, help="Optional cap boxes drawn per image")
    ap.add_argument("--classes", default=None, help="Optional path to classes.txt (one class name per line)")
    ap.add_argument("--include_empty", action="store_true",
                    help="Also include images with missing/empty label files (draws no boxes if label is empty).")
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    imgdir = Path(args.imgdir)
    outdir = Path(args.outdir)

    if not labels_dir.exists():
        raise SystemExit(f"labels_dir not found: {labels_dir}")
    if not imgdir.exists():
        raise SystemExit(f"imgdir not found: {imgdir}")

    class_names = load_class_names(Path(args.classes)) if args.classes else None

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        raise SystemExit(f"No .txt label files found in: {labels_dir}")

    rnd = random.Random(args.seed)
    rnd.shuffle(label_files)

    chosen = []
    for lf in label_files:
        stem = lf.stem
        img_path = find_image_for_stem(imgdir, stem)
        if img_path is None:
            continue

        anns = parse_yolo_label_file(lf)
        if (not anns) and (not args.include_empty):
            continue

        chosen.append((lf, img_path, anns))
        if len(chosen) >= args.n:
            break

    if not chosen:
        msg = (
            "No matching images found (or all labels empty and --include_empty not set).\n"
            "Check that label stems match image basenames and that --imgdir is correct."
        )
        raise SystemExit(msg)

    # output structure: outdir/<train|val>/<stem>.png
    subset_name = labels_dir.name  # "train" or "val" if your folder is named that
    for lf, img_path, anns in chosen:
        out_path = outdir / subset_name / f"{img_path.stem}.png"
        draw_one(img_path, anns, class_names=class_names, out_path=out_path, max_boxes=args.max_boxes)

    print(f"Saved {len(chosen)} overlays to: {outdir / subset_name}")


if __name__ == "__main__":
    main()