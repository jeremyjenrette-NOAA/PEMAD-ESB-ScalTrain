#!/usr/bin/env python3

import argparse
import re
import math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import random

def parse_geometry_to_square(geom_text):
    """
    Parses Habcam GEOMETRY_TEXT.
    Converts the annotated line segment into a square bounding box 
    using the line as the diameter.
    """
    if pd.isna(geom_text):
        return None
    
    # Extract numbers from the string
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(geom_text))
    if len(nums) >= 4:
        x1, y1, x2, y2 = map(float, nums[:4])
        
        # Calculate diameter (length of line) and radius
        diameter = math.hypot(x2 - x1, y2 - y1)
        radius = diameter / 2.0
        
        # Calculate center point
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        
        # Build square bounding box
        tlx = cx - radius
        tly = cy - radius
        brx = cx + radius
        bry = cy + radius
        
        return {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "TLx": tlx, "TLy": tly, "BRx": brx, "BRy": bry
        }
    return None

def load_path_mapping(txt_file):
    """Reads the source text manifest into a dictionary of {filename: absolute_path}"""
    print(f"📄 Loading source paths from {txt_file}...")
    mapping = {}
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                filename = Path(line).name
                mapping[filename] = line
    return mapping

def main():
    parser = argparse.ArgumentParser(description="Visualize Habcam annotations (Line to Square BBox)")
    parser.add_argument("--ann_csv", required=True, help="Path to raw annotations CSV")
    parser.add_argument("--src_txt", required=True, help="Path to the source images manifest TXT")
    parser.add_argument("--out_pdf", required=True, help="Output PDF path")
    parser.add_argument("--n_images", type=int, default=20, help="Number of random images to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--labels", 
        nargs="+", 
        default=None, 
        help="Optional: List of specific class names to visualize. If omitted, visualizes ALL labels."
    )
    args = parser.parse_args()

    # 1. Load Data
    path_mapping = load_path_mapping(args.src_txt)
    
    print(f"📦 Loading annotations from {args.ann_csv}...")
    df = pd.read_csv(args.ann_csv)
    
    # Normalize columns
    if "IMAGE_NAME" in df.columns:
        df = df.rename(columns={"IMAGE_NAME": "image", "CLASS_NAME": "label"})
    
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # 🔥 NEW LOGIC: Filter ONLY if user provided specific labels
    if args.labels:
        target_labels = [l.lower() for l in args.labels]
        df = df[df["label"].isin(target_labels)].copy()
        print(f"🔍 Filtering to show only: {target_labels}")
    else:
        print("🔍 No specific labels requested. Showing ALL classes.")
    
    if df.empty:
        raise SystemExit("❌ No annotations found matching the requested criteria.")

    # 2. Apply Math Transformation
    print("🧠 Converting line annotations to square bounding boxes...")
    geom_data = df["GEOMETRY_TEXT"].apply(parse_geometry_to_square)
    geom_df = pd.DataFrame(geom_data.dropna().tolist())
    
    # Merge the parsed coordinates back into the main dataframe
    df = df.loc[geom_data.notna()].reset_index(drop=True)
    df = pd.concat([df, geom_df], axis=1)

    # 3. Filter for valid images
    available_images = [img for img in df["image"].unique() if img in path_mapping and Path(path_mapping[img]).exists()]
    
    if not available_images:
        raise SystemExit("❌ No images from the filtered CSV were found using the provided manifest.")

    # Sample images
    random.seed(args.seed)
    n_to_sample = min(args.n_images, len(available_images))
    sample_imgs = random.sample(available_images, n_to_sample)
    
    print(f"🎨 Generating validation PDF for {n_to_sample} images...")
    
    out_path = Path(args.out_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 4. Draw and Save to PDF
    with PdfPages(out_path) as pdf:
        for img_name in sample_imgs:
            src = path_mapping[img_name]
            
            # Load Image
            img = mpimg.imread(src)
            img_h, img_w = img.shape[0], img.shape[1]
            
            # Setup Plot
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(img)
            ax.axis("off")
            
            # Subset annotations for this image
            sub_df = df[df["image"] == img_name]
            
            for _, row in sub_df.iterrows():
                # Determine colors based on label exactness
                is_inexact = "inexact" in row["label"]
                line_color = "magenta" if is_inexact else "cyan"
                box_color = "orange" if is_inexact else "yellow"

                # Draw Original Line
                ax.plot([row["x1"], row["x2"]], [row["y1"], row["y2"]], 
                        color=line_color, linewidth=2.5, alpha=0.9)
                
                # 🔥 APPLY STRICT 4-WAY CLIPPING FOR VISUALIZATION
                tlx_clipped = max(0, min(img_w, row["TLx"]))
                tly_clipped = max(0, min(img_h, row["TLy"]))
                brx_clipped = max(0, min(img_w, row["BRx"]))
                bry_clipped = max(0, min(img_h, row["BRy"]))

                # Calculate new clipped width and height
                w = brx_clipped - tlx_clipped
                h = bry_clipped - tly_clipped
                
                # Draw Calculated Square Bounding Box (Clipped)
                rect = Rectangle((tlx_clipped, tly_clipped), w, h, 
                                 edgecolor=box_color, facecolor="none", linewidth=1.5)
                ax.add_patch(rect)
                
                # Add a tiny text label right above the clipped box
                ax.text(tlx_clipped, tly_clipped - 5, row["label"], 
                        color=box_color, fontsize=6, verticalalignment="bottom")

            # Updated title to reflect the more generalized label catching
            ax.set_title(
                f"{img_name}\nOriginal Stereo Image | {len(sub_df)} Annotations\n"
                f"STANDARD CLASSES: Line=Cyan, Box=Yellow | INEXACT SCALLOPS: Line=Magenta, Box=Orange\n"
                f"(Boxes clamped to frame edges)", 
                fontsize=10, pad=10
            )
            
            fig.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"✅ Visualizations saved to → {out_path}")

if __name__ == "__main__":
    main()