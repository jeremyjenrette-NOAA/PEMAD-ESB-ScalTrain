#!/usr/bin/env python3

import argparse
import re
import random
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw


def parse_and_shift_line(geom_text, offset):
    """Parses the original line segment and shifts the X coordinates based on the split."""
    if pd.isna(geom_text):
        return None
    
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(geom_text))
    if len(nums) >= 4:
        x1, y1, x2, y2 = map(float, nums[:4])
        return (x1 - offset, y1, x2 - offset, y2)
    return None

def main():
    parser = argparse.ArgumentParser(description="Visualize Bounding Boxes and Lines on Split Images")
    parser.add_argument("--csv", required=True, help="Path to the unified/split CSV")
    parser.add_argument("--img_dir", required=True, help="Path to the directory containing the split images")
    parser.add_argument("--out_dir", default="viz_output", help="Directory to save the visualized images")
    parser.add_argument("--side", choices=["left", "right"], required=True, help="Which side these images represent")
    parser.add_argument("--n", type=int, default=10, help="Number of random images to visualize")
    
    args = parser.parse_args()
    
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Loading annotations from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    # Filter out background images so we only visualize images with actual annotations
    df_anno = df[df["label"] != "background"].copy()
    unique_images = df_anno["image"].unique()
    
    if len(unique_images) == 0:
        print("❌ No annotated images found in the CSV.")
        return
        
    # Pick a random sample of images to visualize
    sample_size = min(args.n, len(unique_images))
    sample_images = random.sample(list(unique_images), sample_size)
    
    print(f"🎨 Visualizing {sample_size} random images to {out_dir}/ ...")
    
    for fname in sample_images:
        img_path = img_dir / fname
        if not img_path.exists():
            print(f"⚠️ Image missing from disk, skipping: {fname}")
            continue
            
        try:
            with Image.open(img_path) as img:
                # Convert to RGB to ensure colored drawing works (in case images are grayscale)
                img = img.convert("RGB")
                draw = ImageDraw.Draw(img)
                
                # If this is the right side, the offset is exactly the width of the newly split image
                offset = img.width if args.side == "right" else 0
                
                # Get all annotations for this specific image
                sub = df_anno[df_anno["image"] == fname]
                
                for _, row in sub.iterrows():
                    # 1. Draw the Square Bounding Box (Cyan, thick)
                    tlx, tly = row["TLx"], row["TLy"]
                    brx, bry = row["BRx"], row["BRy"]
                    draw.rectangle([tlx, tly, brx, bry], outline="cyan", width=3)
                    
                    # 2. Draw the Original Line Segment (Magenta, thick)
                    line_coords = parse_and_shift_line(row.get("GEOMETRY_TEXT", ""), offset)
                    if line_coords:
                        draw.line(line_coords, fill="magenta", width=3)
                        
                    # 3. Add Label Text
                    label = str(row["label"])
                    draw.text((tlx, max(0, tly - 15)), label, fill="cyan")
                
                save_path = out_dir / f"viz_{fname}"
                img.save(save_path)
                
        except Exception as e:
            print(f"❌ Failed to process {fname}: {e}")

    print(f"✅ Done! Check the '{out_dir}' directory for your visualized images.")

if __name__ == "__main__":
    main()