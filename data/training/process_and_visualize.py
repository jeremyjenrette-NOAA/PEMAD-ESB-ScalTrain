#!/usr/bin/env python3

import argparse
import re
import math
import random
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw

def parse_geometry(geom_text):
    """
    Parses Habcam GEOMETRY_TEXT.
    Returns the bounding box (TLx, TLy, BRx, BRy) and the raw line coordinates (x1, y1, x2, y2).
    """
    if pd.isna(geom_text):
        return None, None, None, None, None
    
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(geom_text))
    if len(nums) >= 4:
        x1, y1, x2, y2 = map(float, nums[:4])
        
        diameter = math.hypot(x2 - x1, y2 - y1)
        radius = diameter / 2.0
        
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        
        return cx - radius, cy - radius, cx + radius, cy + radius, (x1, y1, x2, y2)
    
    return None, None, None, None, None

def generate_unified_csv(csv_path, out_csv):
    """Reads the raw CSV, calculates bboxes, and formats the unified spreadsheet."""
    print(f"📦 Loading annotations from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Rename standard columns to match the target structure
    if "IMAGE_NAME" in df.columns:
        df = df.rename(columns={"IMAGE_NAME": "image"})
    if "CLASS_NAME" in df.columns:
        df = df.rename(columns={"CLASS_NAME": "label"})
        
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["image"] = df["image"].astype(str).str.strip()
    
    # Filter out point annotations
    df = df[~df["GEOMETRY_TEXT"].astype(str).str.lower().str.contains("point", na=False)].copy()
    
    print("🧠 Calculating bounding boxes from line annotations...")
    parsed_data = df["GEOMETRY_TEXT"].apply(parse_geometry)
    
    df["TLx"] = [p[0] for p in parsed_data]
    df["TLy"] = [p[1] for p in parsed_data]
    df["BRx"] = [p[2] for p in parsed_data]
    df["BRy"] = [p[3] for p in parsed_data]
    
    # Drop rows where geometry parsing failed
    df = df.dropna(subset=["TLx", "TLy", "BRx", "BRy", "image"])
    
    # Round coordinates to 2 decimal places to match your example output
    df[["TLx", "TLy", "BRx", "BRy"]] = df[["TLx", "TLy", "BRx", "BRy"]].round(2)
    
    # Reorder columns to exactly match the requested target structure
    front_cols = ["image", "TLx", "TLy", "BRx", "BRy", "label"]
    remaining_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + remaining_cols]
    
    # Save the unified CSV
    df.to_csv(out_csv, index=False)
    print(f"💾 Saved {len(df)} unified annotations to → {out_csv}")
    
    return df

def visualize_annotations(df, img_dir, out_dir, n_samples):
    """Draws the bounding boxes and original line annotations onto a sample of images."""
    img_dir = Path(img_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter for valid images that actually exist in the target directory
    valid_images = [img for img in df["image"].unique() if (img_dir / img).exists()]
    
    if not valid_images:
        print(f"⚠️ No matching images found in {img_dir}. Skipping visualization.")
        return
        
    sample_size = min(n_samples, len(valid_images))
    sample_images = random.sample(valid_images, sample_size)
    
    print(f"\n🎨 Visualizing {sample_size} random images to {out_dir}/ ...")
    
    for fname in sample_images:
        img_path = img_dir / fname
        sub = df[df["image"] == fname]
        
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                draw = ImageDraw.Draw(img)
                
                for _, row in sub.iterrows():
                    # Extract bounding box coordinates
                    tlx, tly = row["TLx"], row["TLy"]
                    brx, bry = row["BRx"], row["BRy"]
                    
                    # Parse the raw line coordinates again just for drawing
                    _, _, _, _, line_coords = parse_geometry(row["GEOMETRY_TEXT"])
                    
                    # Draw the Bounding Box (Cyan, width 3)
                    draw.rectangle([tlx, tly, brx, bry], outline="cyan", width=3)
                    
                    # Draw the Original Line (Magenta, width 3)
                    if line_coords:
                        draw.line(line_coords, fill="magenta", width=3)
                        
                    # Draw the Label text above the box
                    label = str(row["label"])
                    draw.text((tlx, max(0, tly - 15)), label, fill="cyan")
                
                save_path = out_dir / f"viz_{fname}"
                img.save(save_path)
                
        except Exception as e:
            print(f"❌ Failed to process {fname}: {e}")
            
    print(f"✅ Visualization complete. Check the '{out_dir}' directory.")

def main():
    parser = argparse.ArgumentParser(description="Unify Habcam Annotations and Visualize")
    parser.add_argument("--csv", required=True, help="Path to the raw annotations CSV (e.g., 2017_annotations.csv)")
    parser.add_argument("--img_dir", required=True, help="Path to the source images (e.g., 2017tr_split)")
    parser.add_argument("--out_csv", required=True, help="Path to save the unified CSV")
    parser.add_argument("--viz_dir", default="viz_output", help="Directory to save the visualized images")
    parser.add_argument("--n", type=int, default=10, help="Number of random images to visualize")
    
    args = parser.parse_args()
    
    # Step 1 & 2: Generate the unified CSV
    unified_df = generate_unified_csv(args.csv, args.out_csv)
    
    # Step 3: Visualize the results
    visualize_annotations(unified_df, args.img_dir, args.viz_dir, args.n)

if __name__ == "__main__":
    main()
