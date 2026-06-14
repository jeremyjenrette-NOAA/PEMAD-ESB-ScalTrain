import os
import argparse
from PIL import Image
from pathlib import Path

def split_images(input_dir, side, output_dir=None):
    # Ensure input directory exists
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"Error: Directory '{input_dir}' does not exist.")
        return

    # If no output dir is specified, create one based on the input name and side
    if output_dir is None:
        output_dir = f"{input_path.name}_{side}_split"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all PNGs in the directory
    image_files = list(input_path.glob("*.png"))
    if not image_files:
        print(f"No .png files found in {input_dir}.")
        return

    print(f"Found {len(image_files)} images. Splitting the {side} side...")

    success_count = 0
    for img_file in image_files:
        try:
            with Image.open(img_file) as img:
                width, height = img.size
                midpoint = width // 2

                # Define the crop box: (left, upper, right, lower)
                if side.lower() == 'left':
                    crop_box = (0, 0, midpoint, height)
                elif side.lower() == 'right':
                    crop_box = (midpoint, 0, width, height)
                else:
                    print("Error: Side must be 'left' or 'right'.")
                    return

                # Crop and save
                cropped_img = img.crop(crop_box)
                save_path = output_path / img_file.name
                cropped_img.save(save_path)
                success_count += 1

        except Exception as e:
            print(f"Failed to process {img_file.name}: {e}")

    print(f"Success! Saved {success_count} images to '{output_path.resolve()}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split stereo images into left or right halves.")
    
    parser.add_argument("input_dir", type=str, help="Path to the folder containing the images.")
    parser.add_argument("side", type=str, choices=['left', 'right'], help="Which side to keep: 'left' or 'right'.")
    parser.add_argument("-o", "--output_dir", type=str, default=None, 
                        help="(Optional) Path to save the cropped images. Defaults to <input_dir>_<side>_split.")
    
    args = parser.parse_args()
    
    split_images(args.input_dir, args.side, args.output_dir)