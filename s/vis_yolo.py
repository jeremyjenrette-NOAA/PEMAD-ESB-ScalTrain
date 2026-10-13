import os
import pandas as pd


def audit_and_filter_dataset(
    img_dir: str,
    normalized_csv_path: str,
    raw_csv_path: str,
    filtered_csv_output: str = "/Users/jeremy/Downloads/stardata_latest/groundtruth_star_normalized_existing.csv",
    unannotated_output: str = "/Users/jeremy/Downloads/stardata_latest/unannotated_images.txt",
):
    """1.

    Filters normalized CSV to keep only existing images on disk.
    2. Identifies images on disk that have 0 annotations in the raw CSV.
    """
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    # --- Step 0: Index files in image directory ---
    print(f"Scanning directory: {img_dir}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Directory not found: {img_dir}")

    images_on_disk = {
        f
        for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    }
    print(f"Total image files found on disk: {len(images_on_disk):,}")

    # --- Task 1: Filter Normalized CSV for Existing Images ---
    print("\n--- Task 1: Filtering Normalized CSV ---")
    df_norm = pd.read_csv(normalized_csv_path)
    total_norm_rows = len(df_norm)
    total_norm_imgs = df_norm["Imagename"].nunique()

    # Retain only rows where the image file actually exists on disk
    df_norm_exists = df_norm[df_norm["Imagename"].isin(images_on_disk)].copy()

    missing_imgs = set(df_norm["Imagename"]) - images_on_disk
    dropped_rows = total_norm_rows - len(df_norm_exists)

    # Export cleaned CSV
    df_norm_exists.to_csv(filtered_csv_output, index=False)

    print(
        f"Original normalized CSV: {total_norm_rows:,} rows across {total_norm_imgs:,} images"
    )
    print(
        f"Missing images on disk: {len(missing_imgs):,} ({dropped_rows:,} box detections dropped)"
    )
    print(
        f"Saved filtered CSV to '{filtered_csv_output}' ({len(df_norm_exists):,} rows across {df_norm_exists['Imagename'].nunique():,} existing images)"
    )

    # --- Task 2: Find Unannotated Images (using raw CSV) ---
    print("\n--- Task 2: Extracting Unannotated Image List ---")
    df_raw = pd.read_csv(raw_csv_path)
    annotated_imgs_raw = set(df_raw["Imagename"].dropna().unique())

    # Images on disk that do not appear anywhere in the raw CSV
    unannotated_imgs = sorted(list(images_on_disk - annotated_imgs_raw))

    # Save list to file
    with open(unannotated_output, "w") as f:
        for img in unannotated_imgs:
            f.write(f"{img}\n")

    print(
        f"Images on disk WITH annotations (raw): {len(images_on_disk & annotated_imgs_raw):,}"
    )
    print(
        f"Images on disk WITHOUT annotations (background images): {len(unannotated_imgs):,}"
    )
    print(f"Saved unannotated image list to '{unannotated_output}'")


if __name__ == "__main__":
    IMAGE_DIR = "/Volumes/PortableSSD/saltnoaa/images/stardata"
    NORMALIZED_CSV = "/Volumes/PortableSSD/saltnoaa/images/stardata/groundtruth_star_normalized.csv"
    RAW_CSV = "/Volumes/PortableSSD/saltnoaa/images/stardata/groundtruth_star_clean.csv"

    audit_and_filter_dataset(
        img_dir=IMAGE_DIR,
        normalized_csv_path=NORMALIZED_CSV,
        raw_csv_path=RAW_CSV,
    )