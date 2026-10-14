import os
import random
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# Color palette per class (BGR format for OpenCV)
CLASS_COLORS = {
    "Astropecten": (0, 255, 0),  # Bright Green
    "Asterias": (255, 105, 180),  # Pink/Magenta
    "Leptasterias": (0, 165, 255),  # Orange
    "Henricia": (255, 255, 0),  # Cyan
    "Anemone": (0, 255, 255),  # Yellow
}


def visualize_bounding_boxes(
    csv_path: str, img_dir: str, num_samples: int = 5, save_dir: str = None
):
    df = pd.read_csv(csv_path)
    unique_images = df["Imagename"].unique()

    # Select random sample of images
    sampled_images = random.sample(
        list(unique_images), min(num_samples, len(unique_images))
    )

    for img_name in sampled_images:
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Skipping missing image: {img_path}")
            continue

        img = cv2.imread(img_path)
        boxes = df[df["Imagename"] == img_name]

        for _, row in boxes.iterrows():
            tl_x, tl_y = int(row["TLx"]), int(row["TLy"])
            br_x, br_y = int(row["BRx"]), int(row["BRy"])
            cls_name = row["ClassName"]

            color = CLASS_COLORS.get(cls_name, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(img, (tl_x, tl_y), (br_x, br_y), color, thickness=3)

            # Draw label banner
            label = f"{cls_name}"
            (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                img, (tl_x, tl_y - h - 10), (tl_x + w + 6, tl_y), color, -1
            )
            cv2.putText(
                img,
                label,
                (tl_x + 3, tl_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        # Plot output (Convert BGR -> RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.title(f"Image: {img_name} ({len(boxes)} detection(s))")
        plt.axis("off")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"annotated_{img_name}")
            plt.savefig(out_path, bbox_inches="tight", dpi=150)
            plt.close()
            print(f"Saved visualization to: {out_path}")
        else:
            plt.show()


if __name__ == "__main__":
    IMAGE_DIRECTORY = "/Volumes/PortableSSD/saltnoaa/images/stardata"
    NORMALIZED_CSV = "/Volumes/PortableSSD/saltnoaa/images/stardata/groundtruth_star_normalized.csv"

    visualize_bounding_boxes(
        NORMALIZED_CSV, IMAGE_DIRECTORY, num_samples=3, save_dir="/Users/jeremy/Downloads/output_vis"
    )