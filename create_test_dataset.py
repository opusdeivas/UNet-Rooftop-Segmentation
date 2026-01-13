import os
import cv2
import numpy as np
import random

# ---------------- CONFIG ----------------
CROP_SIZE = 512
CROPS_PER_IMAGE = 50
TEST_TARGET = 1_000

# Source folders (original high-res data)
SOURCE_FOLDERS = {
    'test_images': r"C:\Users\PC\Desktop\Roof_segmentation\test\image",
    'test_labels': r"C:\Users\PC\Desktop\Roof_segmentation\test\label"
}

# Output folders (OUTSIDE of /resized to protect existing data)
OUTPUT_FOLDERS = {
    'test_images': r"C:\Users\PC\Desktop\Roof_segmentation\resized_test\test\image",
    'test_labels': r"C:\Users\PC\Desktop\Roof_segmentation\resized_test\test\label"
}


# ---------------- HELPERS ----------------
def grid_positions(h, w, crop):
    ys = list(range(0, h - crop, crop))
    xs = list(range(0, w - crop, crop))
    return [(y, x) for y in ys for x in xs]


def save_pair(img, lbl, out_img, out_lbl, name, y, x):
    cv2.imwrite(os.path.join(out_img, f"{name}_{y}_{x}.png"), img)
    cv2.imwrite(os.path.join(out_lbl, f"{name}_{y}_{x}.png"), lbl)


# ---------------- TEST WITH LABELS ----------------
def process_test_labeled(input_img_dir, input_lbl_dir,
                         output_img_dir, output_lbl_dir,
                         target_count):
    """
    Process test images WITH their corresponding labels.
    Stricter filtering:  skip ALL crops with roof_ratio < 0.01 (no random chance).
    """

    # Create output directories
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    total = 0

    for fname in os.listdir(input_lbl_dir):
        if not fname.lower().endswith(".tif") or "vis" in fname.lower():
            continue

        img_path = os.path.join(input_img_dir, fname)
        lbl_path = os.path.join(input_lbl_dir, fname)

        # Skip if image doesn't exist
        if not os.path.exists(img_path):
            print(f"Warning: Image not found for label {fname}, skipping...")
            continue

        img = cv2.imread(img_path)
        lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)

        if img is None or lbl is None:
            print(f"Warning: Could not read {fname}, skipping...")
            continue

        h, w = lbl.shape
        positions = grid_positions(h, w, CROP_SIZE)
        random.shuffle(positions)

        for y, x in positions[:CROPS_PER_IMAGE]:
            crop_img = img[y:y+CROP_SIZE, x:x+CROP_SIZE]
            crop_lbl = lbl[y:y+CROP_SIZE, x:x+CROP_SIZE]

            # Ensure label is uint8
            if crop_lbl.max() <= 1:
                crop_lbl = (crop_lbl * 255).astype(np.uint8)

            # Calculate roof ratio
            roof_ratio = np.count_nonzero(crop_lbl) / crop_lbl.size

            # STRICTER FILTERING: Skip ALL crops with <1% rooftop coverage
            # (no random chance like in train/val)
            if roof_ratio < 0.01:
                continue

            save_pair(
                crop_img,
                crop_lbl,
                output_img_dir,
                output_lbl_dir,
                os.path.splitext(fname)[0],
                y,
                x
            )

            total += 1
            print(f"\rProcessed:  {total}/{target_count}", end="")

            if total >= target_count: 
                print(f"\nReached target of {target_count} test samples.")
                return

    print(f"\nProcessing complete. Total test samples created: {total}")


# ---------------- RUN ----------------
if __name__ == "__main__": 
    print("Creating test dataset with ground truth labels...")
    print(f"Output directory: {OUTPUT_FOLDERS['test_images']. rsplit('test', 1)[0]}")
    print("-" * 50)

    process_test_labeled(
        SOURCE_FOLDERS['test_images'],
        SOURCE_FOLDERS['test_labels'],
        OUTPUT_FOLDERS['test_images'],
        OUTPUT_FOLDERS['test_labels'],
        TEST_TARGET
    )

    print("\nTest dataset creation complete!")
    print(f"Images saved to: {OUTPUT_FOLDERS['test_images']}")
    print(f"Labels saved to: {OUTPUT_FOLDERS['test_labels']}")