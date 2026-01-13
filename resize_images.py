import os
import cv2
import numpy as np
import random
from PIL import Image

# ---------------- CONFIG ----------------
CROP_SIZE = 512
CROPS_PER_IMAGE = 50

TRAIN_TARGET = 10_000
VAL_TARGET = 1_000
TEST_TARGET = 1_000

# ---------------------------------------

folders = {
    'train_images': r"C:\Users\PC\Desktop\Roof_segmentation\train\image",
    'train_labels': r"C:\Users\PC\Desktop\Roof_segmentation\train\label",
    'val_images': r"C:\Users\PC\Desktop\Roof_segmentation\val\image",
    'val_labels': r"C:\Users\PC\Desktop\Roof_segmentation\val\label",
    'test_images': r"C:\Users\PC\Desktop\Roof_segmentation\test\image"
}

resized_folders = {
    'train_images': r"C:\Users\PC\Desktop\Roof_segmentation\resized\train\image",
    'train_labels': r"C:\Users\PC\Desktop\Roof_segmentation\resized\train\label",
    'val_images': r"C:\Users\PC\Desktop\Roof_segmentation\resized\val\image",
    'val_labels': r"C:\Users\PC\Desktop\Roof_segmentation\resized\val\label",
    'test_images': r"C:\Users\PC\Desktop\Roof_segmentation\resized\test\image"
}

for p in resized_folders.values():
    os.makedirs(p, exist_ok=True)


# ---------------- HELPERS ----------------
def grid_positions(h, w, crop):
    ys = list(range(0, h - crop, crop))
    xs = list(range(0, w - crop, crop))
    return [(y, x) for y in ys for x in xs]


def save_pair(img, lbl, out_img, out_lbl, name, y, x):
    cv2.imwrite(os.path.join(out_img, f"{name}_{y}_{x}.png"), img)
    cv2.imwrite(os.path.join(out_lbl, f"{name}_{y}_{x}.png"), lbl)


# ---------------- TRAIN / VAL ----------------
def process_labeled(input_img_dir, input_lbl_dir,
                    output_img_dir, output_lbl_dir,
                    target_count):

    total = 0

    for fname in os.listdir(input_lbl_dir):
        if not fname.lower().endswith(".tif") or "vis" in fname.lower():
            continue

        img = cv2.imread(os.path.join(input_img_dir, fname))
        lbl = cv2.imread(os.path.join(input_lbl_dir, fname), cv2.IMREAD_GRAYSCALE)

        h, w = lbl.shape
        positions = grid_positions(h, w, CROP_SIZE)
        random.shuffle(positions)

        for y, x in positions[:CROPS_PER_IMAGE]:
            crop_img = img[y:y+CROP_SIZE, x:x+CROP_SIZE]
            crop_lbl = lbl[y:y+CROP_SIZE, x:x+CROP_SIZE]

            # Ensure label is uint8
            if crop_lbl.max() <= 1:
                crop_lbl = (crop_lbl * 255).astype(np.uint8)

                roof_ratio = np.count_nonzero(crop_lbl) / crop_lbl.size

                if roof_ratio < 0.01 and random.random() > 0.2:
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
            if total >= target_count:
                return


# ---------------- TEST ----------------
def process_test(input_img_dir, output_img_dir, target_count):

    total = 0

    for fname in os.listdir(input_img_dir):
        if not fname.lower().endswith(".tif"):
            continue

        img = cv2.imread(os.path.join(input_img_dir, fname))
        h, w, _ = img.shape

        positions = grid_positions(h, w, CROP_SIZE)
        random.shuffle(positions)

        for y, x in positions[:CROPS_PER_IMAGE]:
            crop_img = img[y:y+CROP_SIZE, x:x+CROP_SIZE]
            cv2.imwrite(
                os.path.join(output_img_dir, f"{os.path.splitext(fname)[0]}_{y}_{x}.png"),
                crop_img
            )

            total += 1
            if total >= target_count:
                return


# ---------------- RUN ----------------
process_labeled(
    folders['train_images'],
    folders['train_labels'],
    resized_folders['train_images'],
    resized_folders['train_labels'],
    TRAIN_TARGET
)

process_labeled(
    folders['val_images'],
    folders['val_labels'],
    resized_folders['val_images'],
    resized_folders['val_labels'],
    VAL_TARGET
)

process_test(
    folders['test_images'],
    resized_folders['test_images'],
    TEST_TARGET
)

print("Cropping complete.")
