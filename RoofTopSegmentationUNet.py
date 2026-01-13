import os

import numpy as np
import tensorflow as tf
import random
import tensorflow_datasets as tfds
import tensorflow_advanced_segmentation_models as tasm
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from simple_multi_unet_model import multi_unet_model, jacard_coef
from IPython.display import clear_output 
from PIL import Image


# ---------------- CONFIG ----------------
training = 0
test_image_number = 10 # Number of test images to run inference on

PATCH_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 11
AUTOTUNE = tf.data.AUTOTUNE

DATA_ROOT = r"C:\Users\PC\Desktop\UNet_Rooftop_Segmentation\resized"
OUTPUT_ROOT = r"C:\Users\PC\Desktop\UNet_Rooftop_Segmentation\outputs"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

MODEL_TYPES = ["unet", "resnet50", "efficientnet"]


# ---------------- DATA PIPELINE ----------------
def parse_image_pair(image_path, label_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize [0,1]
    image = tf.image.resize(image, (PATCH_SIZE, PATCH_SIZE))

    label = tf.io.read_file(label_path)
    label = tf.image.decode_png(label, channels=1)
    label = tf.image.resize(label, (PATCH_SIZE, PATCH_SIZE), method="nearest")
    label = tf.cast(label > 0, tf.float32)

    return image, label



def unet_with_backbone(backbone_name="resnet50", input_shape=(256, 256, 3), n_classes=1):
    """
    U-Net with pretrained backbone (ResNet50 or EfficientNetB0)
    """
    inputs = layers.Input(shape=input_shape)
    
    # Select backbone
    if backbone_name == "resnet50":
        backbone = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
        # Skip connection layers for ResNet50
        skip_names = [
            "conv1_relu",           # 128x128
            "conv2_block3_out",     # 64x64
            "conv3_block4_out",     # 32x32
            "conv4_block6_out",     # 16x16
        ]
    elif backbone_name == "efficientnetb0":
        
        backbone = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=inputs)
        
        # Skip connection layers for EfficientNetB0
        skip_names = [
            "block2a_expand_activation",  # 128x128
            "block3a_expand_activation",  # 64x64
            "block4a_expand_activation",  # 32x32
            "block6a_expand_activation",  # 16x16
        ]
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    # Freeze backbone
    backbone.trainable = False
    
    # Get skip connections
    skips = [backbone.get_layer(name).output for name in skip_names]
    
    # Bottleneck (encoder output)
    x = backbone.output
    
    # Decoder
    decoder_filters = [256, 128, 64, 32]
    
    for i, filters in enumerate(decoder_filters):
        # Upsample
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(x)
        
        # Skip connection (if available and shapes match)
        if i < len(skips):
            skip = skips[-(i + 1)]  # Reverse order
            # Resize skip if needed
            if skip.shape[1] != x.shape[1] or skip.shape[2] != x.shape[2]: 
                skip = layers.Resizing(x.shape[1], x.shape[2])(skip)
            x = layers.Concatenate()([x, skip])
        
        # Conv blocks
        x = layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
    
    # Final upsample to match input size
    x = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(x)
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    
    # Output
    outputs = layers.Conv2D(n_classes, (1, 1), activation="sigmoid")(x)
    
    model = Model(inputs, outputs, name=f"unet_{backbone_name}")
    return model

def build_dataset(img_dir, lbl_dir):
    # ADD FILE EXTENSION FILTERING
    imgs = sorted([
        os.path.join(img_dir, f) 
        for f in os.listdir(img_dir)
        if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))
    ])
    lbls = sorted([
        os.path.join(lbl_dir, f) 
        for f in os.listdir(lbl_dir)
        if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))
    ])
    
    # ADD VALIDATION
    assert len(imgs) == len(lbls), f"Mismatch: {len(imgs)} images vs {len(lbls)} labels"
    print(f"Found {len(imgs)} image-label pairs")

    ds = tf.data.Dataset.from_tensor_slices((imgs, lbls))
    
    # Shuffle FIRST with full buffer
    ds = ds.shuffle(buffer_size=len(imgs), reshuffle_each_iteration=True)
    ds = ds.map(parse_image_pair, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ---------------- MODEL FACTORY ----------------
def get_model(model_type):

    tf.keras.backend.clear_session()

    if model_type == "unet":
        return multi_unet_model(
            n_classes=1,
            IMG_HEIGHT=PATCH_SIZE,
            IMG_WIDTH=PATCH_SIZE,
            IMG_CHANNELS=3
        )

    if model_type == "resnet50":
        return unet_with_backbone(
            backbone_name="resnet50",
            input_shape=(PATCH_SIZE, PATCH_SIZE, 3),
            n_classes=1
        )

    if model_type == "efficientnet":
        return unet_with_backbone(
            backbone_name="efficientnetb0", 
            input_shape=(PATCH_SIZE, PATCH_SIZE, 3),
            n_classes=1
        )

    raise ValueError(f"Unknown model type: {model_type}")


# ---------------- PREPROCESSING ----------------
def apply_preprocessing(ds, model_type):
    if model_type == "resnet50":
        return ds.map(lambda x, y: (resnet_preprocess(x * 255.0), y),
                      num_parallel_calls=AUTOTUNE)

    if model_type == "efficientnet": 
        return ds.map(lambda x, y: (eff_preprocess(x * 255.0), y),
                      num_parallel_calls=AUTOTUNE)

    return ds


# ---------------- TRAIN ALL MODELS ----------------
def train_all_models(data_root, output_root):
    train_ds = build_dataset(
        os.path.join(data_root, "train/image"),
        os.path.join(data_root, "train/label")
    )

    val_ds = build_dataset(
        os.path.join(data_root, "val/image"),
        os.path.join(data_root, "val/label")
    )

    test_img_dir = os.path.join(data_root, "test/image")
    test_image_path = os.path.join(test_img_dir, os.listdir(test_img_dir)[0])

    for model_type in MODEL_TYPES:
        print(f"\n{'='*50}")
        print(f"TRAINING {model_type.upper()}")
        print(f"{'='*50}")

        model = get_model(model_type)
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[jacard_coef]
        )

        train_p = apply_preprocessing(train_ds, model_type)
        val_p = apply_preprocessing(val_ds, model_type)

        history = model.fit(
            train_p,
            validation_data=val_p,
            epochs=EPOCHS,
            verbose=1
        )

        # ---- SAVE MODEL ----
        model_path = os.path.join(output_root, f"{model_type}.keras")
        model.save(model_path)

        # ---- SAVE PLOTS ----
        for metric in ["loss", "jacard_coef"]:
            plt.figure()
            plt.plot(history.history[metric], label="train")
            plt.plot(history.history[f"val_{metric}"], label="val")
            plt.legend()
            plt.title(f"{model_type} - {metric}")
            plt.savefig(os.path.join(output_root, f"{model_type}_{metric}.png"))
            plt.close()
        
        tf.keras.backend.clear_session()

    print("\n ALL MODELS TRAINED SUCCESSFULLY")

# ---------------- IoU CALCULATION ----------------
def calculate_iou(y_true, y_pred):
    """
    Calculate Intersection over Union (IoU) for binary masks.
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    
    intersection = np. logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


# ---------------- TEST ALL MODELS ----------------
def test_all_models(data_root, output_root, test_image_number):
    """
    Test all models on the test dataset with IoU calculation and visualizations.
    
    Outputs:
    1. 2x2 grid: Ground truth + 3 model predictions (with IoU scores)
    2. 3-image overlay: Predictions overlaid on original images
    """

    # Use the new test dataset with labels
    test_img_dir = os.path.join(data_root, "test/image")
    test_lbl_dir = os.path.join(data_root, "test/label")

    all_images = sorted([
        f for f in os.listdir(test_img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ])

    if test_image_number > len(all_images):
        raise ValueError(f"test_image_number ({test_image_number}) exceeds available images ({len(all_images)})")

    # Randomly select images WITHOUT replacement
    selected_images = random.sample(all_images, test_image_number)

    # Load all models first
    models_dict = {}
    for model_type in MODEL_TYPES:
        tf.keras.backend.clear_session()
        model_path = os.path.join(output_root, f"{model_type}.keras")
        print(f"Loading {model_type} model...")
        models_dict[model_type] = load_model(
            model_path,
            custom_objects={"jacard_coef": jacard_coef}
        )

    # Track IoU scores for summary
    iou_scores = {model_type: [] for model_type in MODEL_TYPES}

    for img_name in selected_images: 
        print(f"\nProcessing: {img_name}")
        
        # Load image and label
        test_image_path = os.path.join(test_img_dir, img_name)
        test_label_path = os.path.join(test_lbl_dir, img_name)

        img = Image.open(test_image_path).convert("RGB").resize((PATCH_SIZE, PATCH_SIZE))
        img_arr = np.array(img)
        base_arr = np.expand_dims(img_arr, axis=0).astype(np.float32)

        # Load ground truth label
        if os.path.exists(test_label_path):
            lbl = Image.open(test_label_path).convert("L").resize((PATCH_SIZE, PATCH_SIZE), Image.NEAREST)
            ground_truth = (np.array(lbl) > 0).astype(np.uint8)
        else:
            print(f"  Warning: No label found for {img_name}, skipping IoU calculation")
            ground_truth = None

        # Get predictions from all models
        predictions = {}
        ious = {}

        for model_type in MODEL_TYPES:
            model = models_dict[model_type]

            # Prepare input per model
            if model_type == "resnet50":
                arr = resnet_preprocess(base_arr. copy())
            elif model_type == "efficientnet":
                arr = eff_preprocess(base_arr.copy())
            else:
                arr = base_arr.copy() / 255.0

            pred = model.predict(arr, verbose=0)[0, :, :, 0] > 0.5
            predictions[model_type] = pred. astype(np.uint8)

            # Calculate IoU if ground truth exists
            if ground_truth is not None:
                iou = calculate_iou(ground_truth, predictions[model_type])
                ious[model_type] = iou
                iou_scores[model_type]. append(iou)
                print(f"  {model_type}: IoU = {iou:.4f}")
            else:
                ious[model_type] = None

        # ---- VISUALIZATION 1: 2x2 Grid (Ground Truth + 3 Predictions) ----
        create_comparison_grid(
            img_name, ground_truth, predictions, ious, output_root
        )

        # ---- VISUALIZATION 2: Overlay on Original Image ----
        create_overlay_visualization(
            img_name, img_arr, predictions, ious, output_root
        )

    # ---- Print Summary Statistics ----
    print_iou_summary(iou_scores)


def create_comparison_grid(img_name, ground_truth, predictions, ious, output_root):
    """
    Create 2x2 grid visualization: 
    - Top left: Ground truth mask
    - Other 3: Model predictions
    - Model name above each image, IoU below
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f"Segmentation Comparison:  {img_name}", fontsize=14, fontweight='bold')

    # Top-left: Ground Truth
    ax = axes[0, 0]
    if ground_truth is not None: 
        ax.imshow(ground_truth, cmap='gray')
        ax.set_title("Ground Truth", fontsize=12, fontweight='bold', color='green')
    else:
        ax.text(0.5, 0.5, "No Ground Truth", ha='center', va='center', fontsize=12)
        ax.set_title("Ground Truth", fontsize=12, fontweight='bold', color='red')
    ax.axis('off')

    # Other 3 positions:  Model predictions
    positions = [(0, 1), (1, 0), (1, 1)]
    colors = {'unet': 'blue', 'resnet50': 'orange', 'efficientnet': 'purple'}

    for (model_type, (row, col)) in zip(MODEL_TYPES, positions):
        ax = axes[row, col]
        ax.imshow(predictions[model_type], cmap='gray')
        
        # Title with model name
        ax.set_title(f"{model_type. upper()}", fontsize=12, fontweight='bold', 
                     color=colors. get(model_type, 'black'))
        
        # IoU below image
        if ious[model_type] is not None:
            ax.set_xlabel(f"IoU: {ious[model_type]:.4f}", fontsize=11, fontweight='bold')
        else:
            ax.set_xlabel("IoU:  N/A", fontsize=11)
        
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    save_name = f"{os.path.splitext(img_name)[0]}_comparison_grid.png"
    plt. savefig(os.path. join(output_root, save_name), dpi=150, bbox_inches='tight')
    plt.close()


def create_overlay_visualization(img_name, img_arr, predictions, ious, output_root):
    """
    Create overlay visualization: 
    - 3 images side by side
    - Each shows predicted mask overlaid on the original image
    - Different colors for each model
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Prediction Overlays: {img_name}", fontsize=14, fontweight='bold')

    # Color map for each model (RGBA with transparency)
    overlay_colors = {
        'unet': [0, 0, 255],       # Blue
        'resnet50': [255, 165, 0],  # Orange
        'efficientnet': [128, 0, 128]  # Purple
    }

    for ax, model_type in zip(axes, MODEL_TYPES):
        # Create overlay
        overlay = img_arr.copy()
        mask = predictions[model_type]. astype(bool)
        
        # Apply colored overlay where mask is True
        color = overlay_colors[model_type]
        alpha = 0.4  # Transparency
        
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c]
            )

        ax.imshow(overlay. astype(np.uint8))
        ax.set_title(f"{model_type.upper()}", fontsize=12, fontweight='bold')
        
        if ious[model_type] is not None:
            ax.set_xlabel(f"IoU:  {ious[model_type]:.4f}", fontsize=11, fontweight='bold')
        else:
            ax.set_xlabel("IoU: N/A", fontsize=11)
        
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    save_name = f"{os.path.splitext(img_name)[0]}_overlay.png"
    plt.savefig(os.path.join(output_root, save_name), dpi=150, bbox_inches='tight')
    plt.close()


def print_iou_summary(iou_scores):
    """
    Print summary statistics for IoU scores across all test images.
    """
    print("\n" + "=" * 60)
    print("IoU SUMMARY STATISTICS")
    print("=" * 60)

    for model_type in MODEL_TYPES:
        scores = iou_scores[model_type]
        if scores: 
            mean_iou = np.mean(scores)
            std_iou = np.std(scores)
            min_iou = np.min(scores)
            max_iou = np.max(scores)
            print(f"\n{model_type. upper()}:")
            print(f"  Mean IoU:   {mean_iou:.4f}")
            print(f"  Std IoU:   {std_iou:.4f}")
            print(f"  Min IoU:   {min_iou:.4f}")
            print(f"  Max IoU:   {max_iou:.4f}")
        else:
            print(f"\n{model_type.upper()}: No IoU scores available")

    print("\n" + "=" * 60)

        
# ---------------- RUN ----------------
if training == 1:
    train_all_models(DATA_ROOT, OUTPUT_ROOT)

if training == 0:
    test_all_models(DATA_ROOT, OUTPUT_ROOT, test_image_number)    
