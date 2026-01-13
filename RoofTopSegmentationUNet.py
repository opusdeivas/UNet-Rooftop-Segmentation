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
test_image_number = 10

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

# ---------------- TEST ALL MODELS ----------------
def test_all_models(data_root, output_root, test_image_number):

    test_img_dir = os.path.join(data_root, "test/image")
    all_images = sorted(os.listdir(test_img_dir))

    if test_image_number > len(all_images):
        raise ValueError("test_image_number exceeds number of available test images")

    # Randomly select images WITHOUT replacement
    selected_images = random.sample(all_images, test_image_number)

    for model_type in MODEL_TYPES:  # Move model loading OUTSIDE the image loop
        # Clear session before loading each model
        tf.keras.backend.clear_session()
        
        model_path = os.path.join(output_root, f"{model_type}.keras")
        print(f"\nLoading {model_type} model...")
        
        model = load_model(
            model_path,
            custom_objects={"jacard_coef":  jacard_coef}
        )

        for img_name in selected_images:
            test_image_path = os.path.join(test_img_dir, img_name)

            img = Image.open(test_image_path).convert("RGB").resize((PATCH_SIZE, PATCH_SIZE))
            base_arr = np.expand_dims(np.array(img), axis=0).astype(np.float32)

            # Prepare input per model
            if model_type == "resnet50":
                arr = resnet_preprocess(base_arr. copy())
            elif model_type == "efficientnet": 
                arr = eff_preprocess(base_arr. copy())
            else:
                arr = base_arr.copy() / 255.0

            print(f"{model_type} - {img_name}:  {arr.shape}")
            pred = model.predict(arr, verbose=0)[0, :, : , 0] > 0.5

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.axis("off")
            plt.title("Image")

            plt.subplot(1, 2, 2)
            plt.imshow(pred, cmap="gray")
            plt.axis("off")
            plt.title(f"Prediction ({model_type})")

            save_name = f"{os.path.splitext(img_name)[0]}_{model_type}_prediction.png"
            plt.savefig(os.path.join(output_root, save_name))
            plt.close()

        
# ---------------- RUN ----------------
if training == 1:
    train_all_models(DATA_ROOT, OUTPUT_ROOT)

if training == 0:
    test_all_models(DATA_ROOT, OUTPUT_ROOT, test_image_number)    
















































# import tensorflow as tf
# from tensorflow.keras import layers, models # type: ignore
# from keras.utils import to_categorical
# from keras.saving import register_keras_serializable
# from keras.models import load_model
# from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_class_weight
# import tensorflow_datasets as tfds
# import tensorflow_advanced_segmentation_models as tasm
# from keras.applications.resnet50 import preprocess_input as resnet_preprocess
# from simple_multi_unet_model import multi_unet_model, jacard_coef
# import numpy as np
# from IPython.display import clear_output 
# import matplotlib.pyplot as plt
# from PIL import Image
# import random
# import os

# # ===========================================================
# # CONFIG
# # ===========================================================
# root_directory = r"C:\Users\PC\Desktop\Roof_segmentation\resized"
# patch_size = 256
# TRAIN_SIZE = 25
# training = 0

# AUTOTUNE = tf.data.AUTOTUNE

# # ===========================================================
# # TF.DATA DATASET BUILDER (optimized)
# # ===========================================================
# def parse_image_pair(image_path, label_path, img_size=(256, 256)):
#     # Read and decode image
#     image = tf.io.read_file(image_path)
#     image = tf.image.decode_png(image, channels=3)
#     image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize [0,1]
#     image = tf.image.resize(image, img_size)

#     # Read and decode label
#     label = tf.io.read_file(label_path)
#     label = tf.image.decode_png(label, channels=1)
#     label = tf.image.resize(label, img_size, method="nearest")
#     label = tf.cast(label > 0, tf.float32)

#     return image, label

# def build_dataset(image_dir, label_dir, batch_size=8, img_size=(256, 256)):
#     image_files = sorted([
#         os.path.join(image_dir, f)
#         for f in os.listdir(image_dir)
#         if f.endswith(".tif") or f.endswith(".png")
#     ])
#     label_files = sorted([
#         os.path.join(label_dir, f)
#         for f in os.listdir(label_dir)
#         if f.endswith(".tif") or f.endswith(".png")
#     ])

#     ds = tf.data.Dataset.from_tensor_slices((image_files, label_files))
#     ds = ds.shuffle(buffer_size=len(image_files), reshuffle_each_iteration=True)
#     ds = ds.map(lambda x, y: parse_image_pair(x, y, img_size),
#                 num_parallel_calls=AUTOTUNE)
#     ds = ds.batch(batch_size)
#     ds = ds.prefetch(AUTOTUNE)
#     return ds

# # ===========================================================
# # DATASET SETUP
# # ===========================================================
# train_img_dir = os.path.join(root_directory, "train", "image")
# train_lbl_dir = os.path.join(root_directory, "train", "label")

# val_img_dir = os.path.join(root_directory, "val", "image")
# val_lbl_dir = os.path.join(root_directory, "val", "label")

# # build tf.data dataset
# train_ds = build_dataset(train_img_dir, train_lbl_dir, batch_size=8, img_size=(patch_size, patch_size))
# val_ds   = build_dataset(val_img_dir, val_lbl_dir, batch_size=8, img_size=(patch_size, patch_size))


# # For summary/debugging
# # print("\n--- Dataset Summary ---")
# # print(f"Train batches: {len(list(train_ds))}, Val batches: {len(list(val_ds))}")

# # ===========================================================
# # MODEL AND LOSS
# # ===========================================================
# base_model = tf.keras.applications.MobileNetV2(input_shape=[256,256,3], include_top=False)

# weights = [1.222, 1.222]

# @register_keras_serializable(package="CustomLosses")
# def total_loss(y_true, y_pred):
#     dice_loss = tasm.losses.DiceLoss(class_weights=weights)
#     focal_loss = tasm.losses.CategoricalFocalLoss()
#     return dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)

# metrics = ['accuracy', jacard_coef]

# # ===========================================================
# # TRAINING SECTION
# # ===========================================================
# if training == 1:
#     # Infer class count dynamically from one label sample
#     sample_image, sample_label = next(iter(train_ds))
#     n_classes = 1

#     IMG_HEIGHT = patch_size
#     IMG_WIDTH  = patch_size
#     IMG_CHANNELS = 3

#     def get_model():
#         return multi_unet_model(
#             n_classes=n_classes,
#             IMG_HEIGHT=IMG_HEIGHT,
#             IMG_WIDTH=IMG_WIDTH,
#             IMG_CHANNELS=IMG_CHANNELS
#         )

#     model = get_model()
#     model.compile(optimizer='adam', loss=tasm.losses.DiceLoss(class_weights=weights) + tasm.losses.CategoricalFocalLoss(), metrics=['accuracy', jacard_coef])
#     model.summary()

#     # Train model
#     history1 = model.fit(
#         train_ds,
#         validation_data=val_ds,
#         batch_size=8,
#         verbose=1,
#         epochs=11
#     )

#     model.save('rooftop_segmentation_model.keras')

#     # -------------------------------------------------------
#     # SECOND TRAINING STAGE (resnet backbone)
#     # -------------------------------------------------------

#     # BACKBONE = 'resnet34'
#     # # preprocess_input = tasm.get_preprocessing(BACKBONE)

#     # # load dataset again for preprocessing
#     # def preprocess_dataset(ds):
#     #     return ds.map(lambda x, y: (resnet_preprocess(x * 255.0), y), num_parallel_calls=AUTOTUNE)

#     # X_train_prepr = preprocess_dataset(train_ds)
#     # X_test_prepr = preprocess_dataset(val_ds)

#     # model_resnet_backbone = tasm.UNet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation='softmax')
#     # model_resnet_backbone.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
#     # print(model_resnet_backbone.summary())

#     # history2 = model_resnet_backbone.fit(
#     #     X_train_prepr,
#     #     validation_data=X_test_prepr,
#     #     batch_size=10,
#     #     epochs=30,
#     #     verbose=1
#     # )

#     # model.save('rooftop_segmentation_model_cat_crossentropy.kera')

#     # -------------------------------------------------------
#     # PLOTS
#     # -------------------------------------------------------
#     history = history1
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     epochs = range(1, len(loss) + 1)

#     plt.plot(epochs, loss, 'y', label='Training loss')
#     plt.plot(epochs, val_loss, 'r', label='Validation loss')
#     plt.title('Training and validation loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()

#     acc = history.history['jacard_coef']
#     val_acc = history.history['val_jacard_coef']

#     plt.plot(epochs, acc, 'y', label='Training IoU')
#     plt.plot(epochs, val_acc, 'r', label='Validation IoU')
#     plt.title('Training and validation IoU')
#     plt.xlabel('Epochs')
#     plt.ylabel('IoU')
#     plt.legend()
#     plt.show()

    

# # ===========================================================
# # INFERENCE SECTION
# # ===========================================================
# # if training == 0:
# #     model = load_model('rooftop_segmentation_model.keras',
# #                        custom_objects={'dice_loss_plus_2focal_loss': total_loss,
# #                                        'jacard_coef': jacard_coef})
# #     print("model loaded successfully")

# #     # Create test dataset
# #     test_img_dir = os.path.join(root_directory, "test", "image")
# #     test_lbl_dir = os.path.join(root_directory, "test", "label")
# #     test_ds = build_dataset(test_img_dir, test_lbl_dir, batch_size=1, img_size=(patch_size, patch_size))

# #     from keras.metrics import MeanIoU
# #     IOU_keras = MeanIoU(num_classes=2)

# #     all_preds, all_labels = [], []
# #     for x, y in test_ds:
# #         preds = model.predict(x)
# #         preds_argmax = np.argmax(preds, axis=-1)
# #         labels_argmax = np.squeeze(y.numpy().astype(np.int32))
# #         all_preds.append(preds_argmax)
# #         all_labels.append(labels_argmax)
# #     all_preds = np.concatenate(all_preds)
# #     all_labels = np.concatenate(all_labels)

# #     IOU_keras.update_state(all_labels, all_preds)
# #     print("Mean IoU =", IOU_keras.result().numpy())



#     # # Display one prediction
#     # for x, _ in test_ds.take(1):
#     #     prediction = model.predict(x)
#     #     predicted_img = np.argmax(prediction, axis=3)[0, :, :]
#     #     plt.figure(figsize=(12, 6))
#     #     plt.subplot(1, 2, 1)
#     #     plt.title('Testing Image')
#     #     plt.imshow(x[0])
#     #     plt.subplot(1, 2, 2)
#     #     plt.title('Predicted Mask')
#     #     plt.imshow(predicted_img, cmap='gray')
#     #     plt.show()

# if training == 0:
#     model = load_model(
#         'rooftop_segmentation_model.keras',
#         custom_objects={'jacard_coef': jacard_coef}
#     )
#     print("Model loaded successfully")

#     # ---- Specify your test image name here ----
#     test_image_name = "christchurch_1011_4096_2048.png"   # <<== CHANGE THIS
#     test_image_path = os.path.join(root_directory, "test", "image", test_image_name)

#     # ---- Load and preprocess the image ----
#     def load_and_preprocess_image(img_path, img_size=(256, 256)):
#         img = Image.open(img_path).convert('RGB')
#         img = img.resize(img_size)
#         img_array = np.array(img) / 255.0  # Normalize [0,1]
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#         return img_array, img

#     img_array, original_img = load_and_preprocess_image(test_image_path, img_size=(patch_size, patch_size))

#     # ---- Run prediction ----
#     prediction = model.predict(img_array)

#     # For binary segmentation (1 output channel)
#     if prediction.shape[-1] == 1:
#         predicted_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)
#     else:
#         predicted_mask = np.argmax(prediction, axis=3)[0, :, :]

#     # ---- Display result ----
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.title("Input Image")
#     plt.imshow(original_img)
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.title("Predicted Mask")
#     plt.imshow(predicted_mask, cmap='gray')
#     plt.axis("off")

#     plt.show()

