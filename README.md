# UNet Rooftop Segmentation

## Description
This project implements a UNet model for rooftop segmentation using satellite imagery.

## Dataset
The dataset is available on Kaggle: https://www.kaggle.com/datasets/opusdeivas/airs-resized
'
> **Note:** The dataset (~5GB) is not included in this repo. Download from Kaggle and place it in the "resized/" folder locally.

If you want to create your own dataset download: https://www.kaggle.com/datasets/atilol/aerialimageryforroofsegmentation and run resize_images.py

## Package specifics

Tested to work with:

Keras version: 3.10.0
Tensorflow version: 2.20.0

## File structure

UNet_Rooftop_Segmentation/  <br>  
├── README.md  <br>
├── .gitignore  <br>
├── resize_images.py  <br>
├── RoofTopSegmentationUNet.py  <br>
├── simple_multi_unet_model.py  <br>
├── outputs/  <br>
│   ├── efficientnet.keras  <br>
│   ├── resnet50.keras  <br>
│   └── unet.keras  <br>
└── resized/  # Dataset (not included)  <br>
    ├── test/  <br>
    │	└── image/  <br>
    ├── train/  <br>
    │	├── image/  <br>
    │	└── label/  <br>
    └── val/  <br>
  	├── image/  <br>
   	└── label/  <br>
