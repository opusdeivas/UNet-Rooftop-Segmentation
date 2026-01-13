# UNet Rooftop Segmentation

## Description
This project implements a UNet model for rooftop segmentation using satellite imagery.

## Dataset
The dataset is available on Kaggle: [AIRs Resized](https://www.kaggle.com/datasets/opusdeivas/airs-resized)
'
> **Note:** The dataset (~5GB) is not included in this repo. Download from Kaggle and place it in the "resized/" folder locally.

If you want to create your own dataset download: [Aerial Imagery for Roof Segmentation](https://www.kaggle.com/datasets/atilol/aerialimageryforroofsegmentation) and run resize_images.py. 

> **Note:** To have IoU usability, create_test_dataset.py needs to be run separately

## Package specifics

Tested to work with:

- Keras version: 3.10.0
- Tensorflow version: 2.20.0

## File structure
```
UNet_Rooftop_Segmentation/ 
├── README.md  
├── .gitignore  
├── resize_images.py
├── RoofTopSegmentationUNet.py
├── simple_multi_unet_model.py
├── outputs/
│   ├── efficientnet.keras
│   ├── resnet50.keras
│   └── unet.keras
└── resized/  # Dataset (not included) 
    ├── test/ 
    │	└── image/
    │ └── label/ 
    ├── train/ 
    │	├── image/  
    │	└── label/  
    └── val/ 
  	├── image/  
   	└── label/  
```
