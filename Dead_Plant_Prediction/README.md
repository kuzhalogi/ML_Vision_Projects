# Dead Plant Prediction (CNN-Based Vision Model)

A deep learning project that predicts plant diseases using Convolutional Neural Networks (CNN).  
This repository contains the trained model, dataset samples, and the Jupyter notebook used to develop and evaluate the plant disease classification system.

## Project Structure

ML_Vision_Projects/Dead_Plant_Prediction
```
│
├── Model/
│ ├── plant_disease.keras # Saved model file
│ ├── plant_disease_model.json # Model architecture
│ └── plant_disease_model.weights.h5 # Trained weights
│
├── Plant_Disease_CNN.ipynb # Training & evaluation notebook
│
└── Plant_images/ # Image dataset (sample folders, 300 images each)
├── Maize_Common_rust/
├── Potato_Early_blight/
└── Tomato_Bacterial_spot/
```

## Project Overview

This project focuses on identifying plant leaf diseases using a convolutional neural network (CNN).  
It classifies images into **three plant disease categories**:

- **Maize — Common Rust**  
- **Potato — Early Blight**  
- **Tomato — Bacterial Spot**

The goal is to provide a lightweight model that can be integrated into agricultural monitoring systems or mobile applications for early detection of plant diseases.

##  Model Architecture

The CNN used in this project is lightweight and optimized for small datasets.  
Here is the exact architecture:


- **Total Parameters:** ~231K  
- **Trainable Parameters:** All  
- **Output Classes:** 3 (Maize Rust, Potato Early Blight, Tomato Bacterial Spot)

## Notebook
[Plant_Disease_CNN.ipynb](./Plant_Disease_CNN.ipynb) includes:

- Data preprocessing
- Data augmentation
- CNN architecture building
- Training and validation graphs
- Model saving and evaluation
- Sample predictions