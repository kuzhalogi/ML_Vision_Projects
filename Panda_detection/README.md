# Panda Detection using YOLOv8

This project uses the YOLOv8 object detection model to detect pandas in images.  
It includes dataset annotations, training scripts, and a Jupyter notebook for running inference on new images.

## Project Overview

This project focuses on detecting pandas in images using the YOLOv8 model.  
The workflow includes:

- Preparing the dataset and annotations in YOLO format  
- Training YOLOv8 on panda images  
- Validating and testing the model  
- Running inference on new images to detect pandas  


## Project Structure

```python
Panda_detection/
│
├── train/          # train images and labels
├── val/            # validation images and labels
├── test/           # test images and labels
├── yolo_labels/    # YOLO formatted labels
│ ├── train/
│ ├── val/
│ └── test/
├── Yolo_Data_annotation.py # Script to create YOLO annotations from dataset
└── YOLOv8_Inference.ipynb # Notebook to run detection and visualize results
```


