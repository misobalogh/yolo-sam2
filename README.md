# YOLO-SAM2 Object Segmentation

## Overview
This project combines YOLO (You Only Look Once) object detection with SAM2 (Segment Anything Model 2) for precise object segmentation. The workflow involves:
1. Detecting objects using YOLO to get bounding boxes
2. Passing these bounding boxes to SAM2 for high-quality segmentation masks

## Setup

### Requirements
Install the required dependencies:
```
pip install -r requirements.txt
```

### Models
- YOLO model: Trained on custom dataset 
- SAM2 model: Download `sam2_b.pt` from [Ultralytics](https://github.com/ultralytics/ultralytics)

## Usage

### Training
Train the YOLO model on custom dataset:
```python
from ultralytics import YOLO
yolo_model = YOLO("yolo11n.pt")
results = yolo_model.train(data="dataset.yaml", epochs=10, imgsz=640)
```

### Inference
Run inference using the provided script:
```
python yolo_sam2_inference.py --img path/to/image.png --yolo-model runs/detect/train/weights/best.pt --sam-model sam2_b.pt
```

Or see the Jupyter notebook `yolo-sam2.ipynb`.

## Author
Michal Balogh