#!/usr/bin/env python3
"""
YOLO + SAM2 Object Segmentation
-------------------------------
Combined approach using YOLO for object detection and SAM2 for segmentation.
"""

import os
import argparse
import torch
from pathlib import Path
from ultralytics import YOLO, SAM


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLO + SAM2 Object Segmentation")
    parser.add_argument("--img", type=str, default="heightMap.png", help="Path to input image")
    parser.add_argument("--yolo-model", type=str, default="runs/detect/train/weights/best.pt",
                        help="Path to YOLO model")
    parser.add_argument("--sam-model", type=str, default="sam2_b.pt",
                        help="Path to SAM2 model")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Confidence threshold for YOLO detection")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results")
    return parser.parse_args()


def detect_objects(model, image_path, conf):
    """Run object detection using YOLO model."""
    print(f"Running YOLO detection on {image_path}...")
    return model(image_path, conf=conf)[0]


def segment_objects(sam_model, yolo_output, device="auto"):
    """Segment objects using SAM model based on YOLO detections."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running SAM segmentation using {device}...")
    boxes = yolo_output.boxes.xyxy

    # Process image with SAM model
    sam_output = sam_model(
        yolo_output.orig_img,
        bboxes=boxes,
        verbose=False,
        device=device,
        save=True
    )[0]

    # Transfer class information from YOLO to SAM output
    id2label = yolo_output.names
    class_ids = yolo_output.boxes.cls.int().tolist()

    sam_output_ids = {i: class_id for i, class_id in enumerate(class_ids)}
    sam_output.names = {k: id2label[int(v)] for k, v in sam_output_ids.items()}

    return sam_output


def main():
    """Main function."""
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load YOLO model
    print(f"Loading YOLO model from {args.yolo_model}...")
    yolo_model = YOLO(args.yolo_model)

    # Run detection
    yolo_output = detect_objects(yolo_model, args.img, args.conf)

    # Save YOLO detection result
    det_path = os.path.join(args.output_dir, "detection_" + Path(args.img).name)
    yolo_output.save(filename=det_path)
    print(f"Detection result saved to {det_path}")

    # Load SAM model
    print(f"Loading SAM model from {args.sam_model}...")
    sam_model = SAM(args.sam_model)

    # Run segmentation
    sam_output = segment_objects(sam_model, yolo_output, device)

    # Save and display results
    print("Segmentation classes:", sam_output.names)
    seg_path = os.path.join(args.output_dir, "segmentation_" + Path(args.img).name)
    sam_output.save(filename=seg_path)
    print(f"Segmentation result saved to {seg_path}")

    # Display results
    sam_output.show(labels=True)

    return sam_output


if __name__ == "__main__":
    main()
