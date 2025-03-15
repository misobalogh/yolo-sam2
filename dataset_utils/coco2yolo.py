# Michal Balogh, 2025

import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height]
    normalized to [0, 1]
    """
    x, y, w, h = bbox

    # Calculate center coordinates
    x_center = x + w / 2
    y_center = y + h / 2

    # Normalize
    x_center /= img_width
    y_center /= img_height
    w /= img_width
    h /= img_height

    return [x_center, y_center, w, h]


def convert_segmentation_to_yolo(segmentation, img_width, img_height):
    """
    Convert COCO segmentation format to YOLO format (normalized coordinates)
    """
    # YOLO expects [x1, y1, x2, y2, ...] normalized
    yolo_seg = []

    # If segmentation has multiple parts (e.g. A, R - inner and outer part)
    for seg in segmentation:
        if not seg:
            continue

        points = []
        # Process pairs of x,y coordinates
        for i in range(0, len(seg), 2):
            if i+1 < len(seg):
                # Normalize coordinates to [0, 1]
                x = seg[i] / img_width
                y = seg[i+1] / img_height
                points.extend([x, y])

        if points:
            yolo_seg.append(points)

    return yolo_seg


def create_yolo_annotation_file(annotation, img_width, img_height, labels_map):
    """
    Create YOLO annotation file content for a single annotation
    """
    category_id = annotation["category_id"]
    cls_id = labels_map.get(category_id, category_id - 1)  # Convert to 0-indexed

    # Convert bbox
    bbox = annotation["bbox"]
    yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)

    lines = []

    # Add bbox line
    bbox_line = f"{cls_id} {' '.join([str(round(coord, 6)) for coord in yolo_bbox])}"
    lines.append(bbox_line)

    # Add segmentation line(s) if segmentation exists
    if "segmentation" in annotation and annotation["segmentation"]:
        seg_polys = convert_segmentation_to_yolo(annotation["segmentation"], img_width, img_height)
        for poly in seg_polys:
            if poly:  # Only add non-empty polygons
                seg_line = f"{cls_id} {' '.join([str(round(coord, 6)) for coord in poly])}"
                lines.append(seg_line)

    return lines


def coco_to_yolo(coco_dir, output_dir, split="train"):
    """
    Convert COCO format annotations to YOLO format
    """
    # Create output directories
    img_output_dir = os.path.join(output_dir, "images", split)
    label_output_dir = os.path.join(output_dir, "labels", split)

    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)

    # Load COCO annotations
    ann_file = os.path.join(coco_dir, split, "annotations", "instances.json")
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return False

    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    # Create mapping from image_id to image details
    images_map = {img["id"]: img for img in coco_data["images"]}

    # Create mapping from category_id to index (YOLO uses 0-indexed classes)
    categories = sorted(coco_data["categories"], key=lambda x: x["id"])

    # Create class mapping and save classes.txt
    classes = [cat["name"] for cat in categories]
    with open(os.path.join(output_dir, "classes.txt"), 'w') as f:
        for cls_name in classes:
            f.write(f"{cls_name}\n")

    # Create mapping from COCO category_id to YOLO class index
    labels_map = {cat["id"]: i for i, cat in enumerate(categories)}

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    print(f"Converting {split} split to YOLO format...")

    # Process each image
    for img_id, annotations in tqdm(annotations_by_image.items()):
        if img_id not in images_map:
            print(f"Warning: Image ID {img_id} not found in images list!")
            continue

        img_info = images_map[img_id]
        img_file = img_info["file_name"]
        img_width = img_info["width"]
        img_height = img_info["height"]

        # Source image path
        src_img_path = os.path.join(coco_dir, split, "images", img_file)
        if not os.path.exists(src_img_path):
            print(f"Warning: Image not found: {src_img_path}")
            continue

        # Copy image to output directory (or link if preferred)
        dst_img_path = os.path.join(img_output_dir, img_file)
        if not os.path.exists(dst_img_path):
            img = Image.open(src_img_path)
            img.save(dst_img_path)

        # Create label file (same name as image but .txt extension)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(label_output_dir, label_file)

        # Process all annotations for this image
        all_lines = []
        for ann in annotations:
            lines = create_yolo_annotation_file(ann, img_width, img_height, labels_map)
            all_lines.extend(lines)

        # Write label file
        with open(label_path, 'w') as f:
            for line in all_lines:
                f.write(f"{line}\n")

    print(f"Conversion complete for {split} split!")
    return True


def create_dataset_yaml(output_dir, classes):
    """Create a dataset.yaml file for YOLO training"""
    yaml_content = f"""# YOLO Dataset Configuration - Converted from COCO format
path: {output_dir}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/test  # val images (relative to 'path')

# Classes
names:
"""

    # Add class names
    for i, cls_name in enumerate(classes):
        yaml_content += f"  {i}: {cls_name}\n"

    # Write YAML file
    with open(os.path.join(output_dir, "dataset.yaml"), 'w') as f:
        f.write(yaml_content)

    print(f"Created dataset.yaml in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Convert COCO format dataset to YOLO format')
    parser.add_argument('--coco_dir', type=str, default="coco_dataset_v1",
                        help='Path to COCO format dataset directory')
    parser.add_argument('--output_dir', type=str, default="yolo_dataset",
                        help='Output directory for YOLO format dataset')

    args = parser.parse_args()

    coco_dir = args.coco_dir
    output_dir = args.output_dir

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert train split
    train_success = coco_to_yolo(coco_dir, output_dir, split="train")

    # Convert test split
    test_success = coco_to_yolo(coco_dir, output_dir, split="test")

    if train_success or test_success:
        # Load class names from classes.txt to create dataset.yaml
        classes_file = os.path.join(output_dir, "classes.txt")
        if os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f]
            create_dataset_yaml(output_dir, classes)
        else:
            print("Warning: classes.txt not found, could not create dataset.yaml")

    print("Conversion complete!")


if __name__ == "__main__":
    main()