# Michal Balogh, 2025

import os
import time
import json
import datetime
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import tqdm
import cv2

from PIL import Image

from HeightMapDatabase import HeightMapDatabase



class HeightMapInstanceDataset(HeightMapDatabase):
    """Class for creating COCO-format instance segmentation datasets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter_classes = []
        self.instance_output_dir = "instance_masks"
        self.dataset_name = kwargs.get("dataset_name", "HeightMap Instance Dataset")
        self.dataset_description = kwargs.get("dataset_description", "Instance segmentation dataset of heightmaps")
        self.coco_version = kwargs.get("coco_version", "1.0")
        self.coco_url = kwargs.get("coco_url", "http://example.org")
        self.license_name = kwargs.get("license_name", "example_license")
        self.license_url = kwargs.get("license_url", "http://example.org")
        self.contributor = kwargs.get("contributor", "contributor_name")
        self.date_created = kwargs.get("date_created", datetime.datetime.now().strftime("%Y/%m/%d"))

    def create_coco_dataset(self, root_dir, output_dir="coco_dataset",
                            filter_classes=None, split_filtered=True):
        """
        Create a COCO format dataset from heightmaps and annotations.
        Saves images and annotations in the specified output directory.
        """
        start_time = time.time()
        os.makedirs(output_dir, exist_ok=True)
        images_dir = os.path.join(output_dir, "images")
        annotations_dir = os.path.join(output_dir, "annotations")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)

        filtered_images_dir = None
        filtered_coco_data = None

        if split_filtered and filter_classes:
            filtered_images_dir = os.path.join(output_dir, "filtered_images")
            os.makedirs(filtered_images_dir, exist_ok=True)

        coco_data = self._initialize_coco_structure()
        if split_filtered and filter_classes:
            filtered_coco_data = self._initialize_coco_structure()

        self._add_categories_to_coco(coco_data)
        if filtered_coco_data:
            self._add_categories_to_coco(filtered_coco_data)

        image_id, filtered_image_id = 1, 1
        annotation_id, filtered_annotation_id = 1, 1
        stats = {"processed": 0, "filtered": 0, "errors": 0, "annotations": 0, "filtered_annotations": 0}

        for dir_name in tqdm.tqdm(os.listdir(root_dir)):
            dir_path = os.path.join(root_dir, dir_name)
            if not os.path.isdir(dir_path):
                continue
            splitted_path = os.path.join(dir_path, self.splitted_dir)
            if not os.path.isdir(splitted_path):
                continue

            for sub_name in tqdm.tqdm(os.listdir(splitted_path)):
                sub_path = os.path.join(splitted_path, sub_name)
                if not os.path.isdir(sub_path):
                    continue

                try:
                    heightmap_path = os.path.join(sub_path, "heightMap.png")
                    if not os.path.exists(heightmap_path):
                        continue
                    original_image = Image.open(heightmap_path)
                    width, height = original_image.size

                    annotation_file = os.path.join(sub_path, self.csv_file)
                    if not os.path.exists(annotation_file):
                        continue
                    df = pd.read_csv(annotation_file)

                    has_filtered_classes = False
                    if filter_classes and any(cls in df["Unicode"].unique() for cls in filter_classes):
                        has_filtered_classes = True
                        stats["filtered"] += 1

                    if has_filtered_classes and split_filtered:
                        image_filename = f"{filtered_image_id:012d}.png"
                        original_image.save(os.path.join(filtered_images_dir, image_filename))
                        filtered_coco_data["images"].append({
                            "id": filtered_image_id,
                            "width": width,
                            "height": height,
                            "file_name": image_filename,
                            "license": 1,
                            "date_captured": self.date_created
                        })
                        for _, row in df.iterrows():
                            anns = self._process_instance_annotation(
                                row, sub_path, filtered_image_id, filtered_annotation_id, filter_classes
                            )
                            for ann in anns:
                                filtered_coco_data["annotations"].append(ann)
                                filtered_annotation_id += 1
                                stats["filtered_annotations"] += 1
                        filtered_image_id += 1
                    else:
                        image_filename = f"{image_id:012d}.png"
                        original_image.save(os.path.join(images_dir, image_filename))
                        coco_data["images"].append({
                            "id": image_id,
                            "width": width,
                            "height": height,
                            "file_name": image_filename,
                            "license": 1,
                            "date_captured": self.date_created
                        })
                        for _, row in df.iterrows():
                            anns = self._process_instance_annotation(
                                row, sub_path, image_id, annotation_id
                            )
                            for ann in anns:
                                coco_data["annotations"].append(ann)
                                annotation_id += 1
                                stats["annotations"] += 1
                        image_id += 1
                        stats["processed"] += 1

                except Exception as e:
                    stats["errors"] += 1
                    print(f"Error processing {sub_path}: {e}")
                    exit(1)

        with open(os.path.join(annotations_dir, "instances.json"), 'w') as f:
            json.dump(coco_data, f, indent=2)

        if filtered_coco_data and split_filtered:
            with open(os.path.join(annotations_dir, "filtered_instances.json"), 'w') as f:
                json.dump(filtered_coco_data, f, indent=2)

        end_time = time.time()
        stats["time_elapsed"] = end_time - start_time
        stats["total_images"] = image_id - 1
        stats["total_filtered"] = filtered_image_id - 1

        return stats

    def create_coco_dataset_from_split(self, output_dir=None):
        """
        Create COCO format datasets for training and testing splits.
        """
        if output_dir is None:
            output_dir = self.instance_output_dir
        os.makedirs(output_dir, exist_ok=True)

        stats = {}
        for split in ["train", "test"]:
            split_dir = self.get_split_dir(split)
            split_output_dir = os.path.join(output_dir, split)
            os.makedirs(split_output_dir, exist_ok=True)
            stats[split] = self.create_coco_dataset(
                split_dir,
                output_dir=split_output_dir,
                filter_classes=self.filter_classes
            )
        return stats

    def _initialize_coco_structure(self):
        """Initialize the COCO structure for JSON output."""
        return {
            "info": {
                "description": self.dataset_description,
                "url": self.coco_url,
                "version": self.coco_version,
                "year": datetime.datetime.now().year,
                "contributor": self.contributor,
                "date_created": self.date_created
            },
            "licenses": [
                {
                    "id": 1,
                    "name": self.license_name,
                    "url": self.license_url
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }

    def _add_categories_to_coco(self, coco_data):
        """Add categories to the COCO structure using known labels."""
        if not self.labels:
            self.get_all_labels()
        sorted_labels = sorted(list(self.labels))
        categories = []
        for i, label_val in enumerate(sorted_labels):
            category_id = i + 1
            category_name = chr(label_val) if chr(label_val) <= 'z' else f'special_symbol_{label_val}'
            categories.append({
                "id": category_id,
                "name": category_name,
                "supercategory": "glyph"
            })
        coco_data["categories"] = categories

    def _process_instance_annotation(self, row, subdir_path, image_id, annotation_id, exclude_classes=None):
        """
        Process a single row from the CSV to return COCO annotation dict.
        Uses the exact binary mask instead of polygon approximation.
        """
        annotations = []
        unicode_val = int(row["Unicode"])
        if exclude_classes and unicode_val in exclude_classes:
            return annotations
        if self.unicode2id_map is None:
            self.get_unicode_to_id_mapping()
        category_id = self.get_class_id(unicode_val)

        # Get the mask file path
        mask_file = os.path.join(subdir_path, row["Mask"])
        if not os.path.exists(mask_file):
            return annotations

        # Get position information from CSV
        x, y, w, h = int(row["X"]), int(row["Y"]), int(row["Width"]), int(row["Height"])
        bbox = [x, y, w, h]

        # Load the binary mask
        mask = Image.open(mask_file)
        binary_mask = np.array(mask) > 0

        # Calculate area from the binary mask
        area = int(np.sum(binary_mask))

        # Convert binary mask to RLE format for COCO
        rle = self._binary_mask_to_rle(binary_mask, x, y)

        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": area,
            "segmentation": rle,
            "iscrowd": 0
        }
        annotations.append(annotation)
        return annotations

    def _binary_mask_to_rle(self, binary_mask, x_offset=0, y_offset=0):
        """
        Convert a binary mask to RLE format, keeping the exact pixel values.

        Args:
            binary_mask: Binary mask as numpy array
            x_offset: X position where mask should be placed in the image
            y_offset: Y position where mask should be placed in the image

        Returns:
            RLE representation of the mask for COCO format
        """
        # Create a full-size mask with the correct position
        height, width = binary_mask.shape

        # To preserve the exact mask format in COCO that uses polygons,
        # we'll create an array of points that exactly follow the mask boundary
        contours, hierarchy = cv2.findContours(binary_mask.astype(np.uint8),
                                        cv2.RETR_CCOMP, # RETR_CCOMP to capture both external boundaries and holes
                                        cv2.CHAIN_APPROX_NONE)

        segmentations = []
        for i, contour in enumerate(contours):
            # Apply position offset and flatten to COCO polygon format [x1, y1, x2, y2, ...]
            adjusted_contour = contour.reshape(-1, 2)
            adjusted_contour[:, 0] += x_offset  # Add x_offset to x coordinates
            adjusted_contour[:, 1] += y_offset  # Add y_offset to y coordinates

            if len(adjusted_contour) >= 3:  # Need at least 3 points for a polygon
                flattened = adjusted_contour.flatten().tolist()
                segmentations.append(flattened)

        # If no valid contours found, create a simple polygon from the bounding box
        if not segmentations:
            segmentations = [[
                x_offset, y_offset,
                x_offset + width, y_offset,
                x_offset + width, y_offset + height,
                x_offset, y_offset + height
            ]]

        return segmentations


if __name__ == "__main__":
    dataset_utils = HeightMapInstanceDataset(
        train_dir="Path/To/TrainingSet",
        test_dir="Path/To/TestingSet",
        splitted_dir="splitted_2500",
        csv_file="Glyphs.csv",
    )

    # Filter out specific classes if needed to simulate addition of new classes later
    dataset_utils.filter_classes = [134, 136]

    # Create COCO format dataset for both train/test splits
    stats = dataset_utils.create_coco_dataset_from_split(output_dir="coco_dataset_v1")

    # Print results
    print(stats)
