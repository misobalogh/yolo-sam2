# Michal Balogh, 2025

from MeiDatasetDirectoryProcessor import MeiDatasetDirectoryProcessor


import pandas as pd


import os


class HeightMapDatabase:
    """Base class for dataset utilities with common methods."""
    def __init__(
        self,
        train_dir,
        test_dir,
        splitted_dir,
        csv_file = "Glyphs.csv",
        output_path = ".",
        labels_txt = "labels.txt"
        ):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.splitted_dir = splitted_dir
        self.csv_file = csv_file
        self.directory_processor = MeiDatasetDirectoryProcessor(splitted_dir)
        self.num_imgs = 0
        self.split = {
            "train": self.train_dir,
            "test": self.test_dir
        }
        self.output_path = output_path
        self.labels_txt = labels_txt
        self.labels = set()
        self.id2label_dict = {}
        self.id2label_coco_dict = {}
        self.unicode2id_map = None

    def __repr__(self):
        pass

    def dataset_stats(self):
        img_count_train = self.get_image_count("train")
        img_count_test = self.get_image_count("test")
        labels_train = self.get_all_labels(split="train", save_to_file=True, file="labels_train.txt")
        labels_test = self.get_all_labels(split="test", save_to_file=True, file="labels_test.txt")
        num_labels_train = len(labels_train)
        num_labels_test = len(labels_test)
        id2label = self.id2label()

        return {
            "train": {
                "img_count": img_count_train,
                "labels": labels_train,
                "num_labels": num_labels_train,
                },
            "test": {
                "img_count": img_count_test,
                "labels": labels_test,
                "num_labels": num_labels_test,
            },
            "id2label": id2label
        }

    def get_split_dir(self, split="train"):
        if (split not in ["train", "test"]):
            print('Please provide valid split: "train" or "test"')
        return self.split[split]

    def get_image_count(self, split="train"):
        dir = self.get_split_dir(split)
        self.num_imgs = 0
        self.directory_processor.process_images(
            dir,
            self._increase_count,
        )

        return self.num_imgs

    def _increase_count(self, image_path, **kwargs):
        self.num_imgs += 1

    def get_all_labels(self, split="train", save_to_file=False, file=None):
        file_path = os.path.join(self.output_path, file if file else self.labels_txt)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                for line in f:
                    self.labels.add(int(line.strip()))

            self.labels = set(self.labels)
            return self.labels

        dir = self.get_split_dir(split)
        self.directory_processor.process_images(
            dir,
            self._get_labels_from_csv
        )

        self.labels = set(self.labels)

        if save_to_file:
            with open(file_path, "w") as f:
                for label in self.labels:
                    f.write(f"{label}\n")

        return self.labels

    def _get_labels_from_csv(self, image_path, *args, **kwargs):
        csv_file_path = os.path.join(image_path, self.csv_file)
        if os.path.exists(csv_file_path):
            with open(csv_file_path, 'r') as f:
                df = pd.read_csv(f)
                unique_values = df["Unicode"].unique()
                for value in unique_values:
                    self.labels.add(value)
        else:
            print(f"CSV file not found: {csv_file_path}")

    def get_frequency_of_labels(self, split="train", save_to_file=False, file_prefix=None):
        """
        Get the frequency of labels from the dataset.
        Args:
            split: Which dataset split to use ("train" or "test")
            save_to_file: Whether to save the frequency counts to files
            file_prefix: Prefix for the output files (default: "frequency_")

        Returns:
            Tuple of two dictionaries:
            - label_counts: Total occurrences of each label
            - label_counts_unique: Number of images where each label appears
        """
        self.label_counts = {}
        self.label_counts_unique = {}

        dir = self.get_split_dir(split)
        self.directory_processor.process_images(
            dir,
            self._get_unique_labels_from_csv
        )

        if save_to_file:
            prefix = file_prefix if file_prefix else "frequency_"
            # Sort by label value
            sorted_by_label = dict(sorted(self.label_counts.items(), key=lambda x: x[0]))
            # Sort by occurrence count
            sorted_by_occurrences = dict(sorted(self.label_counts.items(), key=lambda x: x[1], reverse=True))
            # Sort unique counts by label value
            unique_sorted_by_label = dict(sorted(self.label_counts_unique.items(), key=lambda x: x[0]))
            # Sort unique counts by occurrence count
            unique_sorted_by_occurrences = dict(sorted(self.label_counts_unique.items(), key=lambda x: x[1], reverse=True))

            # Save all sorted versions
            self._save_frequency_to_file(f"{prefix}sorted_by_label.txt", sorted_by_label)
            self._save_frequency_to_file(f"{prefix}sorted_by_occurrences.txt", sorted_by_occurrences)
            self._save_frequency_to_file(f"{prefix}unique_per_image_sorted_by_label.txt", unique_sorted_by_label)
            self._save_frequency_to_file(f"{prefix}unique_per_image_sorted_by_occurrences.txt", unique_sorted_by_occurrences)

        return self.label_counts, self.label_counts_unique

    def _get_unique_labels_from_csv(self, image_path, *args, **kwargs):
        """
        Process a single image directory to count label frequencies
        - label_counts: Counts every occurrence of each label
        - label_counts_unique: Counts each label only once per image
        """
        csv_file_path = os.path.join(image_path, self.csv_file)
        if os.path.exists(csv_file_path):
            with open(csv_file_path, 'r') as f:
                df = pd.read_csv(f)
                # Count all occurrences
                for label in df["Unicode"]:
                    self.label_counts[label] = self.label_counts.get(label, 0) + 1
                # Count unique occurrences per image
                for label in df["Unicode"].unique():
                    self.label_counts_unique[label] = self.label_counts_unique.get(label, 0) + 1
        else:
            print(f"CSV file not found: {csv_file_path}")

    def _save_frequency_to_file(self, filename, frequency_dict):
        """Save label frequency dictionary to file"""
        filepath = os.path.join(self.output_path, filename)
        with open(filepath, "w") as f:
            for label, freq in frequency_dict.items():
                f.write(f"{label}: {freq}\n")
        print(f"Saved label frequency data to {filepath}")

    def id2label(self, labels=None):
        if not labels and not self.labels:
            self.get_all_labels()

        sorted_labels = sorted(self.labels)
        label_names = [chr(label) if chr(label) <= 'z' else f'speical_symbol_{label}' for label in sorted_labels]
        self.id2label_dict = {0: "tire background"}
        self.id2label_dict.update({i+1: label for i, label in enumerate(label_names)})

        return self.id2label_dict

    def id2label_coco(self):
        if not self.labels:
            self.get_all_labels()
        sorted_labels = sorted(self.labels)
        label_names = [chr(label) if chr(label) <= 'z' else f'speical_symbol_{label}' for label in sorted_labels]
        self.id2label_dict = {0: {"isthing": 0, "name": "tire background"}}
        self.id2label_dict.update({i+1: {"isthing": 1, "name": label} for i, label in enumerate(label_names)})

        return self.id2label_dict

    def get_unicode_to_id_mapping(self, force_refresh=False):
        """
        Create a mapping from Unicode values to class IDs.
        Assigns each unique label a sequential ID (starting from 1).

        Args:
            force_refresh: If True, recreate mapping even if it exists

        Returns:
            Dictionary mapping Unicode values to class IDs
        """
        if self.unicode2id_map is not None and not force_refresh:
            return self.unicode2id_map

        if not self.labels:
            self.get_all_labels()

        labels = sorted(list(self.labels))
        self.unicode2id_map = {label: i+1 for i, label in enumerate(labels)}
        return self.unicode2id_map

    def get_class_id(self, unicode_val):
        """
        Get class ID for a given Unicode value

        Args:
            unicode_val: The Unicode value to look up

        Returns:
            Integer class ID
        """
        if self.unicode2id_map is None:
            self.get_unicode_to_id_mapping()

        return self.unicode2id_map.get(unicode_val, 0)  # Return 0 (background) if not found