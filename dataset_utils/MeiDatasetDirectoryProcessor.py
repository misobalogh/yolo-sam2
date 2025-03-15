# Michal Balogh, 2025

import os
import sys


class MeiDatasetDirectoryProcessor:
    def __init__(self, splitted_dir, image_id_start=1, annotation_id_start=1):
        self.splitted_dir = splitted_dir
        self.image_id = image_id_start
        self.annotation_id = annotation_id_start

    def process_images(self, root_dir, callback, *args, **kwargs):
        """
        Process all images in dataset directory structure using the provided callback function
        """
        for image_dir in os.listdir(root_dir):
            image_path = os.path.join(root_dir, image_dir)
            if not self._is_valid_dir(image_path):
                continue

            splitted_path = os.path.join(image_path, self.splitted_dir)
            if not self._is_valid_dir(splitted_path):
                continue

            for sub_image in os.listdir(splitted_path):
                sub_image_path = os.path.join(splitted_path, sub_image)
                if not self._is_valid_dir(sub_image_path):
                    continue

                callback(sub_image_path, *args,
                        image_id=self.image_id,
                        annotation_id=self.annotation_id,
                        **kwargs)

                self.image_id += 1
                self.annotation_id += 1

    @staticmethod
    def _is_valid_dir(path: str) -> bool:
        if not os.path.isdir(path):
            print(f"Path not found or not a directory: {path}", file=sys.stderr)
            return False
        return True