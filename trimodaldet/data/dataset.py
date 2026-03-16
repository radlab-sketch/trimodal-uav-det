"""
Dataset classes for loading multi-channel images with YOLO format labels.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import yolo_to_coco

# Try to import sklearn for train/test splitting
try:
    from sklearn.model_selection import train_test_split
    SPLIT_ENABLED = True
except ImportError:
    print("scikit-learn not found. Train/test split disabled.")
    SPLIT_ENABLED = False
    train_test_split = None


class NpyYoloDataset(Dataset):
    """
    PyTorch Dataset for loading 5-channel NPY images and YOLO format labels.

    The dataset loads .npy files containing (H, W, 5) images where:
    - Channels 0-2: RGB
    - Channel 3: Thermal
    - Channel 4: Event

    Labels are in YOLO format (.txt files):
    class_id x_center y_center width height (all normalized 0-1)

    Args:
        image_dir: Directory containing .npy image files
        label_dir: Directory containing .txt label files
        mode: 'train', 'test', or 'all' (for visualization)
        test_size: Fraction of data to use for testing (if sklearn available)
        random_state: Random seed for train/test split
    """

    def __init__(self, image_dir, label_dir, mode='train', test_size=0.2, random_state=42):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.mode = mode

        # Find all valid image files (have corresponding non-empty labels)
        all_valid_files = []
        all_image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])
        for img_file in all_image_files:
            label_path = os.path.join(self.label_dir, img_file.replace('.npy', '.txt'))
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    if f.read().strip():
                        all_valid_files.append(img_file)

        if not all_valid_files:
            self.image_files = []
            return

        # Split into train/test if sklearn is available and mode is train/test
        if SPLIT_ENABLED and self.mode in ['train', 'test']:
            train_files, test_files = train_test_split(
                all_valid_files, test_size=test_size, random_state=random_state
            )
            if self.mode == 'train':
                self.image_files = train_files
            else:
                self.image_files = test_files
            print(f"Loaded {len(self.image_files)} files for {self.mode} mode.")
        else:
            self.image_files = all_valid_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = np.load(img_path)
        image_tensor = torch.from_numpy(image).float()

        # Permute from (H, W, C) to (C, H, W) if necessary
        if image_tensor.ndim == 3 and image_tensor.shape[-1] == 5:
            image_tensor = image_tensor.permute(2, 0, 1)

        # Get the actual height and width from the image tensor
        _, img_height, img_width = image_tensor.shape

        # Load labels
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace('.npy', '.txt'))
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    try:
                        parts = line.strip().split()
                        # Shift from 0-indexed (YOLO) to 1-indexed (FasterRCNN)
                        class_id = int(parts[0]) + 1
                        yolo_box = [float(p) for p in parts[1:]]

                        coco_box = yolo_to_coco(yolo_box, img_width, img_height)
                        boxes.append(coco_box)
                        labels.append(class_id)
                    except (ValueError, IndexError):
                        continue

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)

        return image_tensor, target


def collate_fn(batch):
    """
    Custom collate function for object detection.

    Returns:
        Tuple of (images, targets) where images and targets are lists
    """
    return tuple(zip(*batch))


__all__ = ['NpyYoloDataset', 'collate_fn']
