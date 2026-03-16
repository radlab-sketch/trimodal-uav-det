#!/usr/bin/env python
"""
Visualization script for TriModalDet dataset samples.

Usage:
    python scripts/visualize.py --vis 0  # Visualize first sample
    python scripts/visualize.py --vis 5  # Visualize 6th sample
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trimodaldet.config import Config
from trimodaldet.data.dataset import NpyYoloDataset
from trimodaldet.utils.visualization import visualize_dataset_sample, VISUALIZATION_ENABLED


def main():
    # Load configuration from command line
    config = Config.from_args()

    if config.args.vis is None:
        print("Error: --vis flag is required for this script.")
        print("Usage: python scripts/visualize.py --vis 0")
        return

    print("=== TriModalDet Dataset Visualization ===")

    # Load dataset in 'all' mode (no train/test split)
    dataset = NpyYoloDataset(config.image_dir, config.label_dir, mode='all')

    frame_idx = config.args.vis

    if not VISUALIZATION_ENABLED:
        print("Could not visualize: Matplotlib is not installed.")
        return

    if len(dataset) == 0:
        print("Could not visualize: The dataset is empty.")
        return

    if frame_idx >= len(dataset):
        print(f"Error: Frame index {frame_idx} is out of bounds. "
              f"The dataset has {len(dataset)} samples.")
        return

    print(f"\nVisualizing sample at index {frame_idx} from the dataset...")
    image_tensor, target = dataset[frame_idx]
    visualize_dataset_sample(image_tensor, target)


if __name__ == '__main__':
    main()
