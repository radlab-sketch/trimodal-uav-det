"""
Configuration management for TriModalDet.
"""
import os
import argparse
import torch


def get_num_classes(label_dir):
    """
    Scans the label directory to find the highest class ID.

    Args:
        label_dir: Directory containing YOLO format label files

    Returns:
        int: Number of classes found (max_class_id + 1)
    """
    max_class_id = -1
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    if not label_files:
        print("Warning: No label files found. Cannot determine number of classes.")
        return 0

    for label_file in label_files:
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f.readlines():
                try:
                    class_id = int(line.strip().split()[0])
                    if class_id > max_class_id:
                        max_class_id = class_id
                except (ValueError, IndexError):
                    continue
    # Number of classes is max_class_id + 1 (for 0-based indexing)
    return max_class_id + 1


class Config:
    """Configuration class for TriModalDet."""

    def __init__(self):
        # Data paths
        self.data_root = 'data'
        self.image_dir = 'data/images'
        self.label_dir = 'data/labels'

        # Model configuration
        self.model_path = 'trimodaldet.pth'
        self.num_classes = None  # Will be set dynamically
        self.fpn_out_channels = 256

        # Input channels
        self.in_chans_rgb = 3
        self.in_chans_x = 2  # Thermal + Event

        # Backbone configuration
        self.backbone_type = 'mit_b1'  # SegFormer MiT-B1

        # Anchor generator
        self.anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        self.anchor_aspect_ratios = ((0.5, 1.0, 2.0),) * 5

        # RoI pooling
        self.roi_featmap_names = ['0', '1', '2', '3', 'pool']
        self.roi_output_size = 7
        self.roi_sampling_ratio = 2

        # Image normalization (5-channel)
        self.image_mean = [0.485, 0.456, 0.406, 0.5, 0.5]  # RGB + Thermal + Event
        self.image_std = [0.229, 0.224, 0.225, 0.5, 0.5]   # RGB + Thermal + Event

        # Training hyperparameters
        self.num_epochs = 15
        self.batch_size = 16
        self.learning_rate = 0.02
        self.momentum = 0.9
        self.weight_decay = 0.0001

        # Data split
        self.test_size = 0.2
        self.random_state = 42

        # Evaluation
        self.results_dir = 'test_results'
        self.score_threshold = 0.5

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def from_args(cls):
        """Create config from command line arguments."""
        parser = argparse.ArgumentParser(description="TriModalDet Training and Testing")
        parser.add_argument("--train", action="store_true", help="Run the training process.")
        parser.add_argument("--test", action="store_true", help="Run evaluation on the test set.")
        parser.add_argument("--vis", nargs='?', const=0, type=int,
                          help="Visualize a specific sample index from the full dataset and exit.")

        # Data arguments
        parser.add_argument("--data", type=str, default='data', help="Path to data directory")
        parser.add_argument("--model", type=str, default='trimodaldet.pth', help="Path to model checkpoint")

        # Model arguments
        parser.add_argument("--backbone", type=str, default='mit_b1',
                          choices=['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4'],
                          help="Backbone variant: mit_b0 (smallest/fastest), mit_b1 (default), "
                               "mit_b2 (base), mit_b3 (medium), mit_b4 (large)")

        # Training arguments
        parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
        parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
        parser.add_argument("--lr", type=float, default=0.02, help="Learning rate")

        # Evaluation arguments
        parser.add_argument("--results-dir", type=str, default='test_results', help="Directory to save results")

        args = parser.parse_args()

        # Create config and update from args
        config = cls()
        if args.data:
            config.data_root = args.data
            config.image_dir = os.path.join(args.data, 'images')
            config.label_dir = os.path.join(args.data, 'labels')
        if args.model:
            config.model_path = args.model
        if args.backbone:
            config.backbone_type = args.backbone
        if args.epochs:
            config.num_epochs = args.epochs
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.lr:
            config.learning_rate = args.lr
        if args.results_dir:
            config.results_dir = args.results_dir

        config.args = args
        return config

    def auto_detect_num_classes(self):
        """Automatically detect number of classes from label directory."""
        num_classes_from_data = get_num_classes(self.label_dir)
        # Add 1 for the background class, as required by FasterRCNN
        self.num_classes = num_classes_from_data + 1
        print(f"Dynamically determined {num_classes_from_data} classes. "
              f"Model configured for {self.num_classes} (including background).")
        return self.num_classes

    def __repr__(self):
        """String representation of config."""
        config_str = "Configuration:\n"
        for key, value in vars(self).items():
            if not key.startswith('_') and key != 'args':
                config_str += f"  {key}: {value}\n"
        return config_str


__all__ = ['Config', 'get_num_classes']
