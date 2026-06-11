"""
Evaluation logic for TriModalDet.
"""
import torch
from torch.utils.data import DataLoader

from .trainer import Trainer
from ..data.dataset import NpyYoloDataset, collate_fn
from ..utils.metrics import evaluate


class Evaluator:
    """
    Evaluator class for TriModalDet.

    Args:
        config: Configuration object
    """

    def __init__(self, config):
        self.config = config

        # Auto-detect number of classes
        config.auto_detect_num_classes()

        # Setup test dataset and dataloader
        print("\n--- TriModalDet Evaluation ---")
        print("\n1. Setting up test dataset...")
        self.test_dataset = NpyYoloDataset(
            config.image_dir,
            config.label_dir,
            mode='test',
            test_size=config.test_size,
            random_state=config.random_state
        )

        print("\n2. Initializing DataLoader...")
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        # Build model
        print("\n3. Building model...")
        trainer = Trainer.__new__(Trainer)  # Create without __init__
        trainer.config = config
        self.model = trainer.build_model()
        self.model.to(config.device)

    def load_checkpoint(self, path=None):
        """Load model checkpoint."""
        if path is None:
            path = self.config.model_path

        print(f"\n4. Loading model from {path}...")
        checkpoint = torch.load(path, map_location=self.config.device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # Full training checkpoint
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                # Alternative format
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume the dict IS the state dict
                self.model.load_state_dict(checkpoint)
        else:
            # Direct state dict
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print("Model loaded successfully.")

    def evaluate(self):
        """Run evaluation on test set."""
        print("\n5. Running evaluation...")
        results = evaluate(
            self.model,
            self.test_loader,
            self.config.device,
            results_dir=self.config.results_dir
        )
        return results


__all__ = ['Evaluator']
