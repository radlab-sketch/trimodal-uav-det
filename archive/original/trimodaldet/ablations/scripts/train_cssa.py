#!/usr/bin/env python -u
"""
Training script for CSSA fusion ablation experiments.

This script trains TriModalDet with CSSA fusion at Stage 4,
keeping all other components identical to the baseline.

Usage:
    python trimodaldet/ablations/scripts/train_cssa.py \
        --data data/ \
        --epochs 15 \
        --backbone mit_b1 \
        --cssa-thresh 0.5

For 5-epoch sanity check:
    python trimodaldet/ablations/scripts/train_cssa.py \
        --data data/ \
        --epochs 15 \
        --backbone mit_b1 \
        --model checkpoints/cssa_stage4_sanity.pth
"""

import sys
import os
import argparse
from datetime import datetime

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from trimodaldet.config import Config, get_num_classes
from trimodaldet.data.dataset import NpyYoloDataset
from trimodaldet.models.backbone import InterModalBackbone
from trimodaldet.training.evaluator import Evaluator
from trimodaldet.ablations.encoder_cssa import get_encoder_cssa


class CSSATrainer:
    """
    Trainer for CSSA ablation experiments.

    Modified from the original Trainer to use CSSA-enabled encoder.
    """

    def __init__(self, config, cssa_switching_thresh=0.5, cssa_kernel_size=3):
        self.config = config
        self.cssa_switching_thresh = cssa_switching_thresh
        self.cssa_kernel_size = cssa_kernel_size

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Setup data loaders FIRST to get num_classes
        self.train_loader, self.test_loader = self.get_dataloaders()

        # Build model with CSSA encoder (now that num_classes is set)
        self.model = self.build_model_cssa()
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=0.0001
        )

        # Note: Evaluator will be created when needed for evaluation
        # since it creates its own dataloader

    def build_model_cssa(self):
        """Build TriModalDet model with CSSA fusion at Stage 4."""
        config = self.config

        # CSSA-enabled encoder
        print(f"Using CSSA-enabled backbone: {config.backbone_type}")
        print(f"  CSSA switching threshold: {self.cssa_switching_thresh}")
        print(f"  CSSA kernel size: {self.cssa_kernel_size}")

        encoder_base = get_encoder_cssa(
            config.backbone_type,
            in_chans_rgb=config.in_chans_rgb,
            in_chans_x=config.in_chans_x,
            cssa_switching_thresh=self.cssa_switching_thresh,
            cssa_kernel_size=self.cssa_kernel_size
        )

        # Wrap with FPN
        backbone = InterModalBackbone(encoder_base, fpn_out_channels=config.fpn_out_channels)
        print(f"Backbone created. FPN output channels: {backbone.out_channels}")

        # Anchor generator
        anchor_generator = AnchorGenerator(
            sizes=config.anchor_sizes,
            aspect_ratios=config.anchor_aspect_ratios
        )

        # RoI pooler
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=config.roi_featmap_names,
            output_size=config.roi_output_size,
            sampling_ratio=config.roi_sampling_ratio
        )

        # Faster R-CNN with 5-channel normalization from config
        model = FasterRCNN(
            backbone,
            num_classes=config.num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            image_mean=config.image_mean,  # 5 channels from config
            image_std=config.image_std      # 5 channels from config
        )

        return model

    def get_dataloaders(self):
        """Create train and test dataloaders."""
        config = self.config

        # Training dataset
        train_dataset = NpyYoloDataset(
            image_dir=config.data_dir,
            label_dir=config.labels_dir,
            mode='train'
        )

        # Test dataset
        test_dataset = NpyYoloDataset(
            image_dir=config.data_dir,
            label_dir=config.labels_dir,
            mode='test'
        )

        # Update num_classes in config based on dataset
        if config.num_classes is None:
            config.num_classes = get_num_classes(config.labels_dir) + 1  # +1 for background class
            print(f"Detected {config.num_classes} classes from dataset (including background)")

        # Collate function
        def collate_fn(batch):
            return tuple(zip(*batch))

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid worker process warnings
            collate_fn=collate_fn
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid worker process warnings
            collate_fn=collate_fn
        )

        print(f"Train dataset: {len(train_dataset)} images")
        print(f"Test dataset: {len(test_dataset)} images")

        return train_loader, test_loader

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0

        for i, (images, targets) in enumerate(self.train_loader):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            epoch_loss += losses.item()

            # Print progress
            if (i + 1) % 10 == 0:
                print(f"  Batch [{i+1}/{len(self.train_loader)}], Loss: {losses.item():.4f}")

        avg_loss = epoch_loss / len(self.train_loader)
        print(f"Epoch [{epoch}] Average Loss: {avg_loss:.4f}")

        return avg_loss

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("CSSA ABLATION TRAINING")
        print("=" * 60)
        print(f"Backbone: {self.config.backbone_type}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.lr}")
        print(f"Checkpoint path: {self.config.model_path}")
        print("=" * 60 + "\n")

        for epoch in range(1, self.config.epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.config.epochs}")
            print(f"{'='*60}")

            loss = self.train_epoch(epoch)

            # Save checkpoint every 5 epochs
            if epoch % 5 == 0 or epoch == self.config.epochs:
                torch.save(self.model.state_dict(), self.config.model_path)
                print(f"✓ Checkpoint saved to {self.config.model_path}")

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

        # Final evaluation
        print("\nRunning final evaluation on test set...")
        self.model.eval()

        from torchmetrics.detection.mean_ap import MeanAveragePrecision
        metric = MeanAveragePrecision(iou_type="bbox")

        with torch.no_grad():
            for images, targets in self.test_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                predictions = self.model(images)

                # Convert to format expected by torchmetrics
                preds = []
                for pred in predictions:
                    preds.append({
                        'boxes': pred['boxes'].cpu(),
                        'scores': pred['scores'].cpu(),
                        'labels': pred['labels'].cpu()
                    })

                tgts = []
                for tgt in targets:
                    tgts.append({
                        'boxes': tgt['boxes'].cpu(),
                        'labels': tgt['labels'].cpu()
                    })

                metric.update(preds, tgts)

        results = metric.compute()

        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"mAP: {results['map'].item():.4f}")
        print(f"mAP@50: {results['map_50'].item():.4f}")
        print(f"mAP@75: {results['map_75'].item():.4f}")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='CSSA Fusion Ablation Training')

    # Dataset path - can use either --dataset or --data/--labels
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to dataset directory (containing images/ and labels/ subdirs)')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to image data directory (alternative to --dataset)')
    parser.add_argument('--labels', type=str, default=None,
                        help='Path to labels directory (alternative to --dataset)')

    # Other training args
    parser.add_argument('--model', type=str, default='checkpoints/cssa_stage4.pth',
                        help='Path to save model checkpoint')
    parser.add_argument('--backbone', type=str, default='mit_b1',
                        choices=['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4'],
                        help='Backbone architecture')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='Learning rate')
    parser.add_argument('--results-dir', type=str, default='test_results',
                        help='Directory for evaluation results')

    # CSSA-specific args
    parser.add_argument('--cssa-thresh', type=float, default=0.5,
                        help='CSSA channel switching threshold (0.0-1.0)')
    parser.add_argument('--cssa-kernel', type=int, default=3,
                        help='CSSA ECA kernel size (odd number)')

    args = parser.parse_args()

    # Handle dataset path
    if args.dataset:
        # Use dataset/images and dataset/labels
        data_dir = os.path.join(args.dataset, 'images')
        labels_dir = os.path.join(args.dataset, 'labels')
        print(f"Using dataset directory: {args.dataset}")
        print(f"  Images: {data_dir}")
        print(f"  Labels: {labels_dir}")
    elif args.data and args.labels:
        # Use explicit paths
        data_dir = args.data
        labels_dir = args.labels
    else:
        # Default fallback
        data_dir = 'data/images'
        labels_dir = 'data/labels'
        print(f"Warning: No dataset path specified. Using defaults: {data_dir}, {labels_dir}")

    # Create config from args
    config = Config()
    config.data_dir = data_dir
    config.labels_dir = labels_dir
    config.model_path = args.model
    config.backbone_type = args.backbone
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.results_dir = args.results_dir

    # Print configuration
    print("=== CSSA Ablation Configuration ===")
    print(config)
    print(f"CSSA Threshold: {args.cssa_thresh}")
    print(f"CSSA Kernel Size: {args.cssa_kernel}")
    print("=" * 60 + "\n")

    # Create trainer and run
    trainer = CSSATrainer(
        config,
        cssa_switching_thresh=args.cssa_thresh,
        cssa_kernel_size=args.cssa_kernel
    )
    trainer.train()


if __name__ == '__main__':
    main()
