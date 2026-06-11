#!/usr/bin/env python -u
"""
Training script for modality ablation experiments with comprehensive logging.

Tests different combinations of input modalities (RGB, Thermal, Event) to understand
their individual and combined contributions to detection performance.

Usage:
    python trimodaldet/ablations/scripts/train_modality_ablation.py \
        --data ../RGBX_Semantic_Segmentation/data/images \
        --labels ../RGBX_Semantic_Segmentation/data/labels \
        --epochs 10 \
        --backbone mit_b1 \
        --modalities "rgb,thermal" \
        --output-dir results/modality_ablations/rgb_thermal
"""

import sys
import os
import argparse
from datetime import datetime
import time
import json
import csv

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

from trimodaldet.config import Config, get_num_classes
from trimodaldet.data.dataset import NpyYoloDataset
from trimodaldet.ablations.backbone_modality import ModalityConfigurableBackbone
from trimodaldet.models.encoder import get_encoder


class Logger:
    """Comprehensive logger for ablation experiments."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Log files
        self.training_log = os.path.join(output_dir, 'training.log')
        self.epoch_csv = os.path.join(output_dir, 'metrics_per_epoch.csv')
        self.batch_csv = os.path.join(output_dir, 'metrics_per_batch.csv')
        self.config_json = os.path.join(output_dir, 'config.json')
        self.results_json = os.path.join(output_dir, 'final_results.json')
        self.model_info_json = os.path.join(output_dir, 'model_info.json')
        self.eval_history_json = os.path.join(output_dir, 'evaluation_history.json')

        # Open training log
        self.log_file = open(self.training_log, 'w', buffering=1)

        # Initialize CSV files
        self._init_csv_files()

        # Initialize evaluation history
        self.eval_history = []

    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        with open(self.epoch_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'test_mAP', 'test_mAP_50', 'test_mAP_75',
                           'learning_rate', 'epoch_time_min', 'timestamp'])

        with open(self.batch_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'batch', 'loss_total', 'loss_classifier', 'loss_box_reg',
                           'loss_objectness', 'loss_rpn_box_reg', 'learning_rate',
                           'time_per_batch_sec', 'timestamp'])

    def log(self, message, print_to_console=True):
        """Log message to file and optionally print to console."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        self.log_file.write(log_msg + '\n')
        if print_to_console:
            print(log_msg)

    def log_header(self, title, width=60):
        """Log a header section."""
        self.log("=" * width)
        self.log(title.center(width))
        self.log("=" * width)

    def log_config(self, config_dict):
        """Log and save configuration."""
        with open(self.config_json, 'w') as f:
            json.dump(config_dict, f, indent=2)

        self.log_header("EXPERIMENT CONFIGURATION")
        for key, value in config_dict.items():
            self.log(f"  {key}: {value}")

    def log_model_info(self, model):
        """Log model architecture information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model_info = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total_params_M': total_params / 1e6,
            'trainable_params_M': trainable_params / 1e6
        }

        with open(self.model_info_json, 'w') as f:
            json.dump(model_info, f, indent=2)

        self.log_header("MODEL INFORMATION")
        self.log(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        self.log(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

        return model_info

    def log_epoch_metrics(self, epoch, train_loss, test_results, lr, epoch_time_min):
        """Log epoch-level metrics."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(self.epoch_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss,
                           test_results.get('mAP', 0.0) if test_results else 0.0,
                           test_results.get('mAP_50', 0.0) if test_results else 0.0,
                           test_results.get('mAP_75', 0.0) if test_results else 0.0,
                           lr, epoch_time_min, timestamp])

    def log_batch_metrics(self, epoch, batch, loss_dict, lr, time_per_batch):
        """Log batch-level metrics with detailed loss components."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Extract individual loss components
        total_loss = sum(loss for loss in loss_dict.values())
        loss_classifier = loss_dict.get('loss_classifier', 0.0)
        loss_box_reg = loss_dict.get('loss_box_reg', 0.0)
        loss_objectness = loss_dict.get('loss_objectness', 0.0)
        loss_rpn_box_reg = loss_dict.get('loss_rpn_box_reg', 0.0)

        with open(self.batch_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, batch, total_loss, loss_classifier, loss_box_reg,
                           loss_objectness, loss_rpn_box_reg, lr, time_per_batch, timestamp])

    def log_evaluation(self, epoch, eval_results):
        """Log evaluation results and add to history."""
        eval_entry = {
            'epoch': epoch,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': eval_results
        }
        self.eval_history.append(eval_entry)

        # Save evaluation history
        with open(self.eval_history_json, 'w') as f:
            json.dump(self.eval_history, f, indent=2)

    def log_final_results(self, results_dict):
        """Log final evaluation results."""
        with open(self.results_json, 'w') as f:
            json.dump(results_dict, f, indent=2)

        self.log_header("FINAL RESULTS")
        for key, value in results_dict.items():
            if isinstance(value, dict):
                self.log(f"  {key}:")
                for k, v in value.items():
                    self.log(f"    {k}: {v}")
            else:
                self.log(f"  {key}: {value}")

    def close(self):
        """Close log files."""
        if hasattr(self, 'log_file'):
            self.log_file.close()


class ModalityAblationTrainer:
    """Trainer for modality ablation experiments with comprehensive logging."""

    def __init__(self, config, active_modalities, logger):
        self.config = config
        self.active_modalities = active_modalities
        self.logger = logger

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.log(f"Using device: {self.device}")

        # Setup data loaders
        self.train_loader, self.test_loader = self.get_dataloaders()

        # Build model
        self.model = self.build_model()
        self.model.to(self.device)

        # Log model info
        self.model_info = self.logger.log_model_info(self.model)

        # Setup optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=0.0001
        )

        # Track training
        self.start_time = None
        self.best_loss = float('inf')

    def build_model(self):
        """Build Faster R-CNN model with modality-configurable backbone."""
        config = self.config

        self.logger.log(f"Building model with modality-configurable backbone: {config.backbone_type}")
        self.logger.log(f"  Active modalities: {self.active_modalities}")
        self.logger.log(f"  Using BASELINE architecture (MAGE+BiTE fusion, NOT GAFF/CSSA)")

        # Use baseline encoder with standard MAGE+BiTE fusion
        encoder_base = get_encoder(
            backbone_name=config.backbone_type,
            in_chans_rgb=config.in_chans_rgb,
            in_chans_x=config.in_chans_x
        )

        # Wrap with modality-configurable FPN
        backbone = ModalityConfigurableBackbone(
            encoder_base,
            fpn_out_channels=config.fpn_out_channels,
            active_modalities=self.active_modalities
        )
        self.logger.log(f"Backbone created. FPN output channels: {backbone.out_channels}")
        self.logger.log(f"Modality configuration: {backbone.get_modality_config()}")

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

        # Faster R-CNN
        model = FasterRCNN(
            backbone,
            num_classes=config.num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            image_mean=config.image_mean,
            image_std=config.image_std
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

        # Update num_classes
        if config.num_classes is None:
            config.num_classes = get_num_classes(config.labels_dir) + 1
            self.logger.log(f"Detected {config.num_classes} classes (including background)")

        # Collate function
        def collate_fn(batch):
            return tuple(zip(*batch))

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )

        self.logger.log(f"Dataset loaded:")
        self.logger.log(f"  Train: {len(train_dataset)} images")
        self.logger.log(f"  Test: {len(test_dataset)} images")
        self.logger.log(f"  Batches per epoch: {len(train_loader)}")

        return train_loader, test_loader

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        batch_times = []

        self.logger.log_header(f"EPOCH {epoch}/{self.config.epochs}")

        for i, (images, targets) in enumerate(self.train_loader):
            batch_start = time.time()

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
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # Log every 10 batches
            if (i + 1) % 10 == 0:
                avg_batch_time = sum(batch_times[-10:]) / len(batch_times[-10:])
                lr = self.optimizer.param_groups[0]['lr']

                # Create detailed loss string
                loss_details = " | ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])

                self.logger.log(
                    f"  Batch [{i+1}/{len(self.train_loader)}] | "
                    f"Total Loss: {losses.item():.4f} | "
                    f"{loss_details} | "
                    f"LR: {lr:.5f} | "
                    f"Time: {avg_batch_time:.2f}s/batch"
                )

                # Log to batch CSV with loss components
                self.logger.log_batch_metrics(epoch, i+1, loss_dict, lr, avg_batch_time)

        # Epoch summary
        avg_loss = epoch_loss / len(self.train_loader)
        epoch_time = (time.time() - epoch_start) / 60  # minutes
        lr = self.optimizer.param_groups[0]['lr']

        # Run evaluation on test set every 5 epochs and at the end
        test_results = None
        if epoch % 5 == 0 or epoch == self.config.epochs:
            self.logger.log("\nRunning mid-training evaluation on test set...")
            test_results = self.evaluate()
            self.logger.log(
                f"  Test mAP: {test_results['mAP']:.4f} | "
                f"mAP@50: {test_results['mAP_50']:.4f} | "
                f"mAP@75: {test_results['mAP_75']:.4f}"
            )
            # Log evaluation to history
            self.logger.log_evaluation(epoch, test_results)

        self.logger.log(
            f"Epoch {epoch} Complete | Avg Train Loss: {avg_loss:.4f} | "
            f"Time: {epoch_time:.2f} min"
        )

        # Log to epoch CSV
        self.logger.log_epoch_metrics(epoch, avg_loss, test_results, lr, epoch_time)

        return avg_loss

    def evaluate(self):
        """Evaluate model on test set with comprehensive metrics."""
        self.logger.log("Running comprehensive evaluation on test set...")
        self.model.eval()

        from torchmetrics.detection.mean_ap import MeanAveragePrecision
        metric = MeanAveragePrecision(iou_type="bbox")

        num_images = 0
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
                num_images += len(images)

        results = metric.compute()

        # Extract comprehensive metrics
        eval_results = {
            # Mean Average Precision
            'mAP': results['map'].item(),
            'mAP_50': results['map_50'].item(),
            'mAP_75': results['map_75'].item(),
            'mAP_small': results['map_small'].item(),
            'mAP_medium': results['map_medium'].item(),
            'mAP_large': results['map_large'].item(),
            # Mean Average Recall
            'mAR_1': results['mar_1'].item(),
            'mAR_10': results['mar_10'].item(),
            'mAR_100': results['mar_100'].item(),
            'mAR_small': results['mar_small'].item(),
            'mAR_medium': results['mar_medium'].item(),
            'mAR_large': results['mar_large'].item(),
            # Additional info
            'num_test_images': num_images
        }

        return eval_results

    def train(self):
        """Main training loop."""
        self.start_time = time.time()
        self.logger.log_header("TRAINING START")

        for epoch in range(1, self.config.epochs + 1):
            loss = self.train_epoch(epoch)

            # Save checkpoint every 5 epochs and at the end
            if epoch % 5 == 0 or epoch == self.config.epochs:
                # Save full checkpoint with training state
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    'best_loss': self.best_loss,
                    'config': {
                        'active_modalities': self.active_modalities,
                        'backbone': self.config.backbone_type,
                        'batch_size': self.config.batch_size,
                        'lr': self.config.lr
                    }
                }

                checkpoint_path = os.path.join(self.logger.output_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save(checkpoint, checkpoint_path)
                self.logger.log(f"Full checkpoint saved to {checkpoint_path}")

                # Also save latest checkpoint
                latest_path = os.path.join(self.logger.output_dir, 'checkpoint_latest.pth')
                torch.save(checkpoint, latest_path)

                # Save best model (model weights only for deployment)
                if loss < self.best_loss:
                    self.best_loss = loss
                    best_checkpoint_path = os.path.join(self.logger.output_dir, 'checkpoint_best.pth')
                    torch.save(checkpoint, best_checkpoint_path)

                    # Also save model weights only
                    best_weights_path = os.path.join(self.logger.output_dir, 'model_best_weights.pth')
                    torch.save(self.model.state_dict(), best_weights_path)
                    self.logger.log(f"Best checkpoint saved (loss: {loss:.4f})")

        # Training complete
        total_time = (time.time() - self.start_time) / 3600  # hours
        self.logger.log_header("TRAINING COMPLETE")
        self.logger.log(f"Total training time: {total_time:.2f} hours")
        self.logger.log(f"Best loss: {self.best_loss:.4f}")

        # Final evaluation on test set
        self.logger.log_header("FINAL EVALUATION")
        eval_results = self.evaluate()

        # Log final evaluation to history
        self.logger.log_evaluation(self.config.epochs, eval_results)

        # Compile final results with comprehensive information
        final_results = {
            'experiment_id': os.path.basename(self.logger.output_dir),
            'config': {
                'active_modalities': self.active_modalities,
                'modality_config': '+'.join(sorted(self.active_modalities)),
                'architecture': 'baseline_MAGE_BiTE',  # NOT GAFF or CSSA
                'backbone': self.config.backbone_type,
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.lr
            },
            'final_test_results': eval_results,
            'evaluation_history': self.logger.eval_history,
            'training': {
                'best_train_loss': self.best_loss,
                'total_time_hours': total_time,
                'total_epochs': self.config.epochs
            },
            'model': self.model_info,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'output_files': {
                'training_log': 'training.log',
                'metrics_per_epoch': 'metrics_per_epoch.csv',
                'metrics_per_batch': 'metrics_per_batch.csv',
                'evaluation_history': 'evaluation_history.json',
                'best_checkpoint': 'checkpoint_best.pth',
                'best_weights': 'model_best_weights.pth',
                'latest_checkpoint': 'checkpoint_latest.pth'
            }
        }

        self.logger.log_final_results(final_results)

        return final_results


def main():
    parser = argparse.ArgumentParser(description='Modality Ablation Training with Logging')

    # Dataset paths
    parser.add_argument('--data', type=str, required=True, help='Path to image data directory')
    parser.add_argument('--labels', type=str, required=True, help='Path to labels directory')

    # Training args
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for logs and checkpoints')
    parser.add_argument('--backbone', type=str, default='mit_b1',
                        choices=['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4'],
                        help='Backbone architecture')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')

    # Modality-specific args
    parser.add_argument('--modalities', type=str, default='rgb,thermal,event',
                        help='Comma-separated list of active modalities (e.g., "rgb,thermal" or "rgb,event")')

    args = parser.parse_args()

    # Parse modalities
    active_modalities = [m.strip().lower() for m in args.modalities.split(',')]

    # Validate modalities
    valid_modalities = {'rgb', 'thermal', 'event'}
    for m in active_modalities:
        if m not in valid_modalities:
            raise ValueError(f"Invalid modality '{m}'. Must be one of {valid_modalities}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize logger
    logger = Logger(args.output_dir)

    # Create config
    config = Config()
    config.data_dir = args.data
    config.labels_dir = args.labels
    config.backbone_type = args.backbone
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr

    # Log configuration
    config_dict = {
        'experiment_id': os.path.basename(args.output_dir),
        'data_dir': args.data,
        'labels_dir': args.labels,
        'output_dir': args.output_dir,
        'architecture': 'baseline_MAGE_BiTE',  # NOT GAFF or CSSA ablations
        'backbone': args.backbone,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'active_modalities': active_modalities,
        'modality_config': '+'.join(sorted(active_modalities)),
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    logger.log_config(config_dict)

    # Create trainer and run
    try:
        trainer = ModalityAblationTrainer(
            config,
            active_modalities=active_modalities,
            logger=logger
        )
        results = trainer.train()

        logger.log_header("EXPERIMENT COMPLETE")
        logger.log(f"Results saved to: {args.output_dir}")

    except Exception as e:
        logger.log(f"ERROR: {str(e)}")
        logger.log("Experiment failed!")
        raise
    finally:
        logger.close()


if __name__ == '__main__':
    main()
