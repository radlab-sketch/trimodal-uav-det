#!/usr/bin/env python
"""
Training script for TriModalDet.

Usage:
    python scripts/train.py --data data/ --epochs 15 --batch-size 16
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trimodaldet.config import Config
from trimodaldet.training.trainer import Trainer, CudaOutOfMemoryError
from trimodaldet.training.monitor_utils import analyze_and_recommend
from scripts.monitor import ResourceMonitor


def main():
    # Load configuration from command line
    config = Config.from_args()

    print("=== TriModalDet Training ===")
    print(config)

    # Ensure results directory exists
    os.makedirs(config.results_dir, exist_ok=True)

    # Initialize resource monitor
    monitor = ResourceMonitor(
        interval=config.monitor_interval,
        output=config.monitor_output
    )

    # Create trainer
    trainer = Trainer(config)

    # Start monitoring
    monitor.start()

    try:
        # Run training
        trainer.train()
    except CudaOutOfMemoryError as e:
        print(f"\n[Training Aborted] {e}")
        # Save emergency checkpoint
        emergency_path = os.path.join(config.results_dir, 'checkpoint_emergency.pth')
        trainer.save_emergency_checkpoint(emergency_path)
    except KeyboardInterrupt:
        print("\n[Training Interrupted] Keyboard interrupt detected.")
        # Save emergency checkpoint
        emergency_path = os.path.join(config.results_dir, 'checkpoint_emergency.pth')
        trainer.save_emergency_checkpoint(emergency_path)
    except Exception as e:
        print(f"\n[Training Error] {e}")
        raise
    finally:
        # Stop monitoring and print summary
        monitor.stop()
        monitor.print_summary()
        # Print optimization recommendations
        analyze_and_recommend(
            monitor.records,
            batch_size=config.batch_size,
            backbone=config.backbone_type
        )


if __name__ == '__main__':
    main()
