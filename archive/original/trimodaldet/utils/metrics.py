"""
Evaluation metrics for object detection.
"""
import os
import torch
from pprint import pprint

from .visualization import visualize_evaluation_sample

# Try to import torchmetrics
try:
    from torchmetrics.detection import MeanAveragePrecision
    METRICS_ENABLED = True
except ImportError:
    print("torchmetrics not found. Metrics computation disabled.")
    METRICS_ENABLED = False
    MeanAveragePrecision = None


def evaluate(model, data_loader, device, results_dir='test_results'):
    """
    Computes and prints mAP metrics for the model on the given dataset.

    Args:
        model: Trained model
        data_loader: DataLoader with test data
        device: Device to run evaluation on
        results_dir: Directory to save visualization results

    Returns:
        dict: Computed metrics (if torchmetrics available)
    """
    if not METRICS_ENABLED:
        print("Cannot evaluate: torchmetrics is not installed.")
        return None

    # Create directory for results
    os.makedirs(results_dir, exist_ok=True)
    print(f"Saving evaluation visualizations to '{results_dir}/'")

    metric = MeanAveragePrecision(box_format='xyxy')
    model.eval()

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get model predictions
            predictions = model(images)

            # Update metric
            metric.update(predictions, targets)

            # Save visualization for each image in the batch
            for j in range(len(images)):
                img_tensor = images[j]
                gt_target = targets[j]
                pred = predictions[j]
                output_path = os.path.join(results_dir, f"result_batch_{i}_img_{j}.png")
                visualize_evaluation_sample(img_tensor, gt_target, pred, output_path)

    print("\n--- Evaluation Results ---")
    results = metric.compute()
    pprint(results)
    print("--------------------------")

    return results


__all__ = ['evaluate', 'METRICS_ENABLED']
