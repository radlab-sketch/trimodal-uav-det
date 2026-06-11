"""
Visualization utilities for multi-modal object detection.
"""
import numpy as np
import torch

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    VISUALIZATION_ENABLED = True
except ImportError:
    print("matplotlib not found. Visualization disabled.")
    VISUALIZATION_ENABLED = False
    plt = None
    patches = None


def visualize_dataset_sample(image_tensor, target, output_path='dataset_sample.png'):
    """
    Displays a sample from the dataset, showing RGB, Thermal, and Event modalities side-by-side.

    Args:
        image_tensor (torch.Tensor): A single 5-channel image tensor (C, H, W).
        target (dict): The corresponding target dictionary with 'boxes' and 'labels'.
        output_path (str): Path to save the visualization
    """
    if not VISUALIZATION_ENABLED:
        print("Cannot visualize: Matplotlib is not installed.")
        return

    # Ensure tensor is on CPU and in numpy format for visualization
    image_tensor = image_tensor.cpu()
    boxes = target['boxes'].cpu().numpy()

    # Unpack the 5 channels
    # Permute RGB from (C, H, W) to (H, W, C) for displaying
    rgb_img = image_tensor[:3].permute(1, 2, 0).numpy()
    thermal_img = image_tensor[3].numpy()
    event_img = image_tensor[4].numpy()

    # Normalize RGB for display purposes to ensure it's in the [0, 1] range
    rgb_min, rgb_max = np.min(rgb_img), np.max(rgb_img)
    if rgb_max > rgb_min:
        rgb_img = (rgb_img - rgb_min) / (rgb_max - rgb_min)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle("Dataset Sample Visualization", fontsize=16)

    # Display RGB Image
    ax1.imshow(np.clip(rgb_img, 0, 1)) # Clip values as a final safety measure
    ax1.set_title('RGB')
    ax1.axis('off')

    # Display Thermal Image
    ax2.imshow(thermal_img, cmap='gray')
    ax2.set_title('Thermal')
    ax2.axis('off')

    # Display Event Image
    ax3.imshow(event_img, cmap='gray')
    ax3.set_title('Event')
    ax3.axis('off')

    # Draw bounding boxes on all three images
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        width, height = xmax - xmin, ymax - ymin

        # Create a rectangle patch
        rect1 = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        rect2 = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        rect3 = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax1.add_patch(rect1)
        ax2.add_patch(rect2)
        ax3.add_patch(rect3)

    plt.savefig(output_path)
    plt.close(fig)  # Close the figure to free up memory
    print(f"Visualization saved to {output_path}")


def visualize_evaluation_sample(image_tensor, gt_target, prediction, output_path, score_threshold=0.5):
    """
    Saves a visualization of a single sample from evaluation, showing GT and predicted boxes.

    Args:
        image_tensor: Image tensor (C, H, W)
        gt_target: Ground truth target dict
        prediction: Prediction dict from model
        output_path: Path to save visualization
        score_threshold: Minimum score for predicted boxes to display
    """
    if not VISUALIZATION_ENABLED:
        return

    # Prepare data for visualization
    image_tensor = image_tensor.cpu()
    gt_boxes = gt_target.get('boxes', torch.tensor([])).cpu().numpy()
    pred_boxes = prediction.get('boxes', torch.tensor([])).cpu().numpy()
    pred_scores = prediction.get('scores', torch.tensor([])).cpu().numpy()

    # Filter predictions by score
    high_conf_preds = pred_scores >= score_threshold
    pred_boxes = pred_boxes[high_conf_preds]

    # Unpack image channels and prepare for plotting
    rgb_img = image_tensor[:3].permute(1, 2, 0).numpy()
    thermal_img = image_tensor[3].numpy()
    event_img = image_tensor[4].numpy()

    rgb_min, rgb_max = np.min(rgb_img), np.max(rgb_img)
    if rgb_max > rgb_min:
        rgb_img = (rgb_img - rgb_min) / (rgb_max - rgb_min)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle("Ground Truth (Green) vs. Prediction (Red)", fontsize=16)

    # Display images
    ax1.imshow(np.clip(rgb_img, 0, 1)); ax1.set_title('RGB'); ax1.axis('off')
    ax2.imshow(thermal_img, cmap='gray'); ax2.set_title('Thermal'); ax2.axis('off')
    ax3.imshow(event_img, cmap='gray'); ax3.set_title('Event'); ax3.axis('off')

    # Draw ground-truth boxes (Green)
    for box in gt_boxes:
        xmin, ymin, xmax, ymax = box
        for ax in [ax1, ax2, ax3]:
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

    # Draw predicted boxes (Red)
    for box in pred_boxes:
        xmin, ymin, xmax, ymax = box
        for ax in [ax1, ax2, ax3]:
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    plt.savefig(output_path)
    plt.close(fig)


__all__ = ['visualize_dataset_sample', 'visualize_evaluation_sample', 'VISUALIZATION_ENABLED']
