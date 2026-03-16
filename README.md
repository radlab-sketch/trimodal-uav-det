# TriModalDet

Tri-modal object detection using RGB, thermal, and event camera data.

## Overview

TriModalDet is a tri-modal object detection framework that processes RGB, thermal, and event camera inputs through MAGE+BiTE fusion mechanisms. The architecture combines hierarchical feature extraction with modality-aware gating to improve detection performance across varying environmental conditions.

## Architecture

The framework consists of three main components:

**Inter-Modal Backbone**
- Hierarchical inter-sensor backbone based on SegFormer MiT (Mix Transformer)
- Dual-stream architecture with separate pathways for RGB and auxiliary modalities (thermal, event)
- Configurable backbone variants (mit_b0 through mit_b4) for different speed/accuracy trade-offs
- Multi-scale feature extraction through 4 transformer stages

**Tri-Modal Fusion**
- Modality-Aware Gated Exchange (MAGE) for adaptive modality weighting with channel and spatial gating
- Bidirectional Token Exchange (BiTE) with inter-modal attention mechanisms
- Stage-wise fusion enabling cross-modality information exchange at multiple scales

**Detection Head**
- Feature Pyramid Network (FPN) for multi-scale feature aggregation
- Region Proposal Network (RPN) for object proposals
- Faster R-CNN detection head for classification and localization

## Installation

Requirements:
- Python 3.7 or higher
- PyTorch 1.7.0 or higher
- CUDA 10.2 or higher (for GPU training)

Setup:
```bash
cd trimodaldet
pip install -r requirements.txt
```

## Data Format

### Input Images

Images must be 5-channel NumPy arrays saved as .npy files with shape (H, W, 5):
- Channels 0-2: RGB
- Channel 3: Thermal or infrared
- Channel 4: Event camera data

Example:
```python
import numpy as np

image = np.zeros((480, 640, 5), dtype=np.uint8)
image[:, :, 0:3] = rgb_data
image[:, :, 3] = thermal_data
image[:, :, 4] = event_data

np.save('data/images/frame_001.npy', image)
```

### Annotations

Labels use YOLO format (one line per object):
```
class_id x_center y_center width height
```

All coordinates are normalized to [0, 1].

Example (data/labels/frame_001.txt):
```
0 0.5 0.5 0.3 0.4
1 0.2 0.7 0.15 0.2
```

### Directory Structure

Users must create the following directories before training:
```
data/
├── images/    (place .npy files here)
└── labels/    (place .txt files here)
```

## Usage

### Training

```bash
python scripts/train.py --data data/ --epochs 15 --batch-size 2
```

Optional arguments:
- `--backbone`: Backbone variant (default: mit_b1)
  - `mit_b0`: Smallest, fastest (~3.7M parameters)
  - `mit_b1`: Balanced, default (~13.5M parameters)
  - `mit_b2`: Base model (~24.7M parameters)
  - `mit_b3`: Medium-large (~44M parameters)
  - `mit_b4`: Largest, best accuracy (~61.4M parameters)
- `--epochs`: Number of training epochs (default: 15)
- `--batch-size`: Batch size (default: 2)
- `--lr`: Learning rate (default: 0.005)
- `--model`: Path to save model checkpoint (default: trimodaldet.pth)

### Backbone Selection

You can select different encoder backbones to balance speed vs. accuracy:

```bash
# Fastest training/inference (recommended for prototyping)
python scripts/train.py --data data/ --epochs 15 --backbone mit_b0

# Default balanced configuration
python scripts/train.py --data data/ --epochs 15 --backbone mit_b1

# Higher accuracy for production
python scripts/train.py --data data/ --epochs 15 --backbone mit_b4
```

**Backbone Comparison:**

| Variant | Parameters | Depths | Embed Dims | Use Case |
|---------|-----------|---------|------------|----------|
| mit_b0 | ~3.7M | [2,2,2,2] | [32,64,160,256] | Fast prototyping, edge devices |
| mit_b1 | ~13.5M | [2,2,2,2] | [64,128,320,512] | Default, balanced |
| mit_b2 | ~24.7M | [3,4,6,3] | [64,128,320,512] | Higher accuracy |
| mit_b3 | ~44M | [3,4,18,3] | [64,128,320,512] | Medium-large model |
| mit_b4 | ~61.4M | [3,8,27,3] | [64,128,320,512] | Maximum accuracy |

### Evaluation

**Single Model Evaluation:**

```bash
python scripts/test.py --data data/ --model trimodaldet.pth
```

Results are saved to the test_results/ directory with visualizations of ground truth and predicted bounding boxes.

### Visualization

```bash
python scripts/visualize.py --vis 0
```

Visualizes the sample at index 0 from the dataset.

## Ablation Studies

The repository includes comprehensive ablation study implementations to analyze the contribution of different components:

### Fusion Mechanism Ablations

**CSSA (Channel Switching and Spatial Attention):**
```bash
python trimodaldet/ablations/scripts/train_cssa.py \
    --data data/ \
    --epochs 15 \
    --backbone mit_b1 \
    --cssa-thresh 0.5
```

**GAFF (Guided Attentive Feature Fusion):**
```bash
python trimodaldet/ablations/scripts/train_gaff_ablation.py \
    --data data/ \
    --epochs 15 \
    --backbone mit_b1 \
    --gaff-stages "1,2,3,4" \
    --gaff-se-reduction 4
```

### Modality Ablations

Test the contribution of individual modalities by training with different modality combinations:

```bash
# RGB + Thermal
python trimodaldet/ablations/scripts/train_modality_ablation.py \
    --data data/ \
    --modalities rgb thermal

# All modalities (RGB + Thermal + Event)
python trimodaldet/ablations/scripts/train_modality_ablation.py \
    --data data/ \
    --modalities rgb thermal event
```

## Configuration

The Config class in trimodaldet/config.py contains all hyperparameters:

```python
num_epochs = 15              # Training epochs
batch_size = 2               # Batch size
learning_rate = 0.005        # Learning rate
fpn_out_channels = 256       # FPN output channels
```

Modify these parameters based on your hardware and dataset requirements.

## Technical Details

### Inter-Modal Fusion

The Bidirectional Token Exchange (BiTE) combines features from RGB and auxiliary modality streams using inter-modal attention. Query vectors from one modality attend to key-value pairs from the other modality, enabling bidirectional information flow between modalities.

The Modality-Aware Gated Exchange (MAGE) adaptively weights contributions from each modality using both channel and spatial attention mechanisms. This improves robustness when one modality contains noise or missing data by dynamically adjusting feature rectification weights.

### Training

- Optimizer: SGD with momentum
- Loss: Combined classification and bounding box regression from Faster R-CNN
- Data split: Automatic 80/20 train/test split
- Device: Automatic GPU detection with CPU fallback

## Project Structure

```
trimodaldet/
├── trimodaldet/             # Main package
│   ├── models/              # Model architecture
│   │   ├── encoder.py       # InterModalBackbone and variants (mit_b0-b4)
│   │   ├── backbone.py      # InterModalBackbone with FPN wrapper
│   │   ├── fusion.py        # MAGE and BiTE fusion modules
│   │   └── transformer.py   # Transformer building blocks
│   ├── ablations/           # Ablation study implementations
│   │   ├── encoder_cssa.py          # CSSA fusion variants
│   │   ├── encoder_gaff_flexible.py # GAFF fusion variants
│   │   ├── backbone_modality.py     # Modality ablation backbone
│   │   ├── fusion/                  # Fusion module implementations
│   │   └── scripts/                 # Ablation training scripts
│   ├── data/                # Dataset and transforms
│   │   ├── dataset.py       # NpyYoloDataset for 5-channel inputs
│   │   └── transforms.py    # Data augmentation
│   ├── training/            # Training and evaluation
│   │   ├── trainer.py       # Training loop
│   │   └── evaluator.py     # Evaluation metrics
│   ├── utils/               # Utilities and visualization
│   │   ├── metrics.py       # mAP and detection metrics
│   │   └── visualization.py # Bounding box visualization
│   └── config.py            # Configuration and hyperparameters
├── scripts/                 # Entry point scripts
│   ├── train.py             # Main training script
│   ├── test.py              # Evaluation script
│   └── visualize.py         # Dataset visualization
├── data/                    # Data directory (user-created)
│   ├── images/              # .npy files (5-channel images)
│   └── labels/              # .txt files (YOLO format)
├── checkpoints/             # Model checkpoints (created during training)
├── requirements.txt
└── README.md
```

### Key Modules

**InterModalBackbone (`models/encoder.py`):**
- Implements the hierarchical inter-sensor backbone architecture
- Variants: `mit_b0`, `mit_b1`, `mit_b2`, `mit_b3`, `mit_b4`
- Each variant has different depths and embedding dimensions for speed/accuracy trade-offs
- Dual-stream processing with MAGE+BiTE fusion at each stage

**Fusion Modules (`models/fusion.py`):**
- `MAGE` (ModalityAwareGatedExchange): Channel and spatial gating for adaptive modality weighting
- `BiTE` (BidirectionalTokenExchange): Inter-modal attention for bidirectional feature fusion
- `InterModalAttention`: Core attention mechanism for cross-modality interaction

**Backbone Wrapper (`models/backbone.py`):**
- `InterModalBackbone`: Wraps the encoder with Feature Pyramid Network (FPN)
- Provides multi-scale features for detection head
- Compatible with torchvision's Faster R-CNN detector

## License

MIT License
