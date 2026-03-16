"""
TriModalDet Package.

Tri-modal object detection with RGB + Thermal + Event fusion.
"""
from .config import Config
from .models.encoder import mit_b1
from .models.backbone import InterModalBackbone
from .data.dataset import NpyYoloDataset, collate_fn
from .training.trainer import Trainer
from .training.evaluator import Evaluator
from .utils.metrics import evaluate
from .utils.visualization import visualize_dataset_sample, visualize_evaluation_sample

__version__ = '1.0.0'
__all__ = [
    'Config',
    'mit_b1',
    'InterModalBackbone',
    'NpyYoloDataset',
    'collate_fn',
    'Trainer',
    'Evaluator',
    'evaluate',
    'visualize_dataset_sample',
    'visualize_evaluation_sample',
]
