"""
Base fusion block interface for ablation experiments.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class FusionBlock(nn.Module, ABC):
    """
    Abstract base class for multimodal fusion blocks.

    All fusion implementations should inherit from this class and implement
    the forward method with the following signature:

    Args:
        x_rgb: RGB modality features (B, C, H, W)
        x_aux: Auxiliary modality features (B, C, H, W) - Thermal + Event

    Returns:
        fused: Fused feature map (B, C, H, W)
    """

    @abstractmethod
    def forward(self, x_rgb: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        """
        Fuse features from two modalities.

        Args:
            x_rgb: RGB features of shape (B, C, H, W)
            x_aux: Auxiliary features of shape (B, C, H, W)

        Returns:
            Fused features of shape (B, C, H, W)
        """
        pass
