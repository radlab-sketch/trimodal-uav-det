"""
CSSA (Channel Switching and Spatial Attention) fusion module.

Reference:
    Cao et al., "Multimodal Object Detection by Channel Switching and Spatial Attention"
    CVPR 2023 PBVS Workshop
    https://github.com/artrela/mulitmodal-cssa
"""

import torch
import torch.nn as nn
from .base import FusionBlock


class ECABlock(nn.Module):
    """
    Efficient Channel Attention block.

    Uses 1D convolution on channel-wise global average pooled features
    to capture cross-channel interactions efficiently.
    """

    def __init__(self, kernel_size=3):
        super().__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.f = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Channel attention weights of shape (B, C, 1, 1)
        """
        x = self.GAP(x)  # (B, C, 1, 1)

        # Squeeze and transpose for 1D conv: (B, C, 1, 1) -> (B, 1, C)
        x = x.squeeze(-1).transpose(-1, -2)
        x = self.f(x)  # Apply 1D conv on channel dimension
        x = x.transpose(-1, -2).unsqueeze(-1)  # Back to (B, C, 1, 1)

        x = self.sigmoid(x)
        return x


class ChannelSwitching(nn.Module):
    """
    Channel switching module that selectively swaps channels between modalities
    based on learned attention weights.
    """

    def __init__(self, switching_thresh=0.5):
        super().__init__()
        self.k = switching_thresh

    def forward(self, x, x_prime, w):
        """
        Args:
            x: Primary modality features (B, C, H, W)
            x_prime: Alternative modality features (B, C, H, W)
            w: Channel attention weights (B, C, 1, 1)

        Returns:
            Features with channels switched where attention is low (B, C, H, W)
        """
        # If attention weight is low (w < k), take from alternative modality
        mask = w < self.k
        x = torch.where(mask, x_prime, x)
        return x


class SpatialAttention(nn.Module):
    """
    Spatial attention module that learns to weight spatial locations
    based on both average and max pooling across channels.
    """

    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb_feats, ir_feats):
        """
        Args:
            rgb_feats: RGB features (B, C, H, W)
            ir_feats: IR/Thermal features (B, C, H, W)

        Returns:
            Spatially attended and fused features (B, C, H, W)
        """
        B, C, H, W = rgb_feats.shape

        # Concatenate along channel dimension
        x_cat = torch.cat((rgb_feats, ir_feats), dim=1)  # (B, 2C, H, W)

        # Average pooling across channels -> spatial attention
        cap = torch.mean(x_cat, dim=1)  # (B, H, W)
        w_avg = self.sigmoid(cap)
        w_avg = w_avg.unsqueeze(1)  # (B, 1, H, W)

        # Max pooling across channels -> spatial attention
        cmp = torch.max(x_cat, dim=1)[0]  # (B, H, W)
        w_max = self.sigmoid(cmp)
        w_max = w_max.unsqueeze(1)  # (B, 1, H, W)

        # Apply spatial attention to concatenated features
        x_cat_w = x_cat * w_avg * w_max  # (B, 2C, H, W)

        # Split back into modality-specific features
        x_rgb_w = x_cat_w[:, :C, :, :]
        x_ir_w = x_cat_w[:, C:, :, :]

        # Fuse by averaging
        x_fused = (x_rgb_w + x_ir_w) / 2  # (B, C, H, W)

        return x_fused


class CSSABlock(FusionBlock):
    """
    Complete CSSA fusion block combining channel switching and spatial attention.

    This lightweight fusion module:
    1. Applies ECA attention to each modality independently
    2. Performs channel switching based on attention weights
    3. Applies spatial attention to fuse the switched features

    Args:
        switching_thresh: Threshold for channel switching (default: 0.5)
        kernel_size: Kernel size for ECA 1D convolution (default: 3)
    """

    def __init__(self, switching_thresh=0.5, kernel_size=3, **kwargs):
        super().__init__()

        # Separate ECA blocks for each modality
        self.eca_rgb = ECABlock(kernel_size=kernel_size)
        self.eca_aux = ECABlock(kernel_size=kernel_size)

        # Channel switching module
        self.cs = ChannelSwitching(switching_thresh=switching_thresh)

        # Spatial attention module
        self.sa = SpatialAttention()

    def forward(self, x_rgb: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CSSA fusion.

        Args:
            x_rgb: RGB features (B, C, H, W)
            x_aux: Auxiliary modality features (B, C, H, W)

        Returns:
            Fused features (B, C, H, W)
        """
        # Step 1: Compute channel attention for each modality
        rgb_w = self.eca_rgb(x_rgb)  # (B, C, 1, 1)
        aux_w = self.eca_aux(x_aux)  # (B, C, 1, 1)

        # Step 2: Channel switching
        # Where RGB attention is low, take auxiliary channels
        rgb_feats = self.cs(x_rgb, x_aux, rgb_w)
        # Where auxiliary attention is low, take RGB channels
        aux_feats = self.cs(x_aux, x_rgb, aux_w)

        # Step 3: Spatial attention and fusion
        fused_feats = self.sa(rgb_feats, aux_feats)  # (B, C, H, W)

        return fused_feats
