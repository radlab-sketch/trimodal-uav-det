"""
GAFF (Guided Attentive Feature Fusion) Module

Paper: "Guided Attentive Feature Fusion for Multispectral Pedestrian Detection"
Conference: WACV 2021
Authors: Heng Zhang, Elisa Fromont, Sébastien Lefèvre, Bruno Avignon
GitHub: https://github.com/zhanghengdev/GAFF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base import FusionBlock


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for intra-modality channel attention.

    Architecture: GAP → FC → ReLU → FC → Sigmoid

    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck (default: 4)
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        reduced_channels = max(channels // reduction, 1)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Attention-weighted tensor (B, C, H, W)
        """
        b, c, _, _ = x.size()

        # Squeeze: Global average pooling
        y = self.gap(x).view(b, c)

        # Excitation: FC → ReLU → FC → Sigmoid
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)

        # Reshape and apply attention weights
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


class InterModalityAttention(nn.Module):
    """
    Inter-modality attention module.

    Computes inter-modal attention weights between RGB and auxiliary modalities:
    - w_rgb←aux: How much auxiliary information to add to RGB
    - w_aux←rgb: How much RGB information to add to auxiliary

    Architecture: Concat[x_rgb, x_aux] → 1×1 Conv → Split

    Args:
        channels: Number of channels per modality
        shared: If True, use shared conv for both directions (default: False)
    """
    def __init__(self, channels: int, shared: bool = False):
        super().__init__()
        self.channels = channels
        self.shared = shared

        if shared:
            # Single conv produces both attention maps
            self.conv = nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=False)
        else:
            # Separate convs for each direction
            self.conv_rgb = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
            self.conv_aux = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_rgb: torch.Tensor, x_aux: torch.Tensor):
        """
        Args:
            x_rgb: RGB features (B, C, H, W)
            x_aux: Auxiliary features (B, C, H, W)
        Returns:
            Tuple of (w_rgb←aux, w_aux←rgb) attention weights
        """
        # Concatenate both modalities
        concat_feat = torch.cat([x_rgb, x_aux], dim=1)

        if self.shared:
            # Shared convolution
            attn = self.conv(concat_feat)
            w_rgb_from_aux, w_aux_from_rgb = torch.chunk(attn, 2, dim=1)
        else:
            # Separate convolutions
            w_rgb_from_aux = self.conv_rgb(concat_feat)
            w_aux_from_rgb = self.conv_aux(concat_feat)

        # Apply sigmoid activation
        w_rgb_from_aux = self.sigmoid(w_rgb_from_aux)
        w_aux_from_rgb = self.sigmoid(w_aux_from_rgb)

        return w_rgb_from_aux, w_aux_from_rgb


class GAFFBlock(FusionBlock):
    """
    Guided Attentive Feature Fusion (GAFF) block.

    Combines intra-modality and inter-modality attention for guided fusion:
    1. Apply SE attention within each modality
    2. Compute inter-modal attention weights between modalities
    3. Apply guided fusion: x̂ = SE(x) + w_inter * x_other
    4. Merge: Concat[x̂_rgb, x̂_aux] → 1×1 Conv → BN

    Args:
        in_channels: Number of input channels per modality
        out_channels: Number of output channels (default: same as in_channels)
        se_reduction: SE block reduction ratio (default: 4)
        inter_shared: Use shared inter-modality attention (default: False)
        merge_bottleneck: If True, merge to C then expand to 2C (default: False)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        se_reduction: int = 4,
        inter_shared: bool = False,
        merge_bottleneck: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.se_reduction = se_reduction
        self.inter_shared = inter_shared
        self.merge_bottleneck = merge_bottleneck

        # Intra-modality attention (SE blocks)
        self.se_rgb = SEBlock(in_channels, reduction=se_reduction)
        self.se_aux = SEBlock(in_channels, reduction=se_reduction)

        # Inter-modality attention
        self.inter_attn = InterModalityAttention(in_channels, shared=inter_shared)

        # Merge layer
        if merge_bottleneck:
            # 2C → C → 2C pathway
            self.merge_conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
            self.merge_bn1 = nn.BatchNorm2d(in_channels)
            self.merge_conv2 = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False)
            self.merge_bn2 = nn.BatchNorm2d(self.out_channels)
        else:
            # Direct 2C → out_channels
            self.merge_conv = nn.Conv2d(in_channels * 2, self.out_channels, kernel_size=1, bias=False)
            self.merge_bn = nn.BatchNorm2d(self.out_channels)

    def forward(self, x_rgb: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_rgb: RGB features (B, C, H, W)
            x_aux: Auxiliary features (B, C, H, W)
        Returns:
            Fused features (B, out_channels, H, W)
        """
        # Step 1: Intra-modality attention (SE)
        x_rgb_se = self.se_rgb(x_rgb)
        x_aux_se = self.se_aux(x_aux)

        # Step 2: Inter-modality attention (inter-modal attention weights)
        w_rgb_from_aux, w_aux_from_rgb = self.inter_attn(x_rgb, x_aux)

        # Step 3: Guided fusion
        # x̂_rgb = SE_rgb(x_rgb) + w_rgb←aux * x_aux
        # x̂_aux = SE_aux(x_aux) + w_aux←rgb * x_rgb
        x_rgb_guided = x_rgb_se + w_rgb_from_aux * x_aux
        x_aux_guided = x_aux_se + w_aux_from_rgb * x_rgb

        # Step 4: Merge modalities
        concat_feat = torch.cat([x_rgb_guided, x_aux_guided], dim=1)

        if self.merge_bottleneck:
            # Bottleneck pathway
            out = self.merge_conv1(concat_feat)
            out = self.merge_bn1(out)
            out = F.relu(out, inplace=True)
            out = self.merge_conv2(out)
            out = self.merge_bn2(out)
        else:
            # Direct pathway
            out = self.merge_conv(concat_feat)
            out = self.merge_bn(out)

        return out

    def extra_repr(self) -> str:
        return (
            f'in_channels={self.in_channels}, '
            f'out_channels={self.out_channels}, '
            f'se_reduction={self.se_reduction}, '
            f'inter_shared={self.inter_shared}, '
            f'merge_bottleneck={self.merge_bottleneck}'
        )


def build_gaff_block(
    in_channels: int,
    out_channels: Optional[int] = None,
    se_reduction: int = 4,
    inter_shared: bool = False,
    merge_bottleneck: bool = False,
) -> GAFFBlock:
    """
    Factory function to build a GAFF fusion block.

    Args:
        in_channels: Number of input channels per modality
        out_channels: Number of output channels (default: same as in_channels)
        se_reduction: SE block reduction ratio (4 or 8 recommended)
        inter_shared: Use shared inter-modality attention
        merge_bottleneck: Use bottleneck merge pathway

    Returns:
        Configured GAFFBlock instance
    """
    return GAFFBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        se_reduction=se_reduction,
        inter_shared=inter_shared,
        merge_bottleneck=merge_bottleneck,
    )
