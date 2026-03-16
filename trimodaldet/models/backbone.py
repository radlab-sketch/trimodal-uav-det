"""
Inter-modal backbone with Feature Pyramid Network for object detection.

Wraps the Multi-modal encoder with FPN to serve as a backbone for Faster R-CNN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork


class InterModalBackbone(nn.Module):
    """
    Inter-modal backbone with Feature Pyramid Network (FPN).

    Wraps the InterModalBackbone encoder and adds FPN for multi-scale
    feature extraction. Designed to work with torchvision's Faster R-CNN.

    Args:
        encoder: InterModalBackbone encoder instance
        fpn_out_channels: Output channels for FPN (default: 256)
    """

    def __init__(self, encoder, fpn_out_channels=256):
        super().__init__()
        self.encoder = encoder

        # Extract channel dimensions from the encoder stages
        in_channels_list = self.encoder.embed_dims

        # Define the FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=fpn_out_channels,
        )
        self.out_channels = fpn_out_channels

    def forward(self, images_tensor):
        """
        Forward pass through Multi-modal encoder and FPN.

        Args:
            images_tensor: Input tensor of shape (B, 5, H, W)
                          Channels: RGB (0-2) + Thermal (3) + Event (4)

        Returns:
            Dictionary of FPN feature maps {'0': feat0, '1': feat1, ..., 'pool': pool}
        """
        # Split the 5-channel input into RGB (3) and a combined X (2) stream
        rgb_images = images_tensor[:, :3]      # Channels 0, 1, 2
        x_images = images_tensor[:, 3:5]       # Channels 3 (Thermal) and 4 (Event)

        # Get multi-scale features from the inter-modal backbone
        # The output `features` is a list of tensors, one for each stage
        features = self.encoder.forward_features(rgb_images, x_images)

        # The FPN expects a dictionary mapping feature names to tensors
        # Use strings '0' through '3' as names
        feature_dict = {str(i): f for i, f in enumerate(features)}

        # Pass the features through the FPN
        fpn_features = self.fpn(feature_dict)

        # The standard torchvision FPN architecture produces a 5th feature map
        # for the RPN by max-pooling the last level. Add this manually.
        # The key 'pool' is the standard name used by the RoIAlign layer.
        last_feature_map = list(fpn_features.values())[-1]
        fpn_features['pool'] = F.max_pool2d(last_feature_map, 1, 2, 0)

        return fpn_features


__all__ = ['InterModalBackbone']
