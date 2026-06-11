"""
Modality-configurable backbone for ablation experiments.

Allows selective enabling/disabling of input modalities (RGB, Thermal, Event)
to study their individual and combined contributions to detection performance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork


class ModalityConfigurableBackbone(nn.Module):
    """
    Inter-modal backbone with configurable modality inputs.

    Wraps the InterModalBackbone encoder and adds FPN for multi-scale
    feature extraction. Allows selective disabling of modalities for ablation studies.

    Args:
        encoder: InterModalBackbone encoder instance
        fpn_out_channels: Output channels for FPN (default: 256)
        active_modalities: List of active modalities, e.g., ['rgb', 'thermal', 'event']
                          Default is all three modalities enabled
    """

    def __init__(self, encoder, fpn_out_channels=256, active_modalities=None):
        super().__init__()
        self.encoder = encoder

        # Default: all modalities active
        if active_modalities is None:
            active_modalities = ['rgb', 'thermal', 'event']

        self.active_modalities = [m.lower() for m in active_modalities]

        # Validate modalities
        valid_modalities = {'rgb', 'thermal', 'event'}
        for m in self.active_modalities:
            if m not in valid_modalities:
                raise ValueError(f"Invalid modality '{m}'. Must be one of {valid_modalities}")

        # Ensure at least one modality is active
        if len(self.active_modalities) == 0:
            raise ValueError("At least one modality must be active")

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
        Forward pass through Multi-modal encoder and FPN with modality masking.

        Args:
            images_tensor: Input tensor of shape (B, 5, H, W)
                          Channels: RGB (0-2) + Thermal (3) + Event (4)

        Returns:
            Dictionary of FPN feature maps {'0': feat0, '1': feat1, ..., 'pool': pool}
        """
        # Clone the input to avoid modifying the original
        masked_input = images_tensor.clone()

        # Apply modality masking by zeroing out inactive channels
        if 'rgb' not in self.active_modalities:
            masked_input[:, :3] = 0  # Zero out RGB channels

        if 'thermal' not in self.active_modalities:
            masked_input[:, 3:4] = 0  # Zero out Thermal channel

        if 'event' not in self.active_modalities:
            masked_input[:, 4:5] = 0  # Zero out Event channel

        # Split the 5-channel input into RGB (3) and a combined X (2) stream
        rgb_images = masked_input[:, :3]      # Channels 0, 1, 2
        x_images = masked_input[:, 3:5]       # Channels 3 (Thermal) and 4 (Event)

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

    def get_modality_config(self):
        """
        Returns a string representation of the active modality configuration.

        Returns:
            String like "rgb+thermal+event" or "rgb+thermal"
        """
        return "+".join(sorted(self.active_modalities))


__all__ = ['ModalityConfigurableBackbone']
