"""
Flexible GAFF-enabled encoder for stage-wise fusion ablation experiments.

This encoder allows configurable stage selection for GAFF fusion.
Stages not in gaff_stages will use the original MAGE+BiTE fusion.

Example:
    gaff_stages=[4]      → GAFF at stage 4 only (default)
    gaff_stages=[2,3]    → GAFF at stages 2&3, MAGE+BiTE at 1&4
    gaff_stages=[1,2,3,4] → GAFF at all stages
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
from functools import partial

from trimodaldet.models.transformer import Block, OverlapPatchEmbed
from trimodaldet.models.fusion import MAGE, BiTE
from trimodaldet.ablations.fusion.gaff import GAFFBlock


class InterModalBackboneGAFFFlexible(nn.Module):
    """
    Dual-stream transformer encoder with configurable GAFF fusion.

    Args:
        gaff_stages: List of stages (1-4) to use GAFF fusion. Other stages use MAGE+BiTE.
        gaff_se_reduction: SE block reduction ratio for GAFF
        gaff_inter_shared: Use shared inter-modality attention in GAFF
        gaff_merge_bottleneck: Use bottleneck merge pathway in GAFF
    """

    def __init__(self, img_size=224, in_chans_rgb=3, in_chans_x=1, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, norm_fuse=nn.BatchNorm2d,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
                 gaff_stages=[4], gaff_se_reduction=4, gaff_inter_shared=False, gaff_merge_bottleneck=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.gaff_stages = gaff_stages  # List of stages to use GAFF

        print(f"Initializing flexible GAFF encoder:")
        print(f"  GAFF stages: {gaff_stages}")
        print(f"  GAFF config: SE_reduction={gaff_se_reduction}, inter_shared={gaff_inter_shared}, merge_bottleneck={gaff_merge_bottleneck}")
        print(f"  MAGE+BiTE stages: {[i+1 for i in range(4) if (i+1) not in gaff_stages]}")

        # RGB patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans_rgb, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Modal-X patch_embed
        self.extra_patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans_x, embed_dim=embed_dims[0])
        self.extra_patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.extra_patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.extra_patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # Stage 1
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        self.extra_block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.extra_norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        # Stage 2
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        self.extra_block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.extra_norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        # Stage 3
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        self.extra_block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.extra_norm3 = norm_layer(embed_dims[2])
        cur += depths[2]

        # Stage 4
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
        self.extra_block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.extra_norm4 = norm_layer(embed_dims[3])

        # Fusion modules: Create both MAGE+BiTE and GAFF for all stages
        # Only the ones specified in gaff_stages will be used
        self.MAGEs = nn.ModuleList([MAGE(dim=embed_dims[i]) for i in range(4)])
        self.BiTEs = nn.ModuleList([BiTE(dim=embed_dims[i], num_heads=num_heads[i]) for i in range(4)])

        # GAFF blocks for all stages (only used if stage in gaff_stages)
        self.GAFFs = nn.ModuleList([
            GAFFBlock(
                in_channels=embed_dims[i],
                out_channels=embed_dims[i],
                se_reduction=gaff_se_reduction,
                inter_shared=gaff_inter_shared,
                merge_bottleneck=gaff_merge_bottleneck
            )
            for i in range(4)
        ])

    def _fuse_stage(self, x_rgb, x_x, stage_idx):
        """
        Fuse features at a given stage using either GAFF or MAGE+BiTE.

        Args:
            x_rgb: RGB features (B, C, H, W)
            x_x: Extra modality features (B, C, H, W)
            stage_idx: Stage index (0-3 for stages 1-4)

        Returns:
            Fused features (B, C, H, W)
        """
        stage_num = stage_idx + 1  # Convert to 1-indexed

        if stage_num in self.gaff_stages:
            # Use GAFF fusion
            fused = self.GAFFs[stage_idx](x_rgb, x_x)
            return fused
        else:
            # Use baseline MAGE+BiTE fusion
            x_rgb, x_x = self.MAGEs[stage_idx](x_rgb, x_x)
            fused = self.BiTEs[stage_idx](x_rgb, x_x)
            return fused

    def forward_features(self, x_rgb, x_x):
        B = x_rgb.shape[0]
        outs = []

        # Stage 1
        x_rgb, H, W = self.patch_embed1(x_rgb)
        x_x, _, _ = self.extra_patch_embed1(x_x)
        for blk in self.block1: x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block1: x_x = blk(x_x, H, W)
        x_rgb, x_x = self.norm1(x_rgb), self.extra_norm1(x_x)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_x = x_x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        fused1 = self._fuse_stage(x_rgb, x_x, 0)
        outs.append(fused1)

        # For next stage, use fused output as input to both streams
        # This maintains compatibility with the original architecture
        x_rgb, x_x = fused1, fused1

        # Stage 2
        x_rgb, H, W = self.patch_embed2(x_rgb)
        x_x, _, _ = self.extra_patch_embed2(x_x)
        for blk in self.block2: x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block2: x_x = blk(x_x, H, W)
        x_rgb, x_x = self.norm2(x_rgb), self.extra_norm2(x_x)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_x = x_x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        fused2 = self._fuse_stage(x_rgb, x_x, 1)
        outs.append(fused2)

        x_rgb, x_x = fused2, fused2

        # Stage 3
        x_rgb, H, W = self.patch_embed3(x_rgb)
        x_x, _, _ = self.extra_patch_embed3(x_x)
        for blk in self.block3: x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block3: x_x = blk(x_x, H, W)
        x_rgb, x_x = self.norm3(x_rgb), self.extra_norm3(x_x)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_x = x_x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        fused3 = self._fuse_stage(x_rgb, x_x, 2)
        outs.append(fused3)

        x_rgb, x_x = fused3, fused3

        # Stage 4
        x_rgb, H, W = self.patch_embed4(x_rgb)
        x_x, _, _ = self.extra_patch_embed4(x_x)
        for blk in self.block4: x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block4: x_x = blk(x_x, H, W)
        x_rgb, x_x = self.norm4(x_rgb), self.extra_norm4(x_x)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_x = x_x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        fused4 = self._fuse_stage(x_rgb, x_x, 3)
        outs.append(fused4)

        return outs

    def forward(self, x_rgb, x_e):
        return self.forward_features(x_rgb, x_e)


# Variant configurations matching baseline encoder

class mit_b0_gaff_flexible(InterModalBackboneGAFFFlexible):
    """SegFormer MiT-B0 with flexible GAFF fusion."""
    def __init__(self, in_chans_rgb=3, in_chans_x=1, **kwargs):
        super(mit_b0_gaff_flexible, self).__init__(
            in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x,
            embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            **kwargs)


class mit_b1_gaff_flexible(InterModalBackboneGAFFFlexible):
    """SegFormer MiT-B1 with flexible GAFF fusion (default)."""
    def __init__(self, in_chans_rgb=3, in_chans_x=1, **kwargs):
        super(mit_b1_gaff_flexible, self).__init__(
            in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x,
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            **kwargs)


class mit_b2_gaff_flexible(InterModalBackboneGAFFFlexible):
    """SegFormer MiT-B2 with flexible GAFF fusion."""
    def __init__(self, in_chans_rgb=3, in_chans_x=1, **kwargs):
        super(mit_b2_gaff_flexible, self).__init__(
            in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x,
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            **kwargs)


class mit_b3_gaff_flexible(InterModalBackboneGAFFFlexible):
    """SegFormer MiT-B3 with flexible GAFF fusion."""
    def __init__(self, in_chans_rgb=3, in_chans_x=1, **kwargs):
        super(mit_b3_gaff_flexible, self).__init__(
            in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x,
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            **kwargs)


class mit_b4_gaff_flexible(InterModalBackboneGAFFFlexible):
    """SegFormer MiT-B4 with flexible GAFF fusion."""
    def __init__(self, in_chans_rgb=3, in_chans_x=1, **kwargs):
        super(mit_b4_gaff_flexible, self).__init__(
            in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x,
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            **kwargs)


class mit_b5_gaff_flexible(InterModalBackboneGAFFFlexible):
    """SegFormer MiT-B5 with flexible GAFF fusion."""
    def __init__(self, in_chans_rgb=3, in_chans_x=1, **kwargs):
        super(mit_b3_gaff_flexible, self).__init__(
            in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x,
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            **kwargs)


# Helper function to get encoder by name
def get_gaff_encoder(backbone='mit_b1', **kwargs):
    """
    Get GAFF-enabled encoder by backbone name.

    Args:
        backbone: Backbone name (mit_b0 through mit_b4)
        **kwargs: Additional arguments passed to encoder constructor

    Returns:
        GAFF-enabled encoder instance
    """
    encoders = {
        'mit_b0': mit_b0_gaff_flexible,
        'mit_b1': mit_b1_gaff_flexible,
        'mit_b2': mit_b2_gaff_flexible,
        'mit_b3': mit_b3_gaff_flexible,
        'mit_b4': mit_b4_gaff_flexible,
    }

    if backbone not in encoders:
        raise ValueError(f"Unknown backbone: {backbone}. Choose from {list(encoders.keys())}")

    return encoders[backbone](**kwargs)
