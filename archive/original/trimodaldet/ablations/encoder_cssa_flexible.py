"""
Flexible CSSA-enabled encoder for stage-wise fusion ablation experiments.

This encoder allows configurable stage selection for CSSA fusion.
Stages not in cssa_stages will use the original MAGE+BiTE fusion.

Example:
    cssa_stages=[4]      → CSSA at stage 4 only (default)
    cssa_stages=[2,3]    → CSSA at stages 2&3, MAGE+BiTE at 1&4
    cssa_stages=[1,2,3,4] → CSSA at all stages
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
from functools import partial

from trimodaldet.models.transformer import Block, OverlapPatchEmbed
from trimodaldet.models.fusion import MAGE, BiTE
from trimodaldet.ablations.fusion.cssa import CSSABlock


class InterModalBackboneCSSAFlexible(nn.Module):
    """
    Dual-stream transformer encoder with configurable CSSA fusion.

    Args:
        cssa_stages: List of stages (1-4) to use CSSA fusion. Other stages use MAGE+BiTE.
        cssa_switching_thresh: Threshold for channel switching in CSSA
        cssa_kernel_size: Kernel size for ECA in CSSA
    """

    def __init__(self, img_size=224, in_chans_rgb=3, in_chans_x=1, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, norm_fuse=nn.BatchNorm2d,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
                 cssa_stages=[4], cssa_switching_thresh=0.5, cssa_kernel_size=3):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.cssa_stages = cssa_stages  # List of stages to use CSSA

        print(f"Initializing flexible CSSA encoder:")
        print(f"  CSSA stages: {cssa_stages}")
        print(f"  MAGE+BiTE stages: {[i+1 for i in range(4) if (i+1) not in cssa_stages]}")

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

        # Fusion modules: Create both MAGE+BiTE and CSSA for all stages
        # Only the ones specified in cssa_stages will be used
        self.MAGEs = nn.ModuleList([MAGE(dim=embed_dims[i]) for i in range(4)])
        self.BiTEs = nn.ModuleList([BiTE(dim=embed_dims[i], num_heads=num_heads[i]) for i in range(4)])

        # CSSA blocks for all stages (only used if stage in cssa_stages)
        self.CSSAs = nn.ModuleList([
            CSSABlock(switching_thresh=cssa_switching_thresh, kernel_size=cssa_kernel_size)
            for _ in range(4)
        ])

    def _fuse_stage(self, x_rgb, x_x, stage_idx):
        """
        Fuse features at a given stage using either CSSA or MAGE+BiTE.

        Args:
            x_rgb: RGB features (B, C, H, W)
            x_x: Extra modality features (B, C, H, W)
            stage_idx: Stage index (0-3 for stages 1-4)

        Returns:
            Fused features (B, C, H, W)
        """
        stage_num = stage_idx + 1  # Convert to 1-indexed

        if stage_num in self.cssa_stages:
            # Use CSSA fusion
            fused = self.CSSAs[stage_idx](x_rgb, x_x)
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

class mit_b0_cssa_flexible(InterModalBackboneCSSAFlexible):
    """SegFormer MiT-B0 with flexible CSSA fusion."""
    def __init__(self, in_chans_rgb=3, in_chans_x=1, **kwargs):
        super(mit_b0_cssa_flexible, self).__init__(
            in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x,
            embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            **kwargs)


class mit_b1_cssa_flexible(InterModalBackboneCSSAFlexible):
    """SegFormer MiT-B1 with flexible CSSA fusion (default)."""
    def __init__(self, in_chans_rgb=3, in_chans_x=1, **kwargs):
        super(mit_b1_cssa_flexible, self).__init__(
            in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x,
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            **kwargs)


class mit_b2_cssa_flexible(InterModalBackboneCSSAFlexible):
    """SegFormer MiT-B2 with flexible CSSA fusion."""
    def __init__(self, in_chans_rgb=3, in_chans_x=1, **kwargs):
        super(mit_b2_cssa_flexible, self).__init__(
            in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x,
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            **kwargs)


class mit_b3_cssa_flexible(InterModalBackboneCSSAFlexible):
    """SegFormer MiT-B3 with flexible CSSA fusion."""
    def __init__(self, in_chans_rgb=3, in_chans_x=1, **kwargs):
        super(mit_b3_cssa_flexible, self).__init__(
            in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x,
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            **kwargs)


class mit_b4_cssa_flexible(InterModalBackboneCSSAFlexible):
    """SegFormer MiT-B4 with flexible CSSA fusion (largest)."""
    def __init__(self, in_chans_rgb=3, in_chans_x=1, **kwargs):
        super(mit_b4_cssa_flexible, self).__init__(
            in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x,
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            **kwargs)


def get_encoder_cssa_flexible(backbone_type='mit_b1', in_chans_rgb=3, in_chans_x=2, **kwargs):
    """
    Factory function to get flexible CSSA encoder variant.

    Args:
        backbone_type: One of 'mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4'
        in_chans_rgb: Number of RGB channels (default: 3)
        in_chans_x: Number of auxiliary channels (default: 2 for Thermal+Event)
        **kwargs: Additional arguments including:
            - cssa_stages: List of stages to use CSSA (default: [4])
            - cssa_switching_thresh: Threshold for CSSA (default: 0.5)
            - cssa_kernel_size: ECA kernel size (default: 3)

    Returns:
        Flexible CSSA-enabled encoder
    """
    encoder_map = {
        'mit_b0': mit_b0_cssa_flexible,
        'mit_b1': mit_b1_cssa_flexible,
        'mit_b2': mit_b2_cssa_flexible,
        'mit_b3': mit_b3_cssa_flexible,
        'mit_b4': mit_b4_cssa_flexible,
    }

    if backbone_type not in encoder_map:
        raise ValueError(f"Unknown backbone type: {backbone_type}. Choose from {list(encoder_map.keys())}")

    return encoder_map[backbone_type](in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x, **kwargs)
