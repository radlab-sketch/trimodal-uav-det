"""
CSSA-enabled encoder for fusion ablation experiments.

This encoder replaces Stage 4 MAGE+BiTE with CSSA fusion block,
keeping stages 1-3 unchanged from the baseline.
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


class InterModalBackboneCSSA(nn.Module):
    """
    Dual-stream transformer encoder with CSSA fusion at Stage 4.

    Stages 1-3: Use original MAGE + BiTE
    Stage 4: Use CSSA fusion block

    This allows direct comparison of CSSA vs baseline fusion at the deepest stage.
    """

    def __init__(self, img_size=224, in_chans_rgb=3, in_chans_x=1, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, norm_fuse=nn.BatchNorm2d,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
                 cssa_switching_thresh=0.5, cssa_kernel_size=3):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims

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

        # Fusion modules: Stages 1-3 use original MAGE+BiTE, Stage 4 uses CSSA
        self.MAGEs = nn.ModuleList([MAGE(dim=embed_dims[i]) for i in range(3)])  # Only stages 1-3
        self.BiTEs = nn.ModuleList([BiTE(dim=embed_dims[i], num_heads=num_heads[i]) for i in range(3)])  # Only stages 1-3

        # Stage 4: CSSA fusion
        self.cssa_stage4 = CSSABlock(
            switching_thresh=cssa_switching_thresh,
            kernel_size=cssa_kernel_size
        )

    def forward_features(self, x_rgb, x_x):
        B = x_rgb.shape[0]
        outs = []

        # Stage 1 - Original MAGE+BiTE
        x_rgb, H, W = self.patch_embed1(x_rgb)
        x_x, _, _ = self.extra_patch_embed1(x_x)
        for blk in self.block1: x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block1: x_x = blk(x_x, H, W)
        x_rgb, x_x = self.norm1(x_rgb), self.extra_norm1(x_x)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_x = x_x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_x = self.MAGEs[0](x_rgb, x_x)
        outs.append(self.BiTEs[0](x_rgb, x_x))

        # Stage 2 - Original MAGE+BiTE
        x_rgb, H, W = self.patch_embed2(x_rgb)
        x_x, _, _ = self.extra_patch_embed2(x_x)
        for blk in self.block2: x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block2: x_x = blk(x_x, H, W)
        x_rgb, x_x = self.norm2(x_rgb), self.extra_norm2(x_x)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_x = x_x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_x = self.MAGEs[1](x_rgb, x_x)
        outs.append(self.BiTEs[1](x_rgb, x_x))

        # Stage 3 - Original MAGE+BiTE
        x_rgb, H, W = self.patch_embed3(x_rgb)
        x_x, _, _ = self.extra_patch_embed3(x_x)
        for blk in self.block3: x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block3: x_x = blk(x_x, H, W)
        x_rgb, x_x = self.norm3(x_rgb), self.extra_norm3(x_x)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_x = x_x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_x = self.MAGEs[2](x_rgb, x_x)
        outs.append(self.BiTEs[2](x_rgb, x_x))

        # Stage 4 - CSSA FUSION (THE KEY DIFFERENCE!)
        x_rgb, H, W = self.patch_embed4(x_rgb)
        x_x, _, _ = self.extra_patch_embed4(x_x)
        for blk in self.block4: x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block4: x_x = blk(x_x, H, W)
        x_rgb, x_x = self.norm4(x_rgb), self.extra_norm4(x_x)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_x = x_x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # Use CSSA instead of MAGE+BiTE
        fused = self.cssa_stage4(x_rgb, x_x)
        outs.append(fused)

        return outs

    def forward(self, x_rgb, x_e):
        return self.forward_features(x_rgb, x_e)


# Variant configurations matching baseline encoder

class mit_b0_cssa(InterModalBackboneCSSA):
    """SegFormer MiT-B0 with CSSA fusion at Stage 4."""
    def __init__(self, in_chans_rgb=3, in_chans_x=1, **kwargs):
        super(mit_b0_cssa, self).__init__(
            in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x,
            embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            **kwargs)


class mit_b1_cssa(InterModalBackboneCSSA):
    """SegFormer MiT-B1 with CSSA fusion at Stage 4 (default)."""
    def __init__(self, in_chans_rgb=3, in_chans_x=1, **kwargs):
        super(mit_b1_cssa, self).__init__(
            in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x,
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            **kwargs)


class mit_b2_cssa(InterModalBackboneCSSA):
    """SegFormer MiT-B2 with CSSA fusion at Stage 4."""
    def __init__(self, in_chans_rgb=3, in_chans_x=1, **kwargs):
        super(mit_b2_cssa, self).__init__(
            in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x,
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            **kwargs)


class mit_b3_cssa(InterModalBackboneCSSA):
    """SegFormer MiT-B3 with CSSA fusion at Stage 4."""
    def __init__(self, in_chans_rgb=3, in_chans_x=1, **kwargs):
        super(mit_b3_cssa, self).__init__(
            in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x,
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            **kwargs)


class mit_b4_cssa(InterModalBackboneCSSA):
    """SegFormer MiT-B4 with CSSA fusion at Stage 4 (largest)."""
    def __init__(self, in_chans_rgb=3, in_chans_x=1, **kwargs):
        super(mit_b4_cssa, self).__init__(
            in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x,
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            **kwargs)


def get_encoder_cssa(backbone_type='mit_b1', in_chans_rgb=3, in_chans_x=2, **kwargs):
    """
    Factory function to get CSSA encoder variant.

    Args:
        backbone_type: One of 'mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4'
        in_chans_rgb: Number of RGB channels (default: 3)
        in_chans_x: Number of auxiliary channels (default: 2 for Thermal+Event)
        **kwargs: Additional arguments passed to encoder (e.g., cssa_switching_thresh)

    Returns:
        CSSA-enabled encoder
    """
    encoder_map = {
        'mit_b0': mit_b0_cssa,
        'mit_b1': mit_b1_cssa,
        'mit_b2': mit_b2_cssa,
        'mit_b3': mit_b3_cssa,
        'mit_b4': mit_b4_cssa,
    }

    if backbone_type not in encoder_map:
        raise ValueError(f"Unknown backbone type: {backbone_type}. Choose from {list(encoder_map.keys())}")

    return encoder_map[backbone_type](in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x, **kwargs)
