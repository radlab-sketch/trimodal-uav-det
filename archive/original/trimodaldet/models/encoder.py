"""
Inter-Modal Backbone for dual-stream cross-modal fusion.

Implements the hierarchical inter-sensor backbone architecture with parallel RGB and modal-X streams.
"""
import torch
import torch.nn as nn
from functools import partial

from .transformer import Block, OverlapPatchEmbed
from .fusion import MAGE, BiTE


class InterModalBackbone(nn.Module):
    """
    Hierarchical inter-sensor backbone with cross-modal fusion.

    Args:
        img_size: Input image size
        in_chans_rgb: Number of RGB input channels
        in_chans_x: Number of modal-X input channels
        num_classes: Number of classes (not used in backbone)
        embed_dims: Embedding dimensions for each stage
        num_heads: Number of attention heads for each stage
        mlp_ratios: MLP expansion ratios for each stage
        qkv_bias: Whether to use bias in QKV projection
        qk_scale: Scale factor for attention scores
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
        norm_layer: Normalization layer
        norm_fuse: Normalization layer for fusion modules
        depths: Number of blocks in each stage
        sr_ratios: Spatial reduction ratios for each stage
    """

    def __init__(self, img_size=224, in_chans_rgb=3, in_chans_x=1, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, norm_fuse=nn.BatchNorm2d,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims

        # RGB patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans_rgb, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Other modality (X) patch_embed
        self.extra_patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans_x, embed_dim=embed_dims[0])
        self.extra_patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.extra_patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.extra_patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
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

        self.MAGEs = nn.ModuleList([MAGE(dim=embed_dims[i]) for i in range(4)])
        self.BiTEs = nn.ModuleList([BiTE(dim=embed_dims[i], num_heads=num_heads[i]) for i in range(4)])

    def forward_features(self, x_rgb, x_x):
        B = x_rgb.shape[0]
        outs = []

        # stage 1
        x_rgb, H, W = self.patch_embed1(x_rgb)
        x_x, _, _ = self.extra_patch_embed1(x_x)
        for blk in self.block1: x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block1: x_x = blk(x_x, H, W)
        x_rgb, x_x = self.norm1(x_rgb), self.extra_norm1(x_x)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_x = x_x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_x = self.MAGEs[0](x_rgb, x_x)
        outs.append(self.BiTEs[0](x_rgb, x_x))

        # stage 2
        x_rgb, H, W = self.patch_embed2(x_rgb)
        x_x, _, _ = self.extra_patch_embed2(x_x)
        for blk in self.block2: x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block2: x_x = blk(x_x, H, W)
        x_rgb, x_x = self.norm2(x_rgb), self.extra_norm2(x_x)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_x = x_x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_x = self.MAGEs[1](x_rgb, x_x)
        outs.append(self.BiTEs[1](x_rgb, x_x))

        # stage 3
        x_rgb, H, W = self.patch_embed3(x_rgb)
        x_x, _, _ = self.extra_patch_embed3(x_x)
        for blk in self.block3: x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block3: x_x = blk(x_x, H, W)
        x_rgb, x_x = self.norm3(x_rgb), self.extra_norm3(x_x)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_x = x_x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_x = self.MAGEs[2](x_rgb, x_x)
        outs.append(self.BiTEs[2](x_rgb, x_x))

        # stage 4
        x_rgb, H, W = self.patch_embed4(x_rgb)
        x_x, _, _ = self.extra_patch_embed4(x_x)
        for blk in self.block4: x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block4: x_x = blk(x_x, H, W)
        x_rgb, x_x = self.norm4(x_rgb), self.extra_norm4(x_x)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_x = x_x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_x = self.MAGEs[3](x_rgb, x_x)
        outs.append(self.BiTEs[3](x_rgb, x_x))

        return outs

    def forward(self, x_rgb, x_e):
        return self.forward_features(x_rgb, x_e)


class mit_b0(InterModalBackbone):
    """SegFormer MiT-B0 configuration with cross-modal fusion (smallest, fastest)."""

    def __init__(self, in_chans_rgb=3, in_chans_x=1, **kwargs):
        super(mit_b0, self).__init__(
            in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x,
            embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


class mit_b1(InterModalBackbone):
    """SegFormer MiT-B1 configuration with cross-modal fusion."""

    def __init__(self, in_chans_rgb=3, in_chans_x=1, **kwargs):
        super(mit_b1, self).__init__(
            in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x,
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


class mit_b2(InterModalBackbone):
    """SegFormer MiT-B2 configuration with cross-modal fusion."""

    def __init__(self, in_chans_rgb=3, in_chans_x=1, **kwargs):
        super(mit_b2, self).__init__(
            in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x,
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


class mit_b3(InterModalBackbone):
    """SegFormer MiT-B3 configuration with cross-modal fusion."""

    def __init__(self, in_chans_rgb=3, in_chans_x=1, **kwargs):
        super(mit_b3, self).__init__(
            in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x,
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


class mit_b4(InterModalBackbone):
    """SegFormer MiT-B4 configuration with cross-modal fusion (large, deep)."""

    def __init__(self, in_chans_rgb=3, in_chans_x=1, **kwargs):
        super(mit_b4, self).__init__(
            in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x,
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


# Registry pattern for backbone selection
BACKBONE_REGISTRY = {
    'mit_b0': mit_b0,
    'mit_b1': mit_b1,
    'mit_b2': mit_b2,
    'mit_b3': mit_b3,
    'mit_b4': mit_b4,
}


def get_encoder(backbone_name, in_chans_rgb=3, in_chans_x=1):
    """
    Factory function to instantiate encoder by backbone name.

    Args:
        backbone_name: Name of backbone variant ('mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4')
        in_chans_rgb: Number of RGB input channels (default: 3)
        in_chans_x: Number of modal-X input channels (default: 1)

    Returns:
        Instantiated encoder

    Raises:
        ValueError: If backbone_name is not in BACKBONE_REGISTRY

    Example:
        >>> encoder = get_encoder('mit_b1', in_chans_rgb=3, in_chans_x=1)
    """
    if backbone_name not in BACKBONE_REGISTRY:
        available = ', '.join(sorted(BACKBONE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. "
            f"Available options: {available}"
        )

    encoder_class = BACKBONE_REGISTRY[backbone_name]
    return encoder_class(in_chans_rgb=in_chans_rgb, in_chans_x=in_chans_x)


__all__ = ['InterModalBackbone', 'mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4',
           'BACKBONE_REGISTRY', 'get_encoder']
