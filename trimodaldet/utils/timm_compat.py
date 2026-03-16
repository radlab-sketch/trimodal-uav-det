"""
Timm library compatibility module.

Provides fallback implementations of timm.models.layers components
when timm is not installed.
"""
import math
import torch
import torch.nn as nn


# Try to import from timm, fallback to local implementations
try:
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:
    print("timm not found. Using local implementations of DropPath, to_2tuple, trunc_normal_.")

    # Simplified version of DropPath (Stochastic Depth)
    class DropPath(nn.Module):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

        def __init__(self, drop_prob=0., scale_by_keep=True):
            super(DropPath, self).__init__()
            self.drop_prob = drop_prob
            self.scale_by_keep = scale_by_keep

        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()  # binarize
            if self.scale_by_keep:
                output = x.div(keep_prob) * random_tensor
            else:
                output = x * random_tensor
            return output

    def to_2tuple(x):
        """Convert input to 2-tuple if it's not already a tuple/list."""
        if isinstance(x, (list, tuple)):
            return x
        return (x, x)

    def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
        """Initialize tensor with truncated normal distribution."""
        def _no_grad_trunc_normal_(tensor, mean, std, a, b):
            def norm_cdf(x):
                # Cumulative distribution function for standard normal distribution
                return (1. + math.erf(x / math.sqrt(2.))) / 2.

            with torch.no_grad():
                l = norm_cdf((a - mean) / std)
                u = norm_cdf((b - mean) / std)
                tensor.uniform_(2 * l - 1, 2 * u - 1)
                tensor.erfinv_()
                tensor.mul_(std * math.sqrt(2.))
                tensor.add_(mean)
                tensor.clamp_(min=a, max=b)
                return tensor
        return _no_grad_trunc_normal_(tensor, mean, std, a, b)


__all__ = ['DropPath', 'to_2tuple', 'trunc_normal_']
