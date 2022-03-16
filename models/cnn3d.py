import os.path as osp
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from models.nnutils import get_norm, is_batchnorm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# Utilities
# ---------------------------------------------------------------------------- #
class MaxPoolFusion(nn.Module):

    def forward(self, x):
        # x: (B, N, C, D, H, W)

        if x.size(1) == 1:
            return torch.squeeze(x, dim=1)  # Single volume
        else:
            x, _ = torch.max(x, dim=1)  # (B, C, D, H, W)
            return x


# ---------------------------------------------------------------------------- #
# Modules
# ---------------------------------------------------------------------------- #
class Conv3dEncoder(nn.Module):
    """Ref:
    - https://github.com/zgojcic/3DSmoothNet/blob/master/core/architecture.py
    """

    def __init__(self,
                 in_channels,
                 desc_dim,
                 fusion,
                 dropout=False,
                 dropout_prob=0.2,
                 weight_normalize=False,
                 **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.desc_dim = desc_dim
        self.fusion = fusion
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.weight_normalize = weight_normalize
        # Additional arguments
        for k, w in kwargs.items():
            setattr(self, k, w)

        self.wn = lambda m: nn.utils.weight_norm(m) if self.weight_normalize else m
        self._build()

    def _build(self):
        # 3D Convs
        channels = [self.in_channels, 32, 32, 64, 64, 128, 128, self.desc_dim]
        strides = [1] * len(channels)
        strides[2] = 2
        dropout_flags = [False] * len(channels)
        if self.dropout:
            dropout_flags[5] = True

        norm3d = get_norm(norm_type='instance_norm', dim='3d', trainable=False)
        self.convs = nn.ModuleList()
        for lid in range(0, len(channels) - 2):
            block = [
                self.wn(
                    nn.Conv3d(channels[lid],
                              channels[lid + 1],
                              kernel_size=3,
                              stride=strides[lid],
                              padding=1,
                              bias=not is_batchnorm(norm3d))),
                norm3d(channels[lid + 1]),
                nn.ReLU(True),
            ]
            if dropout_flags[lid]:
                block.append(nn.Dropout(self.dropout_prob))
            self.convs.append(nn.Sequential(*block))

        # Fusion
        if self.fusion == 'max_pool':
            self.fuse = MaxPoolFusion()
        else:
            raise RuntimeError('Fusion - {} is not supported.'.format(self.fusion))

        # Embedding
        self.embed = self.wn(
            nn.Conv3d(channels[-2], channels[-1], kernel_size=8, stride=1, padding=0))

    def forward(self, x, l2_normalize):
        # x: (B, N, C, D, H, W)

        B, N, C, D, H, W = x.size()

        # 3D Convs
        x = x.view(-1, C, D, H, W)
        for module in self.convs:
            x = module(x)

        C, D, H, W = x.shape[1:]
        x = x.view(B, N, C, D, H, W)

        # Fusion
        x = self.fuse(x)  # (B, C, D, H, W)

        # Embedding
        x = self.embed(x)  # (B, C, 1, 1, 1)
        x = x.view(B, -1)
        if l2_normalize:
            x = F.normalize(x, p=2, dim=1)
        return x  # (B, C)


cnn3ds = {
    'conv3d': Conv3dEncoder,
}
