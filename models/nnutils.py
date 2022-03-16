import functools
import logging
import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# Utilities
# ---------------------------------------------------------------------------- #
def get_norm(norm_type='instance_norm', dim='1d', trainable=False):
    bns = {
        '1d': nn.BatchNorm1d,
        '2d': nn.BatchNorm2d,
        '3d': nn.BatchNorm3d,
    }
    ins = {
        '1d': nn.InstanceNorm1d,
        '2d': nn.InstanceNorm2d,
        '3d': nn.InstanceNorm3d,
    }
    if norm_type == 'batch_norm':
        return functools.partial(bns[dim], affine=trainable)
    elif norm_type == 'instance_norm':
        return functools.partial(ins[dim], affine=trainable)
    else:
        raise NotImplementedError('Normalization layer - {} is not found'.format(norm_type))


def is_batchnorm(norm_layer):
    if type(norm_layer) == functools.partial:
        return norm_layer.func == nn.BatchNorm1d or norm_layer.func == nn.BatchNorm2d or norm_layer.func == nn.BatchNorm3d
    else:
        name = norm_layer.__class__.__name__
        return name == 'BatchNorm1d' or name == 'BatchNorm2d' or name == 'BatchNorm3d'
