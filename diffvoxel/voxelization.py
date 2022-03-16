from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import torch
from torch.utils.cpp_extension import load

ROOT_DIR = osp.abspath(osp.dirname(__file__))

# JIT
cu = load(
    'cu',
    [
        osp.join(ROOT_DIR, 'cuda', 'utils.cpp'),
        osp.join(ROOT_DIR, 'cuda', 'voxelize_cuda_kernel.cu')
    ],
    verbose=True,
)


# ---------------------------------------------------------------------------- #
# Utilities
# ---------------------------------------------------------------------------- #
def create_voxel_grids(edge_length, resolution):
    """Locate at origin

    Args:
        edge_length (float): 
        resolution (float): 

    Returns:
        np.array: (R, R, R, 3)
    """
    a = np.linspace(0.0, edge_length, resolution + 1,
                    dtype=np.float32) + edge_length / (2. * resolution)
    a = a[:-1]
    c = np.stack(np.meshgrid(a, a, a), axis=3)
    o = np.ones((3,), dtype=np.float32) * ((a[0] + a[-1]) / 2.0)
    c -= np.reshape(o, (1, 1, 1, 3))
    return c


def transform_voxel_grids(centers, scales, rotations, translates):
    """
    Args:
        centers (torch.tensor): (V, 3)
        scales (torch.tensor): (B, S) or None
        rotations (torch.tensor): (B, S, 3, 3), or None
        translates (torch.tensor): (B, 3)

    Returns:
        torch.tensor:
    """
    vcs = centers
    V = vcs.size(0)
    vcs = vcs.view(1, 1, V, -1)  # (1, 1, V, 3)

    # Scaling
    if scales is not None:
        B, S = scales.size()
        vcs = vcs * scales.view(B, S, 1, 1)  # (B, S, V, 3)

    # Rotation
    if rotations is not None:
        assert rotations.size(-1) == rotations.size(-2) == 3
        rotations = torch.unsqueeze(rotations, 2)  # (B, S, 1, 3, 3)
        vcs = torch.matmul(rotations, torch.unsqueeze(vcs, 4))  # (B, S, V, 3, 1)
        vcs = torch.squeeze(vcs, 4)  # (B, S, V, 3)

    # Translation
    if translates is not None:
        B = translates.size(0)
        vcs = vcs + translates.view(B, 1, 1, 3)

    return vcs  # (B, S, V, 3)


# ---------------------------------------------------------------------------- #
# Modules
# ---------------------------------------------------------------------------- #
class VoxelizeFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, points, voxel_centers, voxel_radii, sigma):
        """
        Args:
            ctx :
            points (torch.tensor): (N, 3)
            voxel_centers (torch.tensor): (K, V, 3)
            voxel_radii (torch.tensor): (K, V)
            sigma (float): 

        Returns:
            torch.tensor: (K, V)
        """

        K, V = voxel_centers.size(0), voxel_centers.size(1)
        dtype, device = points.dtype, points.device

        voxel_grids = torch.ones(K, V, dtype=dtype, device=device)

        cu.voxelize_fw(points.contiguous(), voxel_centers.contiguous(),
                       voxel_radii.contiguous(), sigma, voxel_grids.contiguous())

        voxel_grids = 1.0 - voxel_grids

        # Save for backward
        ctx.save_for_backward(points, voxel_centers, voxel_radii, voxel_grids)
        ctx.sigma = sigma

        return voxel_grids

    @staticmethod
    def backward(ctx, grad_voxel_grids):
        """
        Args:
            ctx : 
            grad_voxel_grids (torch.tensor): (K, V)

        Returns:
            torch.tensor: (K, V, 3), (K, V)
        """

        points, voxel_centers, voxel_radii, voxel_grids = ctx.saved_tensors
        grad_voxel_centers = None
        grad_voxel_radii = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            grad_voxel_centers = torch.zeros_like(voxel_centers)
            grad_voxel_radii = torch.zeros_like(voxel_radii)
            cu.voxelize_bw(points.contiguous(), voxel_centers.contiguous(),
                           voxel_radii.contiguous(), ctx.sigma, voxel_grids.contiguous(),
                           grad_voxel_grids.contiguous(), grad_voxel_centers.contiguous(),
                           grad_voxel_radii.contiguous())

        return None, grad_voxel_centers, grad_voxel_radii, None


voxelize = VoxelizeFunc.apply
