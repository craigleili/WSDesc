from __future__ import print_function
from __future__ import division

import os.path as osp
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# Functionals
# ---------------------------------------------------------------------------- #
def l2_distance(x, y, dim, keepdim=False, squared=False, eps=1e-8):
    """
    Args:
        x (torch.Tensor):
        y (torch.Tensor):
        dim (int):
        keepdim (bool): 
        squared (bool): 
        eps (float, optional): 

    Returns:
        torch.Tensor: 
    """
    dist2 = torch.sum((x - y)**2, dim=dim, keepdim=keepdim)
    if squared:
        return dist2
    else:
        dist2 = torch.clamp(dist2, min=eps)
        return torch.sqrt(dist2)


def pairwise_distance_matrix(x, y, squared=False, eps=1e-12):
    """Compute pairwise L2 distance

    Args:
        x (torch.Tensor): (*, M, D)
        y (torch.Tensor): (*, N, D)
        squared (bool): 
        eps (float, optional): 

    Returns:
        torch.Tensor: (*, M, N)
    """
    x2 = torch.sum(x**2, dim=-1, keepdim=True)  # (*, M, 1)
    y2 = torch.sum(y**2, dim=-1, keepdim=True)  # (*, N, 1)
    dist2 = -2.0 * torch.matmul(x, torch.transpose(y, -2, -1))
    dist2 += x2
    dist2 += torch.transpose(y2, -2, -1)
    if squared:
        return dist2
    else:
        dist2 = torch.clamp(dist2, min=eps)
        return torch.sqrt(dist2)


def pinverse_cpu(x):
    """On CPU

    Args:
        x (torch.Tensor): (*, M, N)

    Returns:
        torch.Tensor: (*, N, M)
    """
    device = x.device
    return torch.pinverse(x.cpu()).to(device)


def affine_registration(Xt, Yt, W=None):
    """
    Args:
        Xt (torch.Tensor): (*, N, 3)
        Yt (torch.Tensor): (*, N, 3)
        W (torch.Tensor): (*, N)

    Returns:
        A (torch.Tensor): (*, 3, 3), from X to Y
        t (torch.Tensor): (*, 3), from X to Y
    """
    ones_shape = list(Xt.size())[:-1] + [1]  # [*, N, 1]
    ones = torch.ones(*ones_shape, dtype=Xt.dtype, device=Xt.device)  # (*, N, 1)
    Xt = torch.cat((Xt, ones), dim=-1)  # (*, N, 4)

    if W is None:
        sol_t = pinverse_cpu(Xt) @ Yt  # (*, 4, 3)
        sol = torch.transpose(sol_t, -2, -1)  # (*, 3, 4)
    else:
        W = W / torch.sum(W, dim=-1, keepdim=True)  # (*, N)
        W = torch.diag_embed(W)  # (*, N, N)
        sol_t = pinverse_cpu(W @ Xt) @ (W @ Yt)  # (*, 4, 3)
        sol = torch.transpose(sol_t, -2, -1)  # (*, 3, 4)

    # (*, 3, 3) (*, 3)
    A, t = sol[..., :3], sol[..., 3]
    return A, t


def to_mat4x4(R, t):
    """
    Args:
        R (torch.Tensor): (3, 3)
        t (torch.Tensor): (3,)

    Returns:
        torch.Tensor: (4, 4)
    """
    m = torch.cat((R, torch.unsqueeze(t, dim=1)), dim=1)  # (3, 4)
    r = torch.as_tensor([[0, 0, 0, 1]], dtype=m.dtype, device=m.device)  # (1, 4)
    m = torch.cat((m, r), dim=0)  # (4, 4)
    return m


def spectral_matching(src_keypts, tgt_keypts, inlier_threshold, num_iters=10):
    """Ref:
    - A spectral technique for correspondence problems using pairwise constraints

    Args:
        src_keypts (torch.Tensor): (B, N, 3)
        tgt_keypts (torch.Tensor): (B, N, 3)
        inlier_threshold (float):
        num_iters (int):

    Returns:
        torch.Tensor: (B, N)
    """
    assert src_keypts.size(-2) == tgt_keypts.size(-2)

    # (B, N, N) = (B, N, 1, 3) - (B, 1, N, 3)
    src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
    tgt_dist = torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)
    M = src_dist - tgt_dist  # (B, N, N)
    M = torch.clamp(1.0 - M**2 / inlier_threshold**2, min=0.0)
    diag_mask = 1.0 - torch.eye(M.shape[-2], M.shape[-1], dtype=M.dtype, device=M.device)
    M = torch.mul(M, torch.unsqueeze(diag_mask, 0))

    # Ref:
    # https://github.com/XuyangBai/PointDSC/blob/master/models/PointDSC.py
    leading_eig = torch.ones_like(M[:, :, 0:1])  # (B, N, 1)
    leading_eig_last = leading_eig
    for _ in range(num_iters):
        leading_eig = torch.bmm(M, leading_eig)
        leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
        if torch.allclose(leading_eig, leading_eig_last):
            break
        leading_eig_last = leading_eig
    leading_eig = leading_eig.squeeze(-1)

    return leading_eig


# ---------------------------------------------------------------------------- #
# Modules
# ---------------------------------------------------------------------------- #
class RigidityLoss(nn.Module):
    LOSS_FUNCS = {'l1': F.l1_loss, 'l2': F.mse_loss}

    def __init__(self, inlier_threshold, loss_type='l1', eps=1e-6, **kwargs):
        """
        Args:
            loss_type (str): ['l1', 'l2']
        """
        super().__init__()

        self.inlier_threshold = inlier_threshold
        self.loss_fn = self.LOSS_FUNCS[loss_type]
        self.eps = eps
        # Additional arguments
        for k, w in kwargs.items():
            setattr(self, k, w)

    def ortho_error(self, R):
        RtR = torch.matmul(torch.transpose(R, 0, 1), R)
        I = torch.eye(R.size(-1), dtype=R.dtype, device=R.device)
        return self.loss_fn(RtR, I, reduction='mean')

    def cycle_error(self, A, A_inv):
        AA = torch.matmul(A, A_inv)
        I = torch.eye(A.size(-1), dtype=A.dtype, device=A.device)
        return self.loss_fn(AA, I, reduction='mean')

    def _register(self, fdists, points_x, points_y, inlier_threshold):
        """
        Args:
            fdists (torch.Tensor): (M, N)
            points_x (torch.Tensor): (M, 3)
            points_y (torch.Tensor): (N, 3)
            inlier_threshold (float):

        Returns:
            torch.Tensor: R, t, from Y to X.
        """

        M, N = fdists.size(-2), fdists.size(-1)
        # (M, N)
        fscores = torch.softmax(-fdists, dim=-1)
        # (M, ) (M, )
        nn_weights, nn_indices = torch.max(fscores, dim=-1)
        # (M, N)
        coeffs = torch.zeros_like(fscores).scatter_(dim=-1,
                                                    index=torch.unsqueeze(nn_indices, 1),
                                                    value=1.0) - fscores.detach() + fscores
        points_y = torch.matmul(coeffs, points_y)  # (M, 3)
        sm_weights = spectral_matching(torch.unsqueeze(points_x, 0),
                                       torch.unsqueeze(points_y, 0),
                                       inlier_threshold=inlier_threshold)  # (1, M)
        sm_weights = torch.squeeze(sm_weights, 0)  # (M, )
        weights = nn_weights * sm_weights

        batch_points_x = torch.unsqueeze(points_x, dim=0)
        batch_points_y = torch.unsqueeze(points_y, dim=0)
        batch_weights = torch.unsqueeze(weights, dim=0)

        # (1, 3, 3), (1, 3)    X = R * Y + t
        batch_R, batch_t = affine_registration(batch_points_y, batch_points_x, batch_weights)

        return batch_R[0], batch_t[0]

    def forward(self, fdists_xy, kpts_x, kpts_y, points_x, points_y):
        """
        Args:
            fdists_xy (torch.Tensor): (M, N)
            kpts_x (torch.Tensor): (M, 3)
            kpts_y (torch.Tensor): (N, 3)
            points_x (torch.Tensor): (M', 3)
            points_y (torch.Tensor): (N', 3)

        Returns:
            torch.Tensor:
        """

        R_y2x, t_y2x = self._register(fdists_xy, kpts_x, kpts_y, self.inlier_threshold)
        R_x2y, t_x2y = self._register(torch.transpose(fdists_xy, -2, -1), kpts_y, kpts_x,
                                      self.inlier_threshold)

        R_ortho_err = 0.5 * (self.ortho_error(R_y2x) + self.ortho_error(R_x2y))
        R_cycle_err = self.cycle_error(to_mat4x4(R_y2x, t_y2x), to_mat4x4(R_x2y, t_x2y))

        return R_ortho_err, R_cycle_err
