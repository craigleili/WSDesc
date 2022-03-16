from __future__ import division
from __future__ import print_function

import time
import random
import numpy as np
import open3d as o3d
import torch
import torch_cluster
import math
from scipy.spatial import cKDTree


class KNNSearch(object):

    def __init__(self, points):
        # points: (N, 3)
        self.points = np.asarray(points, dtype=np.float32)
        self.kdtree = cKDTree(points)

    def query(self, kpts, num_samples):
        # kpts: (K, 3)
        kpts = np.asarray(kpts, dtype=np.float32)
        nndists, nnindices = self.kdtree.query(kpts, k=num_samples, n_jobs=8)
        assert len(kpts) == len(nnindices)
        return nnindices  # (K, num_samples)

    def query_ball(self, kpt, radius):
        # kpt: (3, )
        kpt = np.asarray(kpt, dtype=np.float32)
        assert kpt.ndim == 1
        nnindices = self.kdtree.query_ball_point(kpt, radius, n_jobs=8)  # list
        return nnindices


def to_o3d_point_cloud(points, normals=None):
    """ 
    Args:
        points (np.array): (N, 3)
        normals (np.array): (N, 3)

    Returns:
        o3d.geometry.PointCloud:
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def to_o3d_feature(feats):
    """
    Args:
        feats (np.array): (N, D)

    Returns:
        o3d.pipelines.registration.Feature:
    """
    f = o3d.pipelines.registration.Feature()
    f.resize(feats.shape[1], feats.shape[0])
    f.data = np.asarray(feats, dtype=np.float64).transpose()
    return f


def estimate_lrf(pt, ptnn, patch_kernel):
    """Ref:
    https://github.com/fabiopoiesi/dip/blob/master/lrf.py

    Args:
        pt (np.array): (3, )
        ptnn (np.array): (3, NN-1), without pt
        patch_kernel (float):

    Returns:
        np.array: (3, 3)
    """
    ptnn_pt = ptnn - pt[:, np.newaxis]  # (3, NN-1)

    ptnn_cov = (1.0 / ptnn.shape[-1]) * np.dot(ptnn_pt, ptnn_pt.T)  # (3, 3)

    eigvals, eigvecs = np.linalg.eig(ptnn_cov)
    smallest_eigval_idx = np.argmin(eigvals)
    np_hat = eigvecs[:, smallest_eigval_idx]  # (3, )

    zp = np_hat if np.sum(np.dot(np_hat, -ptnn_pt)) >= 0 else -np_hat  # (3, )
    zp /= np.linalg.norm(zp)

    ptnn_pt_zp = np.dot(ptnn_pt.T, zp[:, np.newaxis])  # (NN-1, 1)
    v = ptnn_pt - (ptnn_pt_zp * zp).T  # (3, NN-1)

    # (NN-1, )
    alpha = (patch_kernel - np.linalg.norm(-ptnn_pt, axis=0))**2
    # (NN-1, )
    beta = ptnn_pt_zp.squeeze()**2

    v_alpha_beta = np.dot(v, (alpha * beta)[:, np.newaxis])  # (3, 1)
    if np.linalg.norm(v_alpha_beta) < 1e-4:
        xp = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        xp = v_alpha_beta / np.linalg.norm(v_alpha_beta)
        xp = xp.squeeze()  # (3, 1) -> (3, )

    yp = np.cross(zp, xp)  # (3, )
    yp /= np.linalg.norm(yp)

    xp = np.cross(yp, zp)  # (3, )
    xp /= np.linalg.norm(xp)

    # LRF
    lRg = np.stack((xp, yp, zp), axis=1)

    if np.isnan(np.sum(lRg)):
        return np.identity(3, dtype=np.float32)
    else:
        return np.asarray(lRg, dtype=np.float32)


def register_with_ransac(points_src,
                         points_dst,
                         desc_src=None,
                         desc_dst=None,
                         correspondences=None,
                         dist_thresh=0.05,
                         ransac_n=3):
    """
    Args:
        points_src (np.array): (N, 3)
        points_dst (np.array): (M, 3)
        desc_src (np.array): (N, D)
        desc_dst (np.array): (M, D)
        correspondences (np.array): (K, 2)
        dist_thresh (float): https://github.com/andyzeng/3dmatch-toolbox/blob/master/evaluation/geometric-registration/register2Fragments.m
        ransac_n (int):

    Returns:
        np.array, float: From src to dst (4, 4), scalar
    """
    assert (desc_src is not None and desc_dst is not None) or correspondences is not None

    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(np.copy(points_src))
    pcd_dst = o3d.geometry.PointCloud()
    pcd_dst.points = o3d.utility.Vector3dVector(np.copy(points_dst))

    if correspondences is not None:
        if len(correspondences) < ransac_n:
            duration = 0.
            trans = np.identity(4, dtype=np.float64)
        else:
            corrs = o3d.utility.Vector2iVector(correspondences)
            start_time = time.time()
            result = o3d.registration.registration_ransac_based_on_correspondence(
                pcd_src, pcd_dst, corrs, dist_thresh,
                o3d.registration.TransformationEstimationPointToPoint(False), ransac_n,
                o3d.registration.RANSACConvergenceCriteria(50000, 1000))
            duration = time.time() - start_time
            trans = np.asarray(result.transformation, dtype=np.float64)
    else:
        if min(len(desc_src), len(desc_dst)) < ransac_n:
            duration = 0.
            trans = np.identity(4, dtype=np.float64)
        else:
            feat_src = o3d.registration.Feature()
            feat_src.resize(desc_src.shape[1], desc_src.shape[0])
            feat_src.data = np.asarray(desc_src, dtype=np.float64).transpose()
            feat_dst = o3d.registration.Feature()
            feat_dst.resize(desc_dst.shape[1], desc_dst.shape[0])
            feat_dst.data = np.asarray(desc_dst, dtype=np.float64).transpose()
            start_time = time.time()
            # https://github.com/andyzeng/3dmatch-toolbox/blob/master/core/external/sfm/ransacfitRt.m
            result = o3d.registration.registration_ransac_based_on_feature_matching(
                pcd_src, pcd_dst, feat_src, feat_dst, dist_thresh,
                o3d.registration.TransformationEstimationPointToPoint(False), ransac_n, [
                    o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh)
                ], o3d.registration.RANSACConvergenceCriteria(50000, 1000))
            duration = time.time() - start_time
            trans = np.asarray(result.transformation, dtype=np.float64)
    # src * trans and dst are aligned
    return trans, duration


def rotation_matrix(axis, angle):
    """
    Ref:
    - https://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    - https://github.com/Wallacoloo/printipi/blob/master/util/rotation_matrix.py

    Args:
        axis (np.array or List): (3, )
        angle (float): in radians

    Returns:
        np.array: (4, 4)
    """
    assert len(axis) == 3

    axis = np.asarray(axis, dtype=np.float32)
    axis /= np.linalg.norm(axis)

    ca = math.cos(angle)
    sa = math.sin(angle)
    C = 1 - ca

    x, y, z = axis[0], axis[1], axis[2]

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    matrix = np.identity(4, dtype=np.float32)
    matrix[0, 0] = x * xC + ca
    matrix[0, 1] = xyC - zs
    matrix[0, 2] = zxC + ys
    matrix[1, 0] = xyC + zs
    matrix[1, 1] = y * yC + ca
    matrix[1, 2] = yzC - xs
    matrix[2, 0] = zxC - ys
    matrix[2, 1] = yzC + xs
    matrix[2, 2] = z * zC + ca
    return matrix


def translation_matrix(x):
    """
    Args:
        x (np.array or List): (3, )

    Returns:
        np.array: (4, 4)
    """
    assert len(x) == 3
    m = np.identity(4, dtype=np.float32)
    m[:3, 3] = np.asarray(x, dtype=np.float32)
    return m


def random_transform(points, normals, max_rotation_degree, max_translate):
    """
    Args:
        points (np.array): (N, 3)
        normals (np.array): (N, 3)
        max_rotation_degree (float): 
        max_translate (float):

    Returns:
        np.array: 
    """
    center = np.mean(points, axis=0, keepdims=False)  # (3, )
    assert max_rotation_degree <= 180.0
    mrd = max_rotation_degree * math.pi / 180.0
    rot = rotation_matrix(np.random.uniform(-1.0, 1.0, 3), random.uniform(-mrd, mrd))
    trans = translation_matrix(np.random.uniform(-max_translate, max_translate, 3))
    pose = trans @ translation_matrix(center) @ rot @ translation_matrix(-center)

    points = np.concatenate((points, np.ones((len(points), 1), dtype=points.dtype)), axis=1)
    points = points @ pose.T
    points = points[:, :3]

    if normals is not None:
        normals = np.concatenate((normals, np.zeros((len(normals), 1), dtype=normals.dtype)),
                                 axis=1)
        normals = normals @ pose.T
        normals = normals[:, :3]

    return points, normals, pose


def random_rotate(points, normals):
    """
    Args:
        points (np.array): (N, 3)
        normals (np.array): (N, 3)

    Returns:
        np.array: 
    """

    def rotmat(axis, theta):
        s = math.sin(theta)
        c = math.cos(theta)
        m = np.identity(4, dtype=np.float32)
        if axis == 0:
            m[1, 1] = c
            m[1, 2] = -s
            m[2, 1] = s
            m[2, 2] = c
        elif axis == 1:
            m[0, 0] = c
            m[0, 2] = s
            m[2, 0] = -s
            m[2, 2] = c
        elif axis == 2:
            m[0, 0] = c
            m[0, 1] = -s
            m[1, 0] = s
            m[1, 1] = c
        else:
            raise RuntimeError('The axis - {} is not supported.'.format(axis))
        return m

    pose = rotmat(random.randint(0, 2), random.uniform(0, math.pi * 2.))

    points = np.concatenate((points, np.ones((len(points), 1), dtype=points.dtype)), axis=1)
    points = points @ pose.T
    points = points[:, :3]

    if normals is not None:
        normals = np.concatenate((normals, np.zeros((len(normals), 1), dtype=normals.dtype)),
                                 axis=1)
        normals = normals @ pose.T
        normals = normals[:, :3]

    return points, normals, pose


def filter_outliers(points, normals=None, nb_points=256, radius=0.3):
    dtype = points.dtype
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd_flt, _ = pcd.remove_radius_outlier(nb_points, radius)
    out_points = np.asarray(pcd_flt.points, dtype=dtype)
    if normals is not None:
        out_normals = np.asarray(pcd_flt.normals, dtype=dtype)
    else:
        out_normals = None
    return out_points, out_normals


def farthest_point_sampling(points, max_points=512, gpu=-1):
    # Ref:
    # https://github.com/pytorch/pytorch/issues/40403#issuecomment-648515174
    if gpu >= 0:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')
    npoints = len(points)
    indices = torch_cluster.fps(torch.as_tensor(points, dtype=torch.float32, device=device),
                                torch.zeros(npoints, dtype=torch.int64, device=device),
                                ratio=float(max_points) / npoints,
                                random_start=True)
    indices = indices.cpu().numpy()
    return indices


def exclude(a, b):
    # a: list
    # b: list
    # a - b
    if not isinstance(a, set):
        a = set(a)
    if not isinstance(b, set):
        b = set(b)
    return list(a - b)
