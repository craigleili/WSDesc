from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import glob
import logging
import sys
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import minkowski
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils.io import may_create_folder
from utils.pcd import farthest_point_sampling, estimate_lrf, KNNSearch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- #
# Ref:
# - https://github.com/WangYueFt/prnet
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Util functions
# ---------------------------------------------------------------------------- #
def download(root_dir):
    if not osp.exists(root_dir):
        may_create_folder(root_dir)
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = osp.join(root_dir, osp.basename(www))
        os.system('wget -P {} --no-check-certificate {}'.format(root_dir, www))
        os.system('unzip -j {} -d {}'.format(zipfile, root_dir))
        os.system('rm {}'.format(zipfile))


def load_data(root_dir, partition):
    """
    Args:
        root_dir (str):
        partition (str): ['train', 'test']

    Returns:
        np.array: point clouds, labels
    """
    download(root_dir)

    all_xyz, all_normal, all_label = [], [], []
    for h5_name in glob.glob(os.path.join(root_dir, 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        all_xyz.append(f['data'][:].astype('float32'))  # (2048, 2048, 3)
        all_normal.append(f['normal'][:].astype('float32'))  # (2048, 2048, 3)
        all_label.append(f['label'][:].astype('int64'))  # (2048, 1)
        f.close()
    all_xyz = np.concatenate(all_xyz, axis=0)  # (N, 2048, 3)
    all_normal = np.concatenate(all_normal, axis=0)  # (N, 2048, 3)
    all_label = np.concatenate(all_label, axis=0)  # (N, 1)
    return all_xyz, all_normal, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def farthest_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=768):
    pointcloud1 = pointcloud1.T  # (N, 3)
    pointcloud2 = pointcloud2.T  # (N, 3)
    num_points = pointcloud1.shape[0]

    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points,
                             algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1)

    random_p1 = np.random.random(
        size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))

    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points,
                             algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud2)

    random_p2 = random_p1
    #np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return idx1, idx2


def crop_patch(kpt_indices,
               points,
               normals=None,
               num_points_per_sample=1024,
               sample_radius=0.3,
               with_lrf=False):
    # kpt_indices: (K, )
    # points: (N, 3)
    # normals: (N, 3)
    kpts = points[kpt_indices, :]
    knn_search = KNNSearch(points)

    num_kpts = len(kpts)

    patches = np.zeros((num_kpts, num_points_per_sample, 3), dtype=np.float32)
    patch_normals = np.zeros_like(patches) if normals is not None else None
    lrfs = np.zeros((num_kpts, 3, 3), dtype=np.float32) if with_lrf else None

    for i in range(num_kpts):
        nnindices = knn_search.query_ball(kpts[i, :], sample_radius)
        try:
            nnindices.remove(kpt_indices[i])
        except ValueError:
            pass
        if len(nnindices) < 1:
            nnindices = knn_search.query(kpts[i:i + 1, :], 1 + num_points_per_sample)
            nnindices = nnindices[0, 1:]

        if with_lrf:
            lrfs[i] = estimate_lrf(kpts[i, :], points[nnindices, :].T, sample_radius)

        if len(nnindices) > num_points_per_sample:
            nnindices = np.random.choice(nnindices, num_points_per_sample, replace=False)
        elif len(nnindices) < num_points_per_sample:
            nnindices = np.random.choice(nnindices, num_points_per_sample, replace=True)
        patches[i] = points[nnindices, :]
        if normals is not None:
            patch_normals[i] = normals[nnindices, :]

    return patches, patch_normals, lrfs


# ---------------------------------------------------------------------------- #
# Datasets
# ---------------------------------------------------------------------------- #
class ModelNet40(Dataset):

    def __init__(self,
                 data_root,
                 num_samples,
                 num_points_per_sample,
                 sample_radius,
                 num_points,
                 num_subsampled_points=768,
                 partition='train',
                 gaussian_noise=False,
                 unseen=False,
                 rot_factor=4,
                 category=None):
        super(ModelNet40, self).__init__()

        self.data_root = data_root
        self.num_samples = num_samples
        self.num_points_per_sample = num_points_per_sample
        self.sample_radius = sample_radius

        # (N, 2048, 3), (N, 2048, 3), (N, 1)
        self.data, self.normal, self.label = load_data(data_root, partition)
        self.label = self.label.squeeze()  # (N, )

        if category is not None:
            self.data = self.data[self.label == category]
            self.normal = self.normal[self.label == category]
            self.label = self.label[self.label == category]

        self.num_points = num_points
        self.num_subsampled_points = num_subsampled_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.rot_factor = rot_factor
        if num_points != num_subsampled_points:
            self.subsampled = True
        else:
            self.subsampled = False
        if self.unseen:
            if self.partition == 'test':
                self.data = self.data[self.label >= 20]
                self.normal = self.normal[self.label >= 20]
                self.label = self.label[self.label >= 20]
            elif self.partition == 'train':
                self.data = self.data[self.label < 20]
                self.normal = self.normal[self.label < 20]
                self.label = self.label[self.label < 20]

    def __getitem__(self, item):
        # (2048, 3) -> (num_points=1024, 3)
        pointcloud = self.data[item][:self.num_points]
        normals = self.normal[item][:self.num_points]

        if self.partition != 'train':
            np.random.seed(item)

        anglex = np.random.uniform() * np.pi / self.rot_factor
        angley = np.random.uniform() * np.pi / self.rot_factor
        anglez = np.random.uniform() * np.pi / self.rot_factor
        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])  # (3, 3)
        Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])  # (3, 3)
        Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])  # (3, 3)
        R_ab = Rx.dot(Ry).dot(Rz)  # (3, 3)
        R_ba = R_ab.T  # (3, 3)
        translation_ab = np.array([
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5)
        ])  # (3, )
        translation_ba = -R_ba.dot(translation_ab)  # (3, )

        pointcloud1 = pointcloud.T  # (3, num_points)
        normals1 = normals.T  # (3, num_points)

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        # (3, num_points)
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab,
                                                                          axis=1)
        normals2 = rotation_ab.apply(normals1.T).T  # (3, num_points)

        euler_ab = np.asarray([anglez, angley, anglex])  # (3, )
        euler_ba = -euler_ab[::-1]  # (3, )

        xyznormal1 = np.concatenate((pointcloud1.T, normals1.T), axis=1)  # (num_points, 6)
        xyznormal2 = np.concatenate((pointcloud2.T, normals2.T), axis=1)  # (num_points, 6)
        assert xyznormal1.shape[1] == 6 and xyznormal2.shape[1] == 6
        xyznormal1 = np.random.permutation(xyznormal1)  # (num_points, 6)
        xyznormal2 = np.random.permutation(xyznormal2)  # (num_points, 6)

        pointcloud1, normals1 = xyznormal1[:, :3].T, xyznormal1[:, 3:6].T  # (3, num_points)
        pointcloud2, normals2 = xyznormal2[:, :3].T, xyznormal2[:, 3:6].T  # (3, num_points)

        if self.gaussian_noise:
            pointcloud1 = jitter_pointcloud(pointcloud1)  # (3, num_points)
            pointcloud2 = jitter_pointcloud(pointcloud2)  # (3, num_points)

        if self.subsampled:
            sidx1, sidx2 = farthest_subsample_points(
                pointcloud1, pointcloud2, num_subsampled_points=self.num_subsampled_points)
            pointcloud1 = pointcloud1[:, sidx1]
            pointcloud2 = pointcloud2[:, sidx2]
            normals1 = normals1[:, sidx1]
            normals2 = normals2[:, sidx2]

        pointcloud1 = np.asarray(pointcloud1.T, dtype=np.float32)  # (num_points, 3)
        pointcloud2 = np.asarray(pointcloud2.T, dtype=np.float32)  # (num_points, 3)
        normals1 = np.asarray(normals1.T, dtype=np.float32)  # (num_points, 3)
        normals2 = np.asarray(normals2.T, dtype=np.float32)  # (num_points, 3)

        if self.partition == 'train':
            kpt_indices1 = farthest_point_sampling(pointcloud1, max_points=self.num_samples)
            kpt_indices2 = farthest_point_sampling(pointcloud2, max_points=self.num_samples)
            kpts1 = pointcloud1[kpt_indices1, :]
            kpts2 = pointcloud2[kpt_indices2, :]
        else:
            kpt_indices1 = np.arange(len(pointcloud1))
            kpt_indices2 = np.arange(len(pointcloud2))
            kpts1 = np.copy(pointcloud1)
            kpts2 = np.copy(pointcloud2)

        patches1, patch_normals1, lrfs1 = crop_patch(
            kpt_indices1,
            pointcloud1,
            normals=normals1,
            num_points_per_sample=self.num_points_per_sample,
            sample_radius=self.sample_radius,
            with_lrf=True)

        patches2, patch_normals2, lrfs2 = crop_patch(
            kpt_indices2,
            pointcloud2,
            normals=normals2,
            num_points_per_sample=self.num_points_per_sample,
            sample_radius=self.sample_radius,
            with_lrf=True)

        return {
            'pcd_i': pointcloud1,
            'pcd_j': pointcloud2,
            'normals_i': normals1,
            'normals_j': normals2,
            'kpts_i': kpts1,
            'kpts_j': kpts2,
            'kpt_indices_i': kpt_indices1,
            'kpt_indices_j': kpt_indices2,
            'patches_i': patches1,
            'patches_j': patches2,
            'patch_normals_i': patch_normals1,
            'patch_normals_j': patch_normals2,
            'lrfs_i': lrfs1,
            'lrfs_j': lrfs2,
            'R_ab': np.asarray(R_ab, dtype=np.float32),
            'translation_ab': np.asarray(translation_ab, dtype=np.float32),
            'R_ba': np.asarray(R_ba, dtype=np.float32),
            'translation_ba': np.asarray(translation_ba, dtype=np.float32),
            'euler_ab': np.asarray(euler_ab, dtype=np.float32),
            'euler_ba': np.asarray(euler_ba, dtype=np.float32),
        }

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        return 'ModelNet40: {}'.format(self.partition)
