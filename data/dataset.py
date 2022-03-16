from __future__ import division
from __future__ import print_function

import os.path as osp
import logging
import random
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils.io import list_folders
from utils import lmdb
from utils.pcd import random_rotate, farthest_point_sampling
from utils.pcd import KNNSearch, estimate_lrf, filter_outliers

log = logging.getLogger(__name__)


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
# Testing dataset
# ---------------------------------------------------------------------------- #
class PointCloudDataset(Dataset):

    def __init__(self, data_root, num_points_per_sample, sample_radius, **kwargs):
        self.data_root = data_root
        self.num_points_per_sample = num_points_per_sample
        self.sample_radius = sample_radius
        # Additional arguments
        for k, w in kwargs.items():
            setattr(self, k, w)

        self.pcd_dbs = dict()
        self.pcd_keys = list()
        for scene in list_folders(data_root):
            if scene.endswith('.lmdb'):
                db = lmdb.open_db(osp.join(data_root, scene))
                name = lmdb.read_db_once(db, '__scene__')
                self.pcd_dbs[name] = db
                keys = lmdb.read_db_once(db, '__keys__')
                self.pcd_keys.extend([(name, k) for k in keys])

    def __getitem__(self, index):
        scene = self.pcd_keys[index][0]
        _, seq, name = self.pcd_keys[index][1].split('/')

        def read_pcd(label):
            return lmdb.read_db_once(self.pcd_dbs[scene], '/{}/{}/{}'.format(seq, name, label))

        points = read_pcd('points_sparse')  # (N, 3)
        kpts = read_pcd('kpts')  # (K, 3)
        normals = read_pcd('normals_sparse')

        knn_search = KNNSearch(points)
        kpt_indices = knn_search.query(kpts, 1)
        kpts = points[kpt_indices, :]

        patches, patch_normals, lrfs = crop_patch(
            kpt_indices,
            points,
            normals=normals,
            num_points_per_sample=self.num_points_per_sample,
            sample_radius=self.sample_radius,
            with_lrf=True)

        ret = {
            'pcd': points,
            'normals': normals,
            'kpts': kpts,
            'kpt_indices': kpt_indices,
            'patches': patches,
            'patch_normals': patch_normals,
            'lrfs': lrfs,
            'scene': scene,
            'seq': seq,
            'name': name,
        }
        return ret

    def __len__(self):
        return len(self.pcd_keys)

    def __del__(self):
        self.dispose()

    def dispose(self):
        if self.pcd_dbs is not None:
            for k in self.pcd_dbs.keys():
                self.pcd_dbs[k].close()
                self.pcd_dbs[k] = None

    def __repr__(self):
        return 'PointCloudDataset: testing on {}'.format(self.data_root)


# ---------------------------------------------------------------------------- #
# Training datasets
# ---------------------------------------------------------------------------- #
class PointCloudPairTrainDataset(Dataset):
    """For training with partially overlapped real data with unreliable poses
    """

    def __init__(self, pcd_root, num_samples, num_points_per_sample, sample_radius, rot_augment,
                 **kwargs):
        self.pcd_root = pcd_root
        self.num_samples = num_samples
        self.num_points_per_sample = num_points_per_sample
        self.sample_radius = sample_radius
        self.rot_augment = rot_augment
        # Additional arguments
        for k, w in kwargs.items():
            setattr(self, k, w)

        self.pcd_db = lmdb.open_db(self.pcd_root)
        self.pair_keys = lmdb.read_db_once(self.pcd_db, '__pair_keys__')

    def __getitem__(self, index):
        pair_key = self.pair_keys[index]
        _, scene, seq, name_i, name_j = pair_key.split('/')

        def read_pcd(name, label):
            return lmdb.read_db_once(self.pcd_db,
                                     '/{}/{}/{}/{}'.format(scene, seq, name, label))

        points_i = read_pcd(name_i, 'points')
        points_j = read_pcd(name_j, 'points')
        normals_i = read_pcd(name_i, 'normals')
        normals_j = read_pcd(name_j, 'normals')

        points_i, normals_i = filter_outliers(points_i, normals_i)
        points_j, normals_j = filter_outliers(points_j, normals_j)

        if self.rot_augment:
            if random.uniform(0, 1) > 0.5:
                points_i, normals_i, _ = random_rotate(points_i, normals_i)
            if random.uniform(0, 1) > 0.5:
                points_j, normals_j, _ = random_rotate(points_j, normals_j)

        kpt_indices_i = farthest_point_sampling(points_i, max_points=self.num_samples)
        kpt_indices_j = farthest_point_sampling(points_j, max_points=self.num_samples)

        kpts_i = points_i[kpt_indices_i, :]
        kpts_j = points_j[kpt_indices_j, :]

        patches_i, patch_normals_i, lrfs_i = crop_patch(
            kpt_indices_i,
            points_i,
            normals=normals_i,
            num_points_per_sample=self.num_points_per_sample,
            sample_radius=self.sample_radius,
            with_lrf=True)

        patches_j, patch_normals_j, lrfs_j = crop_patch(
            kpt_indices_j,
            points_j,
            normals=normals_j,
            num_points_per_sample=self.num_points_per_sample,
            sample_radius=self.sample_radius,
            with_lrf=True)

        ret = {
            'pcd_i': points_i,
            'pcd_j': points_j,
            'normals_i': normals_i,
            'normals_j': normals_j,
            'kpts_i': kpts_i,
            'kpts_j': kpts_j,
            'kpt_indices_i': kpt_indices_i,
            'kpt_indices_j': kpt_indices_j,
            'patches_i': patches_i,
            'patches_j': patches_j,
            'patch_normals_i': patch_normals_i,
            'patch_normals_j': patch_normals_j,
            'lrfs_i': lrfs_i,
            'lrfs_j': lrfs_j,
            'scene': scene,
            'seq': seq,
            'name_i': name_i,
            'name_j': name_j,
        }
        return ret

    def __len__(self):
        return len(self.pair_keys)

    def __del__(self):
        self.dispose()

    def __repr__(self):
        return 'PointCloudPairTrainDataset: {}'.format(self.pcd_root)

    def dispose(self):
        if self.pcd_db is not None:
            self.pcd_db.close()
            self.pcd_db = None
