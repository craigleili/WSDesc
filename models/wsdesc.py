from __future__ import print_function
from __future__ import division

import os
import os.path as osp
import sys
import logging
import time
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from collections import defaultdict

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from models.cnn3d import cnn3ds
from diffvoxel.voxelization import voxelize, create_voxel_grids, transform_voxel_grids
from utils.loss import RigidityLoss, pairwise_distance_matrix

log = logging.getLogger(__name__)


def is_invalid_tensor(x):
    return torch.isnan(x).any().item() or torch.isinf(x).any().item()


class WSDesc(pl.LightningModule):

    def __init__(self, **hparams):
        super().__init__()

        self.hparams = hparams

        self.edge_length = hparams['voxelization.edge_length']
        self.resolution = hparams['voxelization.resolution']
        self.sigma = hparams['voxelization.sigma']

        voxel_size = self.edge_length / self.resolution

        voxel_grids = create_voxel_grids(self.edge_length, self.resolution)
        voxel_grids = np.reshape(voxel_grids, (-1, 3))
        self.register_buffer('voxel_grids', torch.from_numpy(voxel_grids))  # (V, 3)

        voxel_radius = voxel_size / 2.0
        voxel_radii = np.ones((len(voxel_grids),), dtype=np.float32) * voxel_radius
        self.register_buffer('voxel_radii', torch.from_numpy(voxel_radii))  # (V, )

        self.voxel_scales = nn.Parameter(
            torch.as_tensor([0.5 * hparams['transformer.max_scale']], dtype=torch.float32))

        self.conv3d = cnn3ds[hparams['conv3d.type']](
            in_channels=hparams['conv3d.in_channels'],
            desc_dim=hparams['conv3d.desc_dim'],
            fusion=hparams['conv3d.fusion'],
            dropout=hparams['conv3d.dropout'],
            dropout_prob=hparams['conv3d.dropout_prob'],
            weight_normalize=hparams['conv3d.weight_normalize'],
        )

        self.rigidity = RigidityLoss(inlier_threshold=hparams['loss.inlier_threshold'],
                                     loss_type=hparams['loss.criterion_type'])

    def forward(self, pcds, kpts, kpt_patches, lrfs, l2_normalize):
        # pcds: [(N, 3), ...]
        # kpts: [(K, 3), ...]
        # kpt_patches: [(K, P, 3), ...]
        # lrfs: [(K, 3, 3), ...]
        # l2_normalize: bool

        assert len(pcds) > 0 and len(kpts) == len(kpt_patches) <= len(pcds)

        all_descs = list()
        all_rotations = list()
        all_scales = list()
        for i in range(len(kpts)):
            K = kpts[i].size(0)

            scales = torch.clamp(self.voxel_scales,
                                 min=self.hparams['transformer.min_scale'],
                                 max=self.hparams['transformer.max_scale'])
            scales = scales.view(1, 1).expand(K, 1)

            rotations = torch.unsqueeze(lrfs[i], 1)  # (K, 1, 3, 3)
            if rotations.size(1) != scales.size(1):
                # (K, S, 3, 3)
                rotations = rotations.expand(rotations.size(0), scales.size(1),
                                             rotations.size(2), rotations.size(3))

            # (K, S, V, 3)
            voxel_positions = transform_voxel_grids(self.voxel_grids, scales, rotations,
                                                    kpts[i])
            S, V, D = voxel_positions.shape[1:]
            voxel_positions = voxel_positions.view(-1, V, D)  # (K*S, V, 3)

            # (K*S, V)
            voxel_radii = self.voxel_radii.view(1, -1) * scales.view(-1, 1)

            # (K*S, V)
            voxel_vals = voxelize(pcds[i], voxel_positions, voxel_radii, self.sigma)
            voxel_vals = voxel_vals.view(K, S, 1, self.resolution, self.resolution,
                                         self.resolution)

            kpt_descs = self.conv3d(voxel_vals, l2_normalize)  # (K, C')

            all_descs.append(kpt_descs)
            all_rotations.append(rotations)
            all_scales.append(scales)

        stats = {
            'rotations': all_rotations,
            'scales': all_scales,
        }
        return all_descs, stats

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        hparams = self.hparams

        l2_normalize = True

        get_data = lambda label: batch[label][0]

        pcd_i = get_data('pcd_i')
        pcd_j = get_data('pcd_j')
        kpts_i = get_data('kpts_i')
        kpts_j = get_data('kpts_j')
        kpt_indices_i = get_data('kpt_indices_i').long()
        kpt_indices_j = get_data('kpt_indices_j').long()
        patches_i = get_data('patches_i')
        patches_j = get_data('patches_j')
        lrfs_i = get_data('lrfs_i')
        lrfs_j = get_data('lrfs_j')

        descs, stats = self([pcd_i, pcd_j], [kpts_i, kpts_j], [patches_i, patches_j],
                            [lrfs_i, lrfs_j], l2_normalize)

        descs_i, descs_j = descs

        loss = 0.0

        desc_dists_ij = pairwise_distance_matrix(descs_i, descs_j, squared=False)
        loss_ortho, loss_cycle = self.rigidity(desc_dists_ij, kpts_i, kpts_j, pcd_i, pcd_j)
        if hparams['loss.ortho_weight'] > 0:
            loss = loss + hparams['loss.ortho_weight'] * loss_ortho
        if hparams['loss.cycle_weight'] > 0:
            loss = loss + hparams['loss.cycle_weight'] * loss_cycle

        if is_invalid_tensor(loss) or loss.item() > hparams['loss.max_thresh']:
            return None

        return loss

    def on_after_backward(self):
        flag = False
        for name, param in self.named_parameters():
            if hasattr(param, 'grad') and param.grad is not None:
                if is_invalid_tensor(param.grad):
                    flag = True
                    break
        if flag:
            self.zero_grad()

    def configure_optimizers(self):
        hparams = self.hparams

        params_dv = [self.voxel_scales]

        optimizer_dv = torch.optim.Adam(params_dv, lr=hparams['optim.lr'], weight_decay=0.002)
        lr_scheduler_dv = torch.optim.lr_scheduler.LambdaLR(optimizer_dv,
                                                            lr_lambda=lambda epoch: 1.0)

        params_bb = list(self.conv3d.parameters())
        optimizer_bb = torch.optim.Adam(params_bb,
                                        lr=hparams['optim.lr'],
                                        weight_decay=hparams['optim.weight_decay'])
        lr_scheduler_bb = torch.optim.lr_scheduler.LambdaLR(optimizer_bb,
                                                            lr_lambda=lambda epoch: 1.0)

        return [optimizer_dv, optimizer_bb], [{
            'scheduler': lr_scheduler_dv,
            'interval': 'step'
        }, {
            'scheduler': lr_scheduler_bb,
            'interval': 'step'
        }]

    def test_step(self, batch, batch_idx, *args):
        hparams = self.hparams

        l2_normalize = True
        batch_size = hparams['data.num_samples']

        get_data = lambda label: batch[label][0]

        pcd = get_data('pcd')
        kpts = get_data('kpts')
        kpt_indices = get_data('kpt_indices').long()
        patches = get_data('patches')
        lrfs = get_data('lrfs')
        scene = get_data('scene')
        seq = get_data('seq')
        name = get_data('name')

        num_kpts = len(kpt_indices)
        all_descs = list()
        all_stats = defaultdict(list)

        for i in range(0, num_kpts, batch_size):
            j = min(i + batch_size, num_kpts)

            descs, stats = self([pcd], [kpts[i:j, :]], [patches[i:j, :, :]], [lrfs[i:j, :, :]],
                                l2_normalize)

            descs = descs[0]
            all_descs.append(descs.cpu().numpy())

            for k, v in stats.items():
                all_stats[k].append(v[0].cpu().numpy())

        all_descs = np.concatenate(all_descs, axis=0)
        assert len(all_descs) == num_kpts
        all_stats = {k: np.concatenate(v, axis=0) for k, v in all_stats.items()}
        assert len(all_stats['scales']) == num_kpts

        self.desc_saver.append(scene, seq, name, all_descs)

    def test_epoch_end(self, outputs):
        self.desc_saver.close()

    def setup(self, stage):
        from data.desc_writer import DescriptorWriter

        hparams = self.hparams

        name = self.hparams['data.test.pcd_path'].split('/')[-1].split('.')[0]
        exp_time = time.strftime('%y-%m-%d_%H-%M-%S')
        self.desc_saver = DescriptorWriter(
            osp.join(hparams['log_dir'], '{}_desc_{}.lmdb'.format(name, exp_time)))

    def train_dataloader(self):
        return self.create_train_dataloader(self.hparams)

    def test_dataloader(self):
        return self.create_test_dataloader(self.hparams)

    @staticmethod
    def create_train_dataloader(hparams):
        from data.dataset import PointCloudPairTrainDataset

        dataset = PointCloudPairTrainDataset(
            pcd_root=osp.join(hparams['data.root'], hparams['data.train.pcd_path']),
            num_samples=hparams['data.num_samples'],
            num_points_per_sample=hparams['data.num_points_per_sample'],
            sample_radius=hparams['data.sample_radius'],
            rot_augment=hparams['data.train.rot_augment'],
        )
        log.info(str(dataset))
        loader = DataLoader(
            dataset,
            hparams['data.batch_size'],
            shuffle=True,
            num_workers=hparams['data.num_workers'],
            pin_memory=True,
            drop_last=True,
        )
        return loader

    @staticmethod
    def create_test_dataloader(hparams):
        from data.dataset import PointCloudDataset

        dataset = PointCloudDataset(
            data_root=osp.join(hparams['data.root'], hparams['data.test.pcd_path']),
            num_points_per_sample=hparams['data.num_points_per_sample'],
            sample_radius=hparams['data.sample_radius'],
        )
        log.info(str(dataset))
        loader = DataLoader(
            dataset,
            hparams['data.batch_size'],
            shuffle=False,
            num_workers=hparams['data.num_workers'],
            pin_memory=False,
            drop_last=False,
        )
        return loader
