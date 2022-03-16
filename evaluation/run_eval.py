from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import os.path as osp
import sys
import pickle
import subprocess
import numpy as np
import yaml
from pathlib import Path

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils import lmdb
from utils.io import may_create_folder, sorted_alphanum
from utils.match3d import write_log
from utils.pcd import register_with_ransac

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- #
# Ref:
# - https://github.com/zgojcic/3DSmoothNet/tree/master/evaluation
# ---------------------------------------------------------------------------- #

INLIER_RATIO_THRESHES = (np.arange(0, 31, dtype=np.float32) * 0.3 / 30).tolist()
INLIER_THRESHES = (np.arange(0, 21, dtype=np.float32) * 0.2 / 20).tolist()


# ---------------------------------------------------------------------------- #
# Scene data readers
# ---------------------------------------------------------------------------- #
class SceneDataReader(object):

    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.db = lmdb.open_db(root_dir)

        self.seqs = dict()
        all_keys = lmdb.read_db_once(self.db, '__keys__')
        for key in all_keys:
            _, seq, name = key.split('/')
            if seq not in self.seqs:
                self.seqs[seq] = list()
            self.seqs[seq].append(name)

    def get_scene_name(self):
        return lmdb.read_db_once(self.db, '__scene__')

    def get_sequence_names(self):
        return sorted_alphanum(self.seqs.keys())

    def get_pcd_names(self, seq):
        return sorted_alphanum(self.seqs[seq])

    def get_points(self, seq, name):
        return lmdb.read_db_once(self.db, '/{}/{}/points'.format(seq, name))

    def get_kpts(self, seq, name):
        return lmdb.read_db_once(self.db, '/{}/{}/kpts'.format(seq, name))

    def get_poses(self, seq):
        return lmdb.read_db_once(self.db, '/{}/poses'.format(seq))

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None


# ---------------------------------------------------------------------------- #
# Desc readers
# ---------------------------------------------------------------------------- #
class LmdbDescReader(object):
    """Read our desciptor from a lmdb file
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.db = lmdb.open_db(root_dir)

    def read(self, scene, seq, name):
        data = lmdb.read_db_once(self.db, '/{}/{}/{}'.format(scene, seq, name))
        return np.nan_to_num(data)

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None


# ---------------------------------------------------------------------------- #
# Utilities
# ---------------------------------------------------------------------------- #
def knn_search(points_src, points_dst, k=1):
    """
    Args:
        points_src (np.array): (N, 3)
        points_dst (np.array): (M, 3)
        k (int):

    Returns:
        np.array: (N, k)
    """
    import open3d as o3d

    kdtree = o3d.geometry.KDTreeFlann(np.asarray(points_dst, dtype=np.float64).transpose())
    points_src = np.asarray(points_src, dtype=np.float64)
    # (k, indices, distance2)
    nnindices = [
        kdtree.search_knn_vector_xd(points_src[i, :], k)[1] for i in range(len(points_src))
    ]
    if k == 1:
        return np.asarray(nnindices, dtype=np.int32)[:, 0]
    else:
        return np.asarray(nnindices, dtype=np.int32)


def compute_overlap_ratio(points_i,
                          points_j,
                          trans,
                          method,
                          dist_thresh=0.05,
                          voxel_size=0.025):
    """Ref:
    - https://github.com/zgojcic/3D_multiview_reg/blob/master/lib/utils.py

    Args:
        points_i (np.array): (N, 3)
        points_j (np.array): (M, 3)
        trans (np.array): (4, 4), from j to i
        method (str): ['3DMatch', 'FCGF']
        dist_thresh (float):
        voxel_size (float):

    Returns:
        float:
    """
    import open3d as o3d

    pcd_i = o3d.geometry.PointCloud()
    pcd_i.points = o3d.utility.Vector3dVector(np.copy(points_i))
    pcd_j = o3d.geometry.PointCloud()
    pcd_j.points = o3d.utility.Vector3dVector(np.copy(points_j))

    trans_inv = np.linalg.inv(trans)

    if method == '3DMatch':
        pcd_i.transform(trans_inv)
        points_i_t = np.asarray(pcd_i.points)

        pcd_j.transform(trans)
        points_j_t = np.asarray(pcd_j.points)

    elif method == 'FCGF':
        pcd_i_down = pcd_i.voxel_down_sample(voxel_size)
        points_i = np.copy(np.asarray(pcd_i_down.points))
        pcd_i_down.transform(trans_inv)
        points_i_t = np.asarray(pcd_i_down.points)

        pcd_j_down = pcd_j.voxel_down_sample(voxel_size)
        points_j = np.copy(np.asarray(pcd_j_down.points))
        pcd_j_down.transform(trans)
        points_j_t = np.asarray(pcd_j_down.points)

        dist_thresh = 3 * voxel_size

    else:
        raise RuntimeError('Method - {} is not supported!'.format(method))

    nnindices_ij = knn_search(points_i, points_j_t, k=1)
    dists_ij = np.linalg.norm(points_i - points_j_t[nnindices_ij, :], axis=1)
    overlap_ij = np.sum(dists_ij < dist_thresh) / float(len(points_i))

    nnindices_ji = knn_search(points_j, points_i_t, k=1)
    dists_ji = np.linalg.norm(points_j - points_i_t[nnindices_ji, :], axis=1)
    overlap_ji = np.sum(dists_ji < dist_thresh) / float(len(points_j))

    return max(overlap_ij, overlap_ji)


def compute_registration_recall(gtlog_path, gtinfo_path, result_path):
    """Use Octave to run .m files

    Args:
        gtlog_path (str):
        gtinfo_path (str):
        result_path (str):

    Returns:
        float, float:
    """
    toolbox_path = osp.join(ROOT_DIR, 'evaluation', '3dmatch-toolbox')

    result = subprocess.run(
        [
            'octave', '-q',
            osp.join(toolbox_path, 'computeRegistrationMetrics.m'),
            osp.join(toolbox_path, 'external', 'ElasticReconstruction'), gtlog_path,
            gtinfo_path, result_path
        ],
        encoding='utf-8',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    metrics = [float(x) for x in result.stdout.strip().split(os.linesep)]
    recall, precision = metrics[0], metrics[1]
    return precision, recall


# ---------------------------------------------------------------------------- #
# Evaluation
# ---------------------------------------------------------------------------- #
def register_fragment_pair(scene_name, seq_name, frag1_name, frag2_name, cloud_reader,
                           desc_reader, poses, run_ransac):
    """Ref:
    - https://github.com/andyzeng/3dmatch-toolbox/blob/master/evaluation/geometric-registration/register2Fragments.m

    Args:
        scene_name (str): 
        seq_name (str): 
        frag1_name (str): cloud_bin_*
        frag2_name (str): cloud_bin_*
        cloud_reader (SceneDataReader):
        desc_reader (DescReader): 
        poses (list): 
        run_ransac (bool):

    Returns:
        dict: 
    """
    import open3d as o3d

    logger.debug('  Start {} - {} - {} - {}'.format(scene_name, seq_name, frag1_name,
                                                    frag2_name))

    frag1_id = int(frag1_name.split('_')[-1])
    frag2_id = int(frag2_name.split('_')[-1])
    assert frag1_id < frag2_id

    overlap_pid = -1
    for pid, pose in enumerate(poses):
        # (id0, id1, id2, mat, corr)
        if pose[0] == frag1_id and pose[1] == frag2_id:
            overlap_pid = pid
            break
    if overlap_pid < 0:
        return {
            'num_inliers': [],
            'inlier_ratios': [],
            'gt_flag': 0,
            'ransac_pose': np.identity(4, dtype=np.float64),
            'ransac_time': 0.,
            'ransac_overlap_ratio': 0.,
            'ransac_rte': 1e8,
            'ransac_rre': 1e8
        }

    frag1_points = cloud_reader.get_points(seq_name, frag1_name)
    frag2_points = cloud_reader.get_points(seq_name, frag2_name)

    frag1_kpts = cloud_reader.get_kpts(seq_name, frag1_name)
    frag2_kpts = cloud_reader.get_kpts(seq_name, frag2_name)

    frag1_descs = desc_reader.read(scene_name, seq_name, frag1_name)
    frag2_descs = desc_reader.read(scene_name, seq_name, frag2_name)

    assert len(frag1_kpts) == len(frag1_descs)
    assert len(frag2_kpts) == len(frag2_descs)

    frag21_nnindices = knn_search(frag2_descs, frag1_descs)
    assert frag21_nnindices.ndim == 1

    frag12_nnindices = knn_search(frag1_descs, frag2_descs)
    assert frag12_nnindices.ndim == 1

    frag2_match_indices = np.flatnonzero(
        np.equal(np.arange(len(frag21_nnindices)), frag12_nnindices[frag21_nnindices]))
    frag1_match_indices = frag21_nnindices[frag2_match_indices]
    frag2_match_kpts = frag2_kpts[frag2_match_indices, :]
    frag1_match_kpts = frag1_kpts[frag1_match_indices, :]

    frag2_pcd_tmp = o3d.geometry.PointCloud()
    frag2_pcd_tmp.points = o3d.utility.Vector3dVector(np.copy(frag2_match_kpts))
    frag2_pcd_tmp.transform(poses[overlap_pid][3])
    distances = np.sqrt(
        np.sum(np.square(frag1_match_kpts - np.asarray(frag2_pcd_tmp.points)), axis=1))

    num_inliers = [np.sum(distances < inlier_thresh) for inlier_thresh in INLIER_THRESHES]
    inlier_ratios = [float(ni) / len(distances) for ni in num_inliers]
    gt_flag = 1

    if run_ransac:
        if len(frag1_kpts) < len(frag2_kpts):
            trans, duration = register_with_ransac(frag1_kpts, frag2_kpts, frag1_descs,
                                                   frag2_descs)
            trans = np.linalg.inv(trans)
        else:
            trans, duration = register_with_ransac(frag2_kpts, frag1_kpts, frag2_descs,
                                                   frag1_descs)
        overlap_ratio = compute_overlap_ratio(frag1_points, frag2_points, trans, 'FCGF')

        trans_gt = poses[overlap_pid][3]
        rte = np.linalg.norm(trans[:3, 3] - trans_gt[:3, 3])
        rre = (np.trace(np.transpose(trans[:3, :3]) @ trans_gt[:3, :3]) - 1.0) / 2.0
        rre = np.rad2deg(np.arccos(np.clip(rre, -1, 1)))

    else:
        trans = np.identity(4, dtype=np.float64)
        duration = 0.
        overlap_ratio = 0.
        rte = 1e8
        rre = 1e8

    return {
        'num_inliers': num_inliers,
        'inlier_ratios': inlier_ratios,
        'gt_flag': gt_flag,
        'ransac_pose': trans,
        'ransac_time': duration,
        'ransac_overlap_ratio': overlap_ratio,
        'ransac_rte': rte,
        'ransac_rre': rre
    }


def run_scene_matching(scene_name,
                       seq_name,
                       desc_type,
                       pcloud_root,
                       desc_root,
                       out_root,
                       run_ransac=True,
                       overlap_thresh=0.3):
    """
    Args:
        scene_name (str): 
        seq_name (str): 
        desc_type (str): 
        pcloud_root (str):
        desc_root (str): 
        out_root (str): 
        run_ransac (bool):
        overlap_thresh (float):

    Returns:
        str: 
    """
    out_folder = osp.join(out_root, desc_type)
    out_filename = '{} {}.txt'.format(scene_name, seq_name)
    if Path(osp.join(out_folder, out_filename)).is_file():
        print('[*] {} already exists. Skip computation.'.format(out_filename))
        return osp.join(out_folder, out_filename)

    cloud_reader = SceneDataReader(osp.join(pcloud_root, scene_name + '.lmdb'))
    fragment_names = cloud_reader.get_pcd_names(seq_name)
    n_fragments = len(fragment_names)
    poses = cloud_reader.get_poses(seq_name)

    register_results = list()
    desc_reader = LmdbDescReader(desc_root)
    for i in range(n_fragments):
        for j in range(i + 1, n_fragments):
            res_dict = register_fragment_pair(scene_name, seq_name, fragment_names[i],
                                              fragment_names[j], cloud_reader, desc_reader,
                                              poses, run_ransac)
            res_dict['frag1_name'] = fragment_names[i]
            res_dict['frag2_name'] = fragment_names[j]
            register_results.append(res_dict)
    desc_reader.close()
    cloud_reader.close()

    with open(osp.join(out_folder, out_filename), 'w') as fh:
        for k in register_results:
            fh.write('- frag1: {}\n'.format(k['frag1_name']))
            fh.write('  frag2: {}\n'.format(k['frag2_name']))
            fh.write('  num_inliers: "{}"\n'.format(' '.join(map(str, k['num_inliers']))))
            fh.write('  inlier_ratios: "{}"\n'.format(' '.join(map(str, k['inlier_ratios']))))
            fh.write('  gt_flag: {}\n'.format(k['gt_flag']))
            fh.write('  ransac_pose: "{}"\n'.format(' '.join(
                map(str, k['ransac_pose'].flatten().tolist()))))
            fh.write('  ransac_time: {}\n'.format(k['ransac_time']))
            fh.write('  ransac_overlap_ratio: {}\n'.format(k['ransac_overlap_ratio']))
            fh.write('  ransac_rte: {}\n'.format(k['ransac_rte']))
            fh.write('  ransac_rre: {}\n'.format(k['ransac_rre']))

    poses_est = [(int(k['frag1_name'].split('_')[-1]), int(k['frag2_name'].split('_')[-1]),
                  n_fragments, k['ransac_pose'])
                 for k in register_results
                 if k['gt_flag'] == 1 and k['ransac_overlap_ratio'] >= overlap_thresh]
    write_log(osp.join(out_folder, out_filename[:-4] + '.log'), poses_est)

    return osp.join(out_folder, out_filename)


def compute_metrics(match_paths,
                    desc_type,
                    run_ransac,
                    gt_root,
                    out_root,
                    rte_thresh,
                    rre_thresh,
                    scene_abbr_fn=None):
    """
    Args:
        match_paths (str): 
        desc_type (str): 
        run_ransac (bool): 
        gt_root (str): 
        out_root (str): 
        rte_thresh (float):
        rre_thresh (float):
        scene_abbr_fn (lambda):
    
    Returns:
        str: 
    """
    scenes = list()
    all_feature_match_recalls = list()
    all_mean_num_inliers = list()
    all_mean_inlier_ratios = list()
    all_ransac_times = list()
    all_ransac_rtes = list()
    all_ransac_rres = list()
    all_success_rates = list()
    all_registration_precisions = list()
    all_registration_recalls = list()

    for match_path in match_paths:
        scene_name_str, seq_name_str = Path(match_path).stem.split()
        if scene_abbr_fn is not None:
            scenes.append(scene_abbr_fn(scene_name_str))
        else:
            scenes.append(scene_name_str)

        seq_num_inliers = list()
        seq_inlier_ratios = list()
        seq_ransac_times = list()
        seq_ransac_rtes = list()
        seq_ransac_rres = list()
        with open(match_path, 'r') as fh:
            for item in yaml.full_load(fh):
                if int(item['gt_flag']) != 1:
                    continue
                seq_num_inliers.append(list(map(int, item['num_inliers'].split())))
                seq_inlier_ratios.append(list(map(float, item['inlier_ratios'].split())))
                seq_ransac_times.append(float(item['ransac_time']))
                seq_ransac_rtes.append(float(item['ransac_rte']))
                seq_ransac_rres.append(float(item['ransac_rre']))
        seq_num_inliers = np.asarray(seq_num_inliers, dtype=np.int32)  # (N, 20)
        seq_inlier_ratios = np.asarray(seq_inlier_ratios, dtype=np.float32)  # (N, 20)
        seq_ransac_times = np.asarray(seq_ransac_times, dtype=np.float32)  # (N, )
        seq_ransac_rtes = np.asarray(seq_ransac_rtes, dtype=np.float32)  # (N, )
        seq_ransac_rres = np.asarray(seq_ransac_rres, dtype=np.float32)  # (N, )

        seq_feature_match_recalls = [
            np.sum(seq_inlier_ratios > inlier_ratio_thresh, axis=0) /
            float(len(seq_inlier_ratios)) for inlier_ratio_thresh in INLIER_RATIO_THRESHES
        ]
        seq_feature_match_recalls = np.asarray(seq_feature_match_recalls, dtype=np.float32)
        all_feature_match_recalls.append(seq_feature_match_recalls)

        all_mean_num_inliers.append(np.mean(seq_num_inliers, axis=0))  # (20, )
        all_mean_inlier_ratios.append(np.mean(seq_inlier_ratios, axis=0))  # (20, )

        if run_ransac:
            all_ransac_times.append(np.mean(seq_ransac_times))

            ransac_success_flags = np.logical_and(seq_ransac_rtes < rte_thresh,
                                                  seq_ransac_rres < rre_thresh)
            all_success_rates.append(
                float(np.sum(ransac_success_flags)) / len(ransac_success_flags))

            all_ransac_rtes.append(np.mean(seq_ransac_rtes[ransac_success_flags]))
            all_ransac_rres.append(np.mean(seq_ransac_rres[ransac_success_flags]))

            gtlog_path = osp.join(gt_root, scene_name_str, seq_name_str, 'gt.log')
            gtinfo_path = osp.join(gt_root, scene_name_str, seq_name_str, 'gt.info')
            resultlog_path = match_path[:-4] + '.log'
            reg_precision, reg_recall = compute_registration_recall(
                gtlog_path, gtinfo_path, resultlog_path)

            all_registration_precisions.append(reg_precision)
            all_registration_recalls.append(reg_recall)

    out_path = osp.join(out_root, '{}-metrics'.format(desc_type))
    with open(out_path + '.pkl', 'wb') as fh:
        to_save = {
            'scenes': scenes,
            'feature_match_recalls': all_feature_match_recalls,
            'mean_num_inliers': all_mean_num_inliers,
            'mean_inlier_ratios': all_mean_inlier_ratios,
            'ransac_times': all_ransac_times,
            'ransac_rtes': all_ransac_rtes,
            'ransac_rres': all_ransac_rres,
            'success_rates': all_success_rates,
            'registration_precisions': all_registration_precisions,
            'registration_recalls': all_registration_recalls,
            'inlier_threshes': INLIER_THRESHES,
            'inlier_ratio_threshes': INLIER_RATIO_THRESHES
        }
        pickle.dump(to_save, fh, protocol=pickle.HIGHEST_PROTOCOL)

    with open(out_path + '.csv', 'w') as fh:
        inlier_threshes = [(5, 0.05), (10, 0.1)]
        inlier_ratio_threshes = [(5, 0.05), (20, 0.2)]

        header_str = 'SceneName'
        for it in inlier_threshes:
            tau_1 = it[1]
            for irt in inlier_ratio_threshes:
                tau_2 = irt[1]
                header_str += ',FMR-{:.2f}t1-{:.2f}t2'.format(tau_1, tau_2)
            header_str += ',AvgInlierRatios-{:.2f}t1'.format(tau_1)
            header_str += ',AvgInliers-{:.2f}t1'.format(tau_1)
        if run_ransac:
            header_str += ',RPrecisions'
            header_str += ',RRecalls'
            header_str += ',RTEs'
            header_str += ',RREs'
            header_str += ',SuccessRate'
            header_str += ',RTimes'
        fh.write(header_str + '\n')

        datatab = list()
        for sid, scene_name in enumerate(scenes):
            row = list()
            for it in inlier_threshes:
                for irt in inlier_ratio_threshes:
                    row.append(all_feature_match_recalls[sid][irt[0], it[0]])
                row.append(all_mean_inlier_ratios[sid][it[0]])
                row.append(all_mean_num_inliers[sid][it[0]])
            if run_ransac:
                row.append(all_registration_precisions[sid])
                row.append(all_registration_recalls[sid])
                row.append(all_ransac_rtes[sid])
                row.append(all_ransac_rres[sid])
                row.append(all_success_rates[sid])
                row.append(all_ransac_times[sid])
            datatab.append(row)

            row_str = scene_name
            for col in row:
                row_str += ',{:.3f}'.format(col)
            fh.write(row_str + '\n')
        datatab = np.asarray(datatab, dtype=np.float32)

        avg_row_str = 'Average'
        std_row_str = 'Std'
        for col in range(datatab.shape[1]):
            avg_row_str += ',{:.3f}'.format(np.mean(datatab[:, col]))
            std_row_str += ',{:.3f}'.format(np.std(datatab[:, col]))
        fh.write(avg_row_str + '\n')
        fh.write(std_row_str + '\n')

    return out_path


def evaluate(cfg, desc_types, desc_roots):
    scene_names = cfg['test_scene_names']
    scene_abbr_fn = lambda sn: cfg['test_scene_abbr_names'][cfg['test_scene_names'].index(sn)]

    pcd_root = cfg['pcd_root']
    gt_root = cfg['gt_root']
    out_root = cfg['log_dir']
    may_create_folder(out_root)
    run_ransac = cfg['run_ransac']
    rte_thresh = cfg['rte_thresh']
    rre_thresh = cfg['rre_thresh']
    seq_name = 'seq-01'

    stat_paths = list()
    for desc_type, desc_root in zip(desc_types, desc_roots):
        may_create_folder(osp.join(out_root, desc_type))

        logger.info('  Start {}'.format(desc_type))

        if cfg['threads'] > 1:
            from joblib import Parallel, delayed
            import multiprocessing
            # Multi-threading
            match_paths = Parallel(n_jobs=cfg['threads'])(delayed(run_scene_matching)(
                scene_name, seq_name, desc_type, pcd_root, desc_root, out_root, run_ransac)
                                                          for scene_name in scene_names)
        else:
            match_paths = list()
            for scene_name in scene_names:
                match_path = run_scene_matching(scene_name, seq_name, desc_type, pcd_root,
                                                desc_root, out_root, run_ransac)
                match_paths.append(match_path)

        # Compute metrics
        stat_path = compute_metrics(match_paths, desc_type, run_ransac, gt_root, out_root,
                                    rte_thresh, rre_thresh, scene_abbr_fn)
        stat_paths.append(stat_path)

    logger.info('All done.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='config.yaml')
    parser.add_argument('--desc_types', nargs='+')
    parser.add_argument('--desc_roots', nargs='+')
    return parser.parse_args()


if __name__ == '__main__':
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    args = parse_args()
    with open(args.cfg, 'r') as fh:
        cfg = yaml.full_load(fh)
    evaluate(cfg, args.desc_types, args.desc_roots)
