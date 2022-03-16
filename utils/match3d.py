import os.path as osp
import sys
import numpy as np

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils import io as uio


def read_log(filepath):
    """Ref:
    - http://redwood-data.org/indoor/fileformat.html
    Args:
        filepath (str):
    Returns:
        list: [tuple, ...]
    """
    lines = uio.read_lines(filepath)
    n_poses = len(lines) // 5
    poses = list()
    for i in range(n_poses):
        items = lines[i * 5].split()  # Meta line
        # pcd_i, pcd_j, num_pcds
        id0, id1, id2 = int(items[0]), int(items[1]), int(items[2])
        # pose that transforms pcd_j to the coordinate frame of pcd_i
        mat = np.zeros((4, 4), dtype=np.float64)
        for j in range(4):
            items = lines[i * 5 + j + 1].split()
            for k in range(4):
                mat[j, k] = float(items[k])
        poses.append((id0, id1, id2, mat))
    return poses


def write_log(filepath, poses):
    with open(filepath, 'w') as fh:
        for pose in poses:
            fh.write('{}\t {}\t {}\n'.format(pose[0], pose[1], pose[2]))
            for i in range(4):
                fh.write('{:.12f}  {:.12f}  {:.12f}  {:.12f}\n'.format(
                    pose[3][i, 0], pose[3][i, 1], pose[3][i, 2], pose[3][i, 3]))
