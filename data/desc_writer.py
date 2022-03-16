import os.path as osp
import sys
import numpy as np
import torch

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils import lmdb


# ---------------------------------------------------------------------------- #
# Descriptor result
# ---------------------------------------------------------------------------- #
class DescriptorWriter(object):

    def __init__(self, desc_root):
        self.desc_db = lmdb.create_db(desc_root)

        self.pcd_keys = list()
        self.timings = list()

    def append(self, scene, seq, name, desc, scales=None, timing=None):
        if type(desc).__module__ != np.__name__:
            desc = desc.detach().cpu().numpy()
        key = '/{}/{}/{}'.format(scene, seq, name)
        lmdb.write_db_once(self.desc_db, key, desc)
        self.pcd_keys.append(key)

        if scales is not None:
            if type(scales).__module__ != np.__name__:
                scales = scales.detach().cpu().numpy()
            lmdb.write_db_once(self.desc_db, key + '/scales', scales)

        if timing is not None:
            self.timings.append(timing)

    def close(self):
        if self.desc_db is not None:
            lmdb.write_db_once(self.desc_db, '__keys__', self.pcd_keys)
            if len(self.timings) > 0:
                lmdb.write_db_once(self.desc_db, '__time__', self.timings)
            self.desc_db.sync()
            self.desc_db.close()
            self.desc_db = None

    def __del__(self):
        self.close()
