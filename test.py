from __future__ import print_function
from __future__ import division

import argparse
import os
import os.path as osp
import sys
import logging
import pytorch_lightning as pl
from pathlib import Path

ROOT_DIR = osp.abspath(osp.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from models.wsdesc import WSDesc

log = logging.getLogger(__name__)


def main(cfg):
    assert cfg.ckpt is not None and osp.isfile(cfg.ckpt)

    hparams_path = osp.join(Path(cfg.ckpt).parent, 'hparams.yaml')
    model = WSDesc.load_from_checkpoint(cfg.ckpt, hparams_file=hparams_path)
    log.info('Load ckpt from {}'.format(cfg.ckpt))

    hparams = model.hparams
    pl.seed_everything(hparams['seed'])

    # Data
    loader = WSDesc.create_test_dataloader(hparams)
    trainer = pl.Trainer(progress_bar_refresh_rate=hparams['progressbar_step'],
                         gpus=hparams['gpus'])
    trainer.test(model, test_dataloaders=loader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='exp/3dmatch')
    return parser.parse_args()


if __name__ == '__main__':
    import torch
    #import torch.multiprocessing
    #torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    cfg = parse_args()
    main(cfg)
