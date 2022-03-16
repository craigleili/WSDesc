from __future__ import print_function
from __future__ import division

import argparse
import os
import os.path as osp
import sys
import logging
import yaml
import time
import pytorch_lightning as pl
from pathlib import Path

ROOT_DIR = osp.abspath(osp.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from models.wsdesc import WSDesc
from utils.io import may_create_folder

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #
def main(cfg_path):
    with open(cfg_path, 'r') as fh:
        hparams = yaml.full_load(fh)

    pl.seed_everything(hparams['seed'])

    hparams['log_dir'] = osp.join(hparams['log_dir'], time.strftime('%y-%m-%d_%H-%M-%S'))
    may_create_folder(hparams['log_dir'])

    model = WSDesc(**hparams)
    log.info(str(model))

    lr_cb = pl.callbacks.LearningRateMonitor('step')
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=hparams['log_dir'],
        filename='{epoch}',
        verbose=True,
        save_top_k=-1,
        period=hparams['data.valid.epoch_step'],
    )
    trainer = pl.Trainer(
        gradient_clip_val=hparams['optim.grad_clip'],
        progress_bar_refresh_rate=hparams['progressbar_step'],
        gpus=hparams['gpus'],
        max_epochs=hparams['data.train.epochs'],
        callbacks=[lr_cb],
        checkpoint_callback=ckpt_cb,
        check_val_every_n_epoch=hparams['data.valid.epoch_step'],
        num_sanity_val_steps=0,
    )

    # Save hparams to logger
    with open(osp.join(hparams['log_dir'], 'hparams.yaml'), 'w') as fh:
        yaml.dump(hparams, fh)

    trainer.fit(model)
    for _ in range(hparams['data.test.iterations']):
        trainer.test(ckpt_path=None)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', default='config/default.yaml')
    return parser.parse_args()


if __name__ == '__main__':
    import torch
    #import torch.multiprocessing
    #torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    cfg = parse_args()
    main(cfg.cfg_path)
