# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import open3d as o3d  # prevent loading error
import sys
import logging
import torch
from DGRutil.params import DGRParams
from os.path import join
from datetime import datetime
from dateutil import tz
import json
import os
from config.config import get_config
from easydict import EasyDict as edict
from datasets.DGR.data_loaders import make_data_loader
from core.trainer import WeightedProcrustesTrainer

torch.manual_seed(0)
torch.cuda.manual_seed(0)


def setLog(logname):
    # ch = logging.StreamHandler(sys.stdout)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # create a handler to write the log
    # fh = logging.handlers.TimedRotatingFileHandler(logname, when='M', interval=1, backupCount=5,encoding='utf-8')  # 指定utf-8格式编码，避免输出的日志文本乱码
    fh = logging.FileHandler(logname, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)

    # 创建一个handler，用于将日志输出到控制台
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)

    # output format
    formatter = logging.Formatter('%(asctime)s %(message)s',
                                  datefmt='%m/%d %H:%M:%S')
    fh.setFormatter(formatter)
    # ch.setFormatter(formatter)

    logger.addHandler(fh)
    # logger.addHandler(ch)


def main(config, resume=False):

    train_loader = make_data_loader(config,
                                    config.train_phase,
                                    config.batch_size,
                                    num_workers=config.train_num_workers,
                                    shuffle=True)

    if config.test_valid:
        val_loader = make_data_loader(config,
                                      config.val_phase,
                                      config.val_batch_size,
                                      num_workers=config.val_num_workers,
                                      shuffle=True)
    else:
        val_loader = None

    trainer = WeightedProcrustesTrainer(
        config=config,
        data_loader=train_loader,
        val_data_loader=val_loader,
    )

    trainer.train()

def load_config():
    #  get parameter information
    config = get_config()
    dconfig = vars(config)

    resume_config = json.load(open(dconfig['config_file'], 'r'))
    for k in dconfig:
        if k not in ['resume_dir'] and k in resume_config:
            dconfig[k] = resume_config[k]

    tz_sh = tz.gettz('Asia/Shanghai')
    now = datetime.now(tz=tz_sh)

    dconfig['out_dir'] = join('./outputs/Experiment',
                              '{}-v{}'.format(dconfig['dataset'], dconfig['voxel_size']), dconfig['inlier_model'],
                                              '{}-lr{}-epoch{}-b{}-i{}-o{}'.format(dconfig['optimizer'], dconfig['lr'],
                                                                                   dconfig['max_epoch'],
                                                                                   dconfig['batch_size'],
                                                                                   dconfig['num_train_iter'],
                                                                                   dconfig['feat_model_n_out']),now.strftime("%m-%d-%H-%M-%S"))
    if not (os.path.exists(dconfig['out_dir'])):
        os.makedirs(dconfig['out_dir'], 777)
        os.chmod(dconfig['out_dir'], mode=0o777)

    return dconfig


if __name__ == "__main__":
    dconfig = load_config()

    setLog(os.path.join(dconfig['out_dir'], 'output.log'))

    # 读入之前的配置
    if dconfig['resume_dir']:
        resume_config = json.load(open(dconfig['resume_dir'] + '/config.json', 'r'))
        for k in dconfig:
            if k not in ['resume_dir'] and k in resume_config:
                dconfig[k] = resume_config[k]
        #  之前训练得到的模型
        dconfig['resume'] = resume_config['out_dir'] + '/checkpoint.pth'

    # Convert to dict
    config = edict(dconfig)
    main(config)
