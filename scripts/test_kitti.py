# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import os
import sys
import logging
import argparse
import numpy as np
import open3d as o3d


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from scripts.test_3dmatch import rte_rre

from config.config import get_config

from core.deep_global_registration import DeepGlobalRegistration

from datasets.DGR.kitti_loader import KITTINMPairDataset
from datasets.DGR.base_loader import CollationFunctionFactory

from DGRutil.pointcloud import make_open3d_point_cloud, make_open3d_feature, pointcloud_to_spheres, \
    find_matching_lines, pointcloud_to_boxes
from DGRutil.timer import AverageMeter, Timer
from os.path import join
from datetime import datetime
from dateutil import tz
import json
from tqdm import tqdm
from datasets.DGR.data_loaders import make_data_loader

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])

TE_THRESH = 5       # m
RE_THRESH = 10      # deg
VISUALIZE = True


def visualize_pair(xyz0, xyz1, T, voxel_size):
    # visualize the pair before and after transformation

    # pcd0 = pointcloud_to_spheres(xyz0,
    #                              voxel_size,
    #                              np.array([0, 0, 1]),
    #                              sphere_size=0.6)
    # pcd1 = pointcloud_to_spheres(xyz1,
    #                              voxel_size,
    #                              np.array([0, 1, 0]),
    #                              sphere_size=0.6)
    pcd0 = make_open3d_point_cloud(xyz0, np.array([0, 0, 1]))
    pcd1 = make_open3d_point_cloud(xyz1, np.array([0, 1, 0]))
    o3d.visualization.draw_geometries([pcd0, pcd1], window_name="origin")
    pcd0.transform(T)
    o3d.visualization.draw_geometries([pcd0, pcd1], window_name="registration")


def analyze_stats(stats):
    print('Total result mean')
    print(stats.mean(0))

    sel_stats = stats[stats[:, 0] > 0]
    print(sel_stats.mean(0))


def evaluate(config, data_loader, method):
    data_timer = Timer()

    test_iter = data_loader.__iter__()
    N = len(test_iter)

    stats = np.zeros((N, 5))  # bool succ, rte, rre, time, drive id

    for i in tqdm(range(len(data_loader))):
        data_timer.tic()
        try:
            data_dict = test_iter.next()
        except ValueError as exc:
            continue
        data_timer.toc()

        drive = data_dict['extra_packages'][0]['drive']
        xyz0, xyz1 = data_dict['pcd0'][0], data_dict['pcd1'][0]
        T_gt = data_dict['T_gt'][0].numpy()
        xyz0np, xyz1np = xyz0.numpy(), xyz1.numpy()

        T_pred = method.register(xyz0np, xyz1np)


        stats[i, :3] = rte_rre(T_pred, T_gt, TE_THRESH, RE_THRESH)
        stats[i, 3] = method.reg_timer.diff + method.feat_timer.diff
        stats[i, 4] = drive

        if stats[i, 0] == 0:
            logging.info(f"Failed with RTE: {stats[i, 1]}, RRE: {stats[i, 2]}")

        # if VISUALIZE:
        #     visualize_pair(xyz0, xyz1, T_pred, config.voxel_size)

    succ_rate, rte, rre, avg_time, _ = stats.mean(0)
    logging.info(
        f"Data time: {data_timer.avg}, Feat time: {method.feat_timer.avg}," +
        f" Reg time: {method.reg_timer.avg}, RTE: {rte}," +
        f" RRE: {rre}, Success: {succ_rate * 100} %")

    # Save results
    filename = f'kitti-stats_{method.__class__.__name__}'
    if os.path.isdir(config.out_dir):
        out_file = os.path.join(config.out_dir, filename)
    else:
        out_file = filename  # save it on the current directory
    print(f'Saving the stats to {out_file}')
    np.savez(out_file, stats=stats)
    # analyze_stats(stats)


class Dict2Obj(object):
    """convert a dictionary into a class"""

    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])

    def __repr__(self):
        """print 或str 时，让实例对象以字符串格式输出"""
        return "<Dict2Obj: %s>" % self.__dict__

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
    dconfig = Dict2Obj(dconfig)

    return dconfig


if __name__ == '__main__':
    config = load_config()

    dgr = DeepGlobalRegistration(config)

    # only get the first one of the batch for testing
    test_loader = make_data_loader(config,
                                    config.test_phase,
                                    1,
                                    num_workers=config.test_num_workers,
                                    shuffle=False)

    evaluate(config, test_loader, dgr)
