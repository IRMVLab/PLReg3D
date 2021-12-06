import os
import configparser
import time
import numpy as np

class DGRParams:
    """
    Params for training MinkLoc models on Oxford dataset
    """
    def __init__(self, params_path):
        """
        Configuration files
        :param path: General configuration file
        :param model_params: Model-specific configuration
        """
        assert os.path.exists(params_path), 'Cannot find configuration file: {}'.format(params_path)
        self.params_path = params_path
        self.config = configparser.ConfigParser()
        self.config.read(self.params_path)

    def load(self, args):

        params = self.config['DEFAULT']
        args.dataset = params.get('DATASET', 'KITTINMPairDataset')
        args.kitti_dir = params.get('KITTI_PATH', '/media/qzj/Dataset/slamDataSet/kitti/data_odometry_velodyne/')
        args.weights = params.get('FCGF_WEIGHTS', '/home/qzj/code/registration/DeepGlobalRegistration/pretrain/KITTI-v0.3-ResUNetBN2C-conv1-5-nout32.pth')

        params = self.config['TRAIN']
        args.feat_model = params.get('MODEL', 'ResUNetBN2C')
        args.inlier_model = params.get('INLIER_MODEL', 'ResUNetBN2C')
        args.feat_conv1_kernel_size = params.getint('CONV1_KERNEL_SIZE', 5)
        args.voxel_size = params.getfloat('VOXEL_SIZE', 0.3)
        args.use_random_scale = params.getboolean('RANDOM_SCALE', True)
        args.batch_size = params.getint('BATCH_SIZE', 8)
        args.val_batch_size = params.getint('BATCH_SIZE', 8)
        args.feat_model_n_out = params.getint('MODEL_N_OUT', 32)
        args.positive_pair_search_voxel_size_multiplier = params.getint('POSITIVE_PAIR_SEARCH_VOXEL_SIZE_MULTIPLIER', 4)
        args.optimizer = params.get('OPTIMIZER', 'SGD')
        args.lr = params.getfloat('LR', 1e-2)
        args.exp_gamma = params.getfloat('EXP_GAMMA', 0.99)
        args.max_epoch = params.getint('MAX_EPOCH', 100)
        args.iter_size = params.getint('ITER_SIZE', 1)
        args.success_rte_thresh = params.getfloat('success_rte_thresh', 2)
        args.success_rre_thresh = params.getfloat('success_rre_thresh', 5)

        return args
