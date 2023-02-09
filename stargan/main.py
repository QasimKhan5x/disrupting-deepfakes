import os
import random

import numpy as np
import torch

from stargan.config import get_config
from stargan.data_loader import get_loader
from stargan.psolver import Disruptor


def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    
    config.outputs_dir = os.path.join(
        'stargan', 'experiments', config.outputs_dir)
    config.log_dir = os.path.join(config.outputs_dir, config.log_dir)
    config.model_save_dir = os.path.join(config.outputs_dir, config.model_save_dir)
    config.sample_dir = os.path.join(config.outputs_dir, config.sample_dir)
    config.result_dir = os.path.join(config.outputs_dir, config.result_dir)

    # Create directories if not exist.
    if not os.path.exists(config.outputs_dir):
        os.makedirs(config.outputs_dir)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    set_seed(config.seed)
    celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                config.celeba_crop_size, config.image_size, config.batch_size,
                                'CelebA', config.mode, config.num_workers)

    # Solver for training and testing DeepFake Disruptor
    solver = Disruptor(config, celeba_loader).cuda()
    solver.train()    


if __name__ == '__main__':
    config = get_config()
    print(config)
    main(config)