import argparse
import os
import random

import numpy as np
import torch

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

    # Create directories if not exist.
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
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--eps', type=float, default=0.05, help='epsilon for perturbation')
    parser.add_argument('--order', type=int, default=2, help='distance metric')
    
    # Training configuration.
    parser.add_argument('--seed', type=int, default=0, help='seed for experiments')
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=30, help='number of total epochs for training P')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for G')
    parser.add_argument('--beta1', type=float, default=0.99, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume', default=False, action='store_true', help='resume training from last epoch')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    parser.add_argument('--alpha', type=float, default=0.12, help="alpha for gradnorm")

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=48)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--disable_tensorboard', action='store_true', default=False)

    # Directories.
    parser.add_argument('--gen_ckpt', type=str,
                        default='stargan/stargan_celeba_128/models/200000-G.ckpt')
    parser.add_argument('--detector_path', type=str,
                        default='detection/detector_c23.pth')
    parser.add_argument('--celeba_image_dir', type=str, default='stargan/data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='stargan/data/celeba/list_attr_celeba.txt')
    parser.add_argument('--log_dir', type=str, default='stargan/logs')
    parser.add_argument('--model_save_dir', type=str,
                        default='stargan/perturbation_models')
    parser.add_argument('--sample_dir', type=str, default='stargan/samples')
    parser.add_argument('--result_dir', type=str, default='stargan/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--sample_step', type=int, default=5)

    config = parser.parse_args()
    print(config)
    main(config)