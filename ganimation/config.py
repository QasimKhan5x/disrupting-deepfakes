import argparse


def get_config(args=None):
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=17,
                        help='dimension of domain labels')
    parser.add_argument('--image_size', type=int,
                        default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64,
                        help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64,
                        help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6,
                        help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6,
                        help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=160,
                        help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10,
                        help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10,
                        help='weight for gradient penalty')
    parser.add_argument('--lambda_sat', type=float, default=0.1,
                        help='weight for attention saturation loss')
    parser.add_argument('--lambda_smooth', type=float, default=1e-4,
                        help='weight for the attention smoothing loss')
    parser.add_argument('--eps', type=float, default=0.05, help='epsilon for perturbation')
    parser.add_argument('--order', type=int, default=2, help='distance metric')
    
    # Training configuration
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for experiments')
    parser.add_argument('--dataset', type=str, default='CelebA',
                        choices=['CelebA', 'RaFD', 'Both'])
    parser.add_argument('--batch_size', type=int,
                        default=32, help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of total epochs for training P')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate for G')
    parser.add_argument('--beta1', type=float, default=0.99,
                        help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='beta2 for Adam optimizer')
    parser.add_argument('--resume', default=False,
                        action='store_true', help='resume training from last epoch')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help="alpha for gradnorm")
    parser.add_argument('--detector', type=str, default='xception', choices=['xception', 'resnet18', 'resnet50'])

    
    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=48)
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'animation'])
    parser.add_argument('--disable_tensorboard',
                        action='store_true', default=False)
    parser.add_argument('--num_sample_targets', type=int, default=4,
                        help="number of targets to use in the samples visualization")

    # Directories.
    parser.add_argument('--gen_ckpt', type=str,
                        default='ganimation/7001-37-G.ckpt')
    parser.add_argument('--detector_path', type=str,
                        default='detection/detector_c23.pth')
    parser.add_argument('--image_dir', type=str,
                        default='ganimation/data/celeba/images_aligned')
    parser.add_argument('--attr_path', type=str,
                        default='ganimation/data/celeba/list_attr_celeba.txt')
    parser.add_argument('--outputs_dir', type=str, default='experiment1')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--model_save_dir', type=str, default='models')
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--result_dir', type=str, default='results')

    parser.add_argument('--animation_images_dir', type=str,
                        default='data/celeba/images_aligned/new_small')
    parser.add_argument('--animation_attribute_images_dir', type=str,
                        default='animations/eric_andre/attribute_images')
    parser.add_argument('--animation_attributes_path', type=str,
                        default='animations/eric_andre/attributes.txt')
    parser.add_argument('--animation_models_dir', type=str,
                        default='models')
    parser.add_argument('--animation_results_dir', type=str,
                        default='out')
    parser.add_argument('--animation_mode', type=str, default='animate_image',
                        choices=['animate_image', 'animate_random_batch'])

    # Step size.
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--sample_step', type=int, default=1)

    if args is []:
        config = parser.parse_args(args=[])
    else:
        config = parser.parse_args()
    return config


def str2bool(v):
    return v.lower() in ('true')
