"""영인 작성 config"""

import argparse

def str2bool(v):
    return v.lower() in ('true')

parser = argparse.ArgumentParser()

# Dataloader configruations
parser.add_argument('--dataset', type=str, default='Portrait', help='dataset name. default=Portrait')
parser.add_argument('--dataroot', type=str, default='dataset_origin', help='path to dataset. default=dataset')
parser.add_argument('--tnr_transform_mode', type=int, default=0, choices=[0,1], help='mode to training transform. default=0')
parser.add_argument('--crop_size', type=int, default=256, help='crop size for the custom ImageFolder dataset. default=256')
parser.add_argument('--image_size', type=int, default=128, help='image resolution. default=128')
parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size. default=16')
parser.add_argument('--num_workers', type=int, default=1, help='num of workers for generating batch. default=1')

# Model configuration.
parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset). default=5')
parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G. default=64')
parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D. default=64')
parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G. default=6')
parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D. default=6')
parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss. default=1')
parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss. default=10')
parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty. default=10')

# Training configuration.
parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D. default=200000')
parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr. default=100000')
parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G. default=0.0001')
parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D. default=0.0001')
parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update. default=5')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer. default=0.999')
parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step. default=None')

# Test configuration.
parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step. default=200000')

# Miscellaneous.
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--use_tensorboard', type=str2bool, default=True)

# Directories.
parser.add_argument('--log_dir', type=str, default='stargan/logs')
parser.add_argument('--model_save_dir', type=str, default='stargan/models')
parser.add_argument('--sample_dir', type=str, default='stargan/samples')
parser.add_argument('--result_dir', type=str, default='stargan/results')

# Step size.
parser.add_argument('--log_step', type=int, default=10, help='step interval for printing log message. default=10')
parser.add_argument('--sample_step', type=int, default=1000, help='step interval for saving sample output images. default=1000')
parser.add_argument('--model_save_step', type=int, default=10000, help='step interval for saving checkpoint. default=10000')
parser.add_argument('--lr_update_step', type=int, default=1000, help='step interval for updating learning rate. default=1000')

# Convert configurations.
parser.add_argument('--dict_path', type=str, default='final.pth',  help='path to state_dict file for converting to Caffe2 file')

def get_config():
    config = parser.parse_args()
    return config