import argparse

parser = argparse.ArgumentParser()

# Dataloader configuration
parser.add_argument('--dataset', type=str, default='ImageFolder')
parser.add_argument('--dataroot', type=str, default='./dataset', help='dataroot')
parser.add_argument('--crop_size', type=int, default=256, help='crop size for the custom ImageFolder dataset')
parser.add_argument('--image_size', type=int, default=128, help='image resolution')
parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
parser.add_argument('--num_workers', type=int, default=0)

# Model configuration.
parser.add_argument('--c_dim', type=int, default=4, help='dimension of domain labels (1st dataset)') # num of classes
parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')

# Training configuration.
parser.add_argument('--num_iters', type=int, default=10000, help='number of total iterations for training D, default=200000')
parser.add_argument('--num_iters_decay', type=int, default=5000, help='number of iterations for decaying lr')
parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

# Test configuration
parser.add_argument('--trained_G', type=str, default='', help='trained Generator model path')

# Miscellaneous.
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

# Directories.
parser.add_argument('--model_save_dir', type=str, default='stargan/models')
parser.add_argument('--sample_dir', type=str, default='stargan/samples')
parser.add_argument('--result_dir', type=str, default='stargan/results')

# Step size.
parser.add_argument('--log_step', type=int, default=50)
parser.add_argument('--sample_step', type=int, default=1000)
parser.add_argument('--model_save_step', type=int, default=2000)
parser.add_argument('--lr_update_step', type=int, default=1000)

def get_config():
    return parser.parse_args()