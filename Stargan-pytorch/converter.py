"""
Convert Pytorch model(.pth or .ckpt) to Caffe2(.pb) file
"""

import torch
import torch.onnx as onnx
from torch.autograd import Variable

from model import Generator
from config import get_config

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def convert(config):
    netG = Generator(config.g_conv_dim, config.c_dim, config.g_repeat_num).to(device)
    netG.load_state_dict(torch.load(config.dict_path, map_location=lambda storage, loc: storage))
    netG.eval()

    dummy_img = Variable(torch.randn(1, 3, config.image_size, config.image_size))
    dummy_label = Variable(torch.randn(1, 5))
    converted = onnx.export(netG, (dummy_img, dummy_label), "Generator-onnx.onnx")
    print("[*] Converting completed!")


if __name__ == '__main__':
    config = get_config()
    convert(config)
