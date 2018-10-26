"""
Convert Pytorch model(.pth or .ckpt) to Caffe2(.pb) file
"""

import torch
import torch.nn as nn
import torch.onnx as onnx
from torch.autograd import Variable

from model import Generator
from config import get_config

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConvLayer(nn.Module):
    def __init__(self, inch, outch, kernel_size, stride):
       super(ConvLayer, self).__init__()

       rp = kernel_size // 2
       self.rp = nn.ReflectionPad2d(rp)
       self.conv = nn.Conv2d(inch, outch, kernel_size, stride)

    def forward(self, x):
        out = self.rp(x)
        out = self.conv(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super(ResidualBlock, self).__init__()

        self.conv1 = ConvLayer(ch, ch, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(ch, affine=True)
        self.conv2 = ConvLayer(ch, ch, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(ch, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

class UpsampleConvLayer(nn.Module):
    def __init__(self, inch, outch, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()

        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(mode='nearest', scale_factor=upsample)

        rp = kernel_size // 2
        self.rp = nn.ReflectionPad2d(rp)
        self.conv2d = nn.Conv2d(inch, outch, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        out = self.rp(x)
        out = self.conv2d(out)
        return out

class TransferNet(nn.Module):
    def __init__(self):
        super(TransferNet, self).__init__()

        # setup conv layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)

        # setup residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # setup upsampling layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = UpsampleConvLayer(32, 3, kernel_size=9, stride=1)

        # setup activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        input: 1, 3, 300, 300
        conv output: 1, 128, 75, 75
        residual output: 1, 128, 75, 75
        deconv output: 1, 3, 300, 300
        """
        h = self.relu(self.in1(self.conv1(x)))
        h = self.relu(self.in2(self.conv2(h)))
        h = self.relu(self.in3(self.conv3(h)))

        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)

        h = self.relu(self.in4(self.deconv1(h)))
        h = self.relu(self.in5(self.deconv2(h)))
        out = self.deconv3(h)
        return out

def convert(config):
    # testing for small network
    # net = torch.nn.Linear(10, 2)
    # d = torch.rand(10, requires_grad=True)
    # onnx.export(net, d, "converted.onnx", export_params=True)

    # testing for other networks
    tf = TransferNet()
    d = torch.randn(1, 3, 300, 300, requires_grad=True)
    print(tf)
    onnx.export(tf, d, "converted.onnx", export_params=True)
    print("Complete!")

    # conversion our model
    netG = Generator(config.g_conv_dim, config.c_dim, config.g_repeat_num).to(device)
    netG.load_state_dict(torch.load(config.dict_path, map_location=lambda storage, loc: storage))
    print(netG)

    dummy_img = torch.randn(1, 3, config.image_size, config.image_size, requires_grad=True)
    dummy_label = torch.randn(1, 5, requires_grad=True)
    converted = onnx._export(netG, (dummy_img, dummy_label), "Generator-onnx.onnx", export_params=True)
    print("[*] Converting completed!")

if __name__ == '__main__':
    config = get_config()
    convert(config)
