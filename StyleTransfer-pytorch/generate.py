"""
generate style-transferred image
"""
import torch
import torchvision.transforms as T
from model.nst import TransferNet
from utils import normalize_batch, save_image

import argparse, os
from PIL import Image

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--convert_mode', type=str, required=True, choices=['vector', 'water', 'oil'])
    parser.add_argument('--input_path', type=str, required=True, help='path to an image converted')
    parser.add_argument('--output_path', type=str, required=True, help='path to save converted image')

    config = parser.parse_args()
    return config

def convert(config):
    # get pretrained weight
    config.model_path = "weights/style_{}.pth".format(config.convert_mode)

    # load pretrained model
    net = TransferNet()
    net.load_state_dict(torch.load(config.model_path, map_location=lambda storage, loc: storage))
    net = net.to(device)

    # open input image file
    input_img = Image.open(config.input_path)

    # define transform
    transform = T.Compose([
        T.Resize(300),
        T.ToTensor(),
        T.Lambda(lambda x: x.mul(255))
    ]
    )

    # convert input image file to tensor
    input_img = transform(input_img)
    input_img = input_img.unsqueeze(0).to(device)

    # forwarding to Transfer network
    output_img = net(input_img)
    output_img = normalize_batch(output_img)

    # save output image
    output_img = output_img.cpu()[0]
    save_image(config.output_path, output_img)

def main():
    config = get_config()
    convert(config)

if __name__ == '__main__':
    main()