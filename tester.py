from model import Generator
from torchvision.utils import save_image
import torch
import numpy as np
import os

class Tester(object):
    def __init__(self, test_loader, config):
        self.test_loader = test_loader
        self.netG_path = config.trained_G
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.g_conv_dim = config.g_conv_dim
        self.c_dim = config.c_dim
        self.g_repeat_num = config.g_repeat_num
        self.result_dir = config.result_dir

        self.load_model()

    def load_model(self):
        print("Loading trained model...")
        G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        G.load_state_dict(torch.load(self.netG_path, map_location=lambda storage, loc: storage))
        self.G = G.to(self.device)

        self.G.eval() # switch model to eval mode

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5):
        """Generate target domain labels for debugging and testing."""
        c_trg_list = []
        for i in range(c_dim):
            c_trg = self.label2onehot(torch.ones(c_org.size(0)) * i, c_dim)
            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(self.test_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i + 1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))