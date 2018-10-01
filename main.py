"""영인 작성 main"""

import os
from train import trainer
from dataloader import get_loader
from torch.backends import cudnn
from config import get_config

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # our case
    if config.dataset == 'Portrait':
        train_loader, test_loader = get_loader(config.dataroot, config.tnr_transform_mode,
                                               crop_size=config.crop_size, image_size=config.image_size,
                                               batch_size=config.batch_size, num_workers=config.num_workers)

    tnr = trainer(train_loader, test_loader, config)

    if config.mode == 'train':
        tnr.train()
    elif config.mode == 'test':
        tnr.test()

if __name__ == '__main__':
    config = get_config()
    main(config)