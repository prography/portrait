import os
from trainer import Trainer
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
    if config.dataset == 'ImageFolder':
        train_loader, test_loader = get_loader(config.dataroot, crop_size=config.crop_size, image_size=config.image_size,
                                batch_size=config.batch_size, num_workers=config.num_workers)

    trainer = Trainer(train_loader, config)

    if config.mode == 'train':
        trainer.train()
    elif config.mode == 'test':
        trainer.test()

if __name__ == '__main__':
    config = get_config()
    main(config)