"""완수 작성 dataloader"""

from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader

import os


def get_loader(rootpath, image_size, crop_size, batch_size=128, num_workers=1):

    train_path = os.path.join(rootpath, 'train')
    test_path = os.path.join(rootpath, 'test')

    train_traonsform = T.Compose([
        # TODO Augmentation 요수 추가
        T.CenterCrop(crop_size),
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5))
    ])

    test_transform = T.Compose([
        T.CenterCrop(crop_size),
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5))
    ])

    train_dataset = ImageFolder(train_path, transform=train_traonsform)
    test_dataset = ImageFolder(test_path, transform=test_transform)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, test_loader