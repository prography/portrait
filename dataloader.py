from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader

import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_loader(rootpath, tnr_transform_mode, image_size, crop_size, batch_size=128, num_workers=1):

    train_path = os.path.join(rootpath, 'train')
    test_path = os.path.join(rootpath, 'test')

    if tnr_transform_mode == 0:
        train_transform = T.Compose([
            # data augmentation
            # 1. random rotation
            # 2. random horizontal flip
            T.RandomRotation(45.0),
            T.RandomHorizontalFlip(),
            T.CenterCrop(crop_size),
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
        ])
    else:
        train_transform = T.Compose([
            # data augmentation
            # 1. random crop
            # 2. random vertical flip
            T.RandomCrop(crop_size),
            T.RandomVerticalFlip(),
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

    train_dataset = ImageFolder(train_path, transform=train_transform)
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