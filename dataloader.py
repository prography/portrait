'''
dataloader.py에는 데이터 디렉터리/ 데이터셋에서 데이터를 불러오는 코드를 작성.
별도의 작업 필요 없기 때문에 ImageFolder만으로 구현 가능.
'''
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader

import os, cv2, numpy

def get_loader(dataroot, crop_size, image_size, batch_size, num_workers=1):
    print("Preparing dataloader...")

    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.CenterCrop(crop_size),
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    test_transform = T.Compose([
        T.CenterCrop(crop_size),
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    train_path = os.path.join(dataroot, "train")
    test_path = os.path.join(dataroot, "test")

    train_set = ImageFolder(train_path, train_transform)
    test_set = ImageFolder(test_path, test_transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    print("Prepare dataloader completed!")
    return train_loader, test_loader

if __name__ == "__main__":
    dataroot = "data"

    # get_loader의 parameter 전달은 실제로는 config에서 받아오는 방식으로 구현
    train_loader, test_loader = get_loader(dataroot, crop_size=178, image_size=128, batch_size=1)

    label_dict = {0: "oil-painting", 1: "pointilism", 2: "real-face"}

    for idx, (image, label) in enumerate(train_loader):
        image = image.numpy()
        image = numpy.transpose(image[0], (1, 2, 0))
        b, g, r  = cv2.split(image)
        image = cv2.merge([r, g, b])
        cv2.imshow("image", image)
        cv2.waitKey(0)
        print(label_dict[label.item()])