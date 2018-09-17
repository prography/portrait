'''
dataloader.py에는 데이터 디렉터리/ 데이터셋에서 데이터를 불러오는 코드를 작성.
별도의 작업 필요 없기 때문에 ImageFolder만으로 구현 가능.
'''
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

to_pil = ToPILImage()

import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# denormalize image
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

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
    train_idx2cls = {v: k for k, v in train_set.class_to_idx.items()}
    test_idx2cls = {v: k for k, v in test_set.class_to_idx.items()}

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    print("Prepare dataloader completed!")
    return train_loader, test_loader, train_idx2cls, test_idx2cls
    # return train_loader, train_idx2cls

if __name__ == "__main__":
    dataroot = r'D:\Deep_learning\Data\멘티_그려줘'

    # get_loader의 parameter 전달은 실제로는 main 함수에서 get_config() 통해 받아오는 방식으로 구현
    train_loader, test_loader, train_idx2cls, test_idx2cls = get_loader(dataroot, crop_size=178, image_size=128, batch_size=1)

    print("length of training dataloader", len(train_loader))
    print("length of testing dataloader", len(test_loader))

    for idx, (image, label) in enumerate(train_loader):
        # print(image.shape)
        image = to_pil(denorm(image.data[0]))
        # plt.imshow(image)
        # plt.show()
        print(train_idx2cls[label.item()])

    print("Training dataloader NO problem!")

    for idx, (image, label) in enumerate(test_loader):
        # print(image.shape)
        image = to_pil(denorm(image.data[0]))
        # plt.imshow(image)
        # plt.show()
        print(test_idx2cls[label.item()])

    print("Testing dataloader NO problem!")


