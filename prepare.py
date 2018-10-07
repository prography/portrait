"""영인 작성 Image Parsing Error 해결"""

import os
import PIL.Image as pil

from config import get_config
from dataloader import get_loader

if __name__ == '__main__':
    """abspath 함수 안에 dataset 경로 넣고, config의 dataroot을 해당 경로로 설정해주면 됨."""

    dataroot = os.path.abspath("dataset_origin")
    rm_cnt = 0
    for root, dirs, files in os.walk(dataroot):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                image = pil.open(file_path)
            except OSError:
                os.remove(file_path)
                print("Remove image!", file_path)
                rm_cnt += 1
    print("Remove %d images!" % rm_cnt)

    config = get_config()
    train_loader, test_loader = get_loader('dataset_origin', tnr_transform_mode=0,
                                           crop_size=config.crop_size, image_size=config.image_size,
                                           batch_size=config.batch_size, num_workers=config.num_workers)
    log_interval = int(len(train_loader)*0.1)
    for idx, (img, cls) in enumerate(train_loader):
        if (idx+1) % log_interval == 0:
            print("[%d/%d]" % (idx+1, len(train_loader)))