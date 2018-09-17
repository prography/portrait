import os, shutil

def prepare_dataset(dataroot):
    pass

if __name__ == "__main__":
    dataroot = "D:\Deep_learning\Data\멘티_그려줘\oil-painting".replace('\\', '/')
    dirs = [os.path.join(dataroot, dir) for dir in os.listdir(dataroot)]
    os.makedirs(os.path.join(dataroot, 'train'), exist_ok=True)
    for dir in dirs:
        for file in os.listdir(dir):
            full_src = os.path.join(os.path.join(dir, file))
            full_dest = os.path.join(dataroot, 'train', dir.split('\\')[-1] + file)
            # print(full_src)
            # print(full_dest)
            # print()
            shutil.copy(full_src, full_dest)
    print("copy completed!")