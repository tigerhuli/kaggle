import os
import pandas as pd
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


class HappyWhaleTrainDataset(Dataset):
    def __init__(self, annotations_file=None, img_dir=None, transform=None, target_transform=None):
        data = pd.read_csv('data/output/train_set.csv')
        self.img_labels = data['individual_num']
        self.img_dir = 'data/output/archive/train_images_256'
        self.img_names = data['image']

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_name = self.img_names[idx].replace('jpg', 'png')
        img_path = os.path.join(self.img_dir, image_name)
        image = Image.open(img_path)
        image = transforms.ToTensor()(image)
        label = self.img_labels[idx]
        return image, label

class HappyWhaleValidationDataset(Dataset):
    def __init__(self, annotations_file=None, img_dir=None, transform=None, target_transform=None):
        data = pd.read_csv('data/output/validation_set.csv')
        self.img_labels = data['individual_num']
        self.img_dir = 'data/output/archive/train_images_256'
        self.img_names = data['image']

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_name = self.img_names[idx].replace('jpg', 'png')
        img_path = os.path.join(self.img_dir, image_name)
        image = Image.open(img_path)
        image = transforms.ToTensor()(image)
        label = self.img_labels[idx]
        return image, label


if __name__ == '__main__':
    train_data = HappyWhaleTrainDataset()
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    train_imgs, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_imgs.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    img = train_imgs[0].squeeze()
    label = train_labels[0]

    plt.imshow(img, cmap='gray')
    plt.show()
    print(f"Label: {label}")
