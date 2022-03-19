from matplotlib import image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from sqlalchemy import null
from torchvision.io import read_image
import os


def analysis_labels():
    # 分析标签数据
    file = 'data_analysis/train.csv'
    data = pd.read_csv(file)

    species = data['species']
    print(f'number of samples: {len(species)}')
    print(f'number of species: {len(np.unique(species))}')

    ids = data['individual_id']
    print(f'number of ids: {len(np.unique(ids))}')

    ids_num = {}
    id_num_max = 0
    for id in ids:
        ids_num[id] = ids_num.get(id, 0)+1
        id_num_max = max(id_num_max, ids_num[id])

    print(id_num_max)
    plt.figure()
    plt.bar(ids_num.keys(), ids_num.values())
    plt.axis('off')

    specie_num = {}
    specie_num_max = 0
    for specie in species:
        specie_num[specie] = specie_num.get(specie, 0)+1
        specie_num_max = max(specie_num_max, specie_num[specie])

    print(specie_num_max)
    plt.figure()
    plt.bar(specie_num.keys(), specie_num.values())
    plt.xticks(rotation=-90)

    plt.show()


def print_image_size():
    # 打印图像尺寸
    img = Image.open('data_analysis/000a8f2d5c316a.jpg')
    # img.show()
    print(img.size)
    img = img.resize((256, 256))
    print(img.size)
    img.show()


def print_image_type():
    # 打印图像类型
    img = read_image('data_analysis/000a8f2d5c316a.jpg')
    print(type(img))


def find_gray_image():
    # 查找灰度图像
    images_dir = 'data/train_images'
    for root, _, files in os.walk(images_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            img = Image.open(file_path)
            if img.mode == 'rgb' or img.mode == 'RGB':
                continue

            print(file_name)
            print(img.mode)
            break


def gray_image_to_color_image():
    # 灰度图像转彩色图像
    img_path = 'data/train_images/00398cfc6f6675.jpg'
    img = Image.open(img_path)

    img = ImageOps.colorize(img, black='black', white='white')
    img.show()
    print(img.mode)

def show_one_class_images():
    # 展示一类图像
    data = pd.read_csv('data/train.csv')
    images = data['image']
    labels = data['individual_id']
    id2images = {}

    for image, label in zip(images, labels):
        if not label in id2images:
            id2images[label] = []
        id2images[label].append(image)

    for id, images in id2images.items():
        if len(images) < 10:
            continue

        print(f"for image label: {id}, images: {len(images)} ")
        for image in images:
            img = Image.open(os.path.join('data/train_images', image))
            img.show()
        break

if __name__ == '__main__':
    # analysis_labels()
    # analysis_image()
    # analysis_image2()
    # analysis_image3()
    gray_image_to_color_image()
    # analysis_image4()
