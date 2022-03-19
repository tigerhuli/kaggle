import pandas as pd
import time

if __name__ == '__main__':
    file_path = 'data/train_num.csv'
    data = pd.read_csv(file_path)

    imgs = data['image']
    nums = data['individual_num']

    num2img = {}
    img2num = {}

    for img, num in zip(imgs, nums):
        img2num[img] = num
        num2img[num] = img

    validation_imgs = list(num2img.values())[0:100]
    train_imgs = [img for img in imgs if (img not in validation_imgs)]

    print(f'validation images num {len(validation_imgs)}')
    print(f'train images num {len(train_imgs)}')
    print(f'total images num {len(imgs)}')
    print(f'validation images + train images num {len(validation_imgs)+len(train_imgs)}')

    validation_nums = []
    for img in validation_imgs:
        validation_nums.append(img2num[img])
    validation_data = pd.DataFrame({'image':validation_imgs, 'individual_num':validation_nums})
    validation_data.to_csv('data/validation_set.csv', index=False)

    train_nums = []
    for img in train_imgs:
        train_nums.append(img2num[img])
    train_data = pd.DataFrame({'image':train_imgs, 'individual_num':train_nums})
    train_data.to_csv('data/train_set.csv', index=False)
