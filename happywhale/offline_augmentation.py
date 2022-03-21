from PIL import Image, ImageOps
import os
from tqdm import tqdm


def resize_images(in_dir, out_dir):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir, 0o666)
    for root, _, files in os.walk(in_dir):
        for file in tqdm(files):
            img = Image.open(os.path.join(root, file))
            img = img.resize((256, 256))
            if img.mode == 'L':
                img = ImageOps.colorize(img, black='black', white='white')
            img.save(os.path.join(out_dir, file))


if __name__ == '__main__':
    test_in_dir = 'data/archive/seg_img_test'
    test_out_dir = 'data/output/archive/test_images_256'
    train_in_dir = 'data/archive/seg_img'
    train_out_dir = 'data/output/archive/train_images_256'

    resize_images(train_in_dir, train_out_dir)
    # resize_images(test_in_dir, test_out_dir)
