from net import HappyWhaleNet
import torch
import os
from torchvision import transforms
from PIL import Image, ImageOps
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HappyWhaleNet().to(device)
    model = torch.load('models/net_1.pt')

    id_map_data = pd.read_csv('data/id_to_num_map.csv')
    num_2_id = {}
    for id, num in zip(id_map_data['id'], id_map_data['num']):
        num_2_id[int(num)] = id

    model.eval()
    file_dir = 'data/test_images_256'
    image_names = []
    image_labels = []
    for root, _, files in os.walk(file_dir):
        for name in tqdm(files):
            image_names.append(name)
            file_path = os.path.join(root, name)
            img = Image.open(file_path)
            img = ImageOps.grayscale(img)
            img_tensor = transforms.ToTensor()(img).to(device).unsqueeze(0)
            logics = model(img_tensor)[0]
            if torch.max(logics) < 0.2:
                image_labels.append('new_individual')
                continue

            _, nums = torch.topk(logics, 5)
            label = ''
            splitor = ''
            for num in nums.cpu().numpy():
                id = num_2_id[num]
                label = label+splitor+id
                splitor = ' '
            image_labels.append(label)

    prediction_data = pd.DataFrame({'image': image_names, 'predictions': image_labels})
    prediction_data.to_csv('data/1_submission.csv', index=False)
