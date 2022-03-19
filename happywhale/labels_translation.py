import pandas as pd

if __name__ == '__main__':
    file_name = './data/train.csv'
    data = pd.read_csv(file_name)
    ids = data['individual_id']

    ids_set = set(ids)
    id2num = {}
    for i, id in enumerate(ids_set):
        id2num[id] = i
    nums = [id2num[id] for id in ids]

    img_names = data['image']

    translation_data = pd.DataFrame({'image': img_names, 'individual_id': ids, 'individual_num': nums})
    translation_data.to_csv('data/train_num.csv', index=False)

    a = {'id': list(id2num.keys()), 'num': list(id2num.values())}
    id2num_data = pd.DataFrame(a)
    id2num_data.to_csv('data/id_to_num_map.csv', index=False)
