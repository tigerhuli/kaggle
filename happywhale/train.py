from tqdm import tqdm
from dataset import HappyWhaleTrainDataset, HappyWhaleValidationDataset
from torch.utils.data import DataLoader
from net import HappyWhaleNet
from torch import nn
import torch
import matplotlib.pyplot as plt
import pandas as pd
import time

def train_loop(dataloader, model, loss_fn, optimizer, device):
    batchs = tqdm(dataloader, unit='batch')
    loss_sum = 0.0
    for x, y in batchs:
        preds = model(x.to(device))
        loss = loss_fn(preds, y.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batchs.set_postfix(loss=loss.item())
        loss_sum += loss.cpu().item()
    return loss_sum/len(batchs)

def validation_loop(dataloader, model, device):
    correct = 0.0
    for x, y in tqdm(dataloader):
        preds = model(x.to(device))
        correct += (preds.argmax(1) == y.to(device)).type(torch.float).sum().cpu().item()

    ave_correct = correct/len(dataloader.dataset)
    print(f'validation accuracy: {ave_correct}')
    return ave_correct

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    batch_size = 16
    learning_rate = 1e-3
    epochs = 30

    train_data = HappyWhaleTrainDataset()
    validation_data = HappyWhaleValidationDataset()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

    model = HappyWhaleNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)

    loss_list = []
    accuracy_list = []
    for epoch in range(epochs):
        print(f'start {epoch+1} epoch -------------------------------------------------------------------------------')
        time.sleep(0.01) # 防止tqdm打印错乱
        loss_list.append(train_loop(train_loader, model, loss_fn, optimizer, device))
        time.sleep(0.01) # 防止tqdm打印错乱
        accuracy_list.append(validation_loop(validation_loader, model, device))

    torch.save(model, 'models/net_1.model')
    plt.figure()
    plt.plot(loss_list)
    plt.show(block=False)

    plt.figure()
    plt.plot(accuracy_list)
    plt.show()

    train_data = pd.DataFrame({'loss': loss_list, 'accuracy': accuracy_list})
    train_data.to_csv('models/net_1_train.csv')
    print('done')
