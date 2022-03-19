import torch.nn as nn
import torch.nn.functional as F
import torch


class HappyWhaleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(3, 6, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 6, 3, padding='same'),
            nn.ReLU(),
        )

        self.linear = nn.Linear(64*64*6, 15587)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logics = self.linear(x)
        return logics
